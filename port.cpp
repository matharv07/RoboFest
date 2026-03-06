#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <deque>
#include <algorithm>
#include <iio.h>
#include <ad9361.h>

namespace fs = std::filesystem;
typedef std::complex<float> cf32;

// ───────────── Constants ─────────────
const double FS = 8e6;
const size_t NUM_S = 262144;
const double TONE = 543e3;
const double EPS = 1e-15;
const int SMOOTH_WIN = 15;

// ───────────── DSP Functions ─────────────
double to_dB(float mag) {
    return 20.0 * std::log10(std::max(mag, (float)EPS));
}

float lockin(const std::vector<cf32>& buf) {
    if (buf.empty()) return 0.0f;
    cf32 acc(0, 0);
    for (size_t i = 0; i < buf.size(); ++i) {
        float phase = -2.0f * M_PI * TONE * (i / FS);
        acc += buf[i] * std::polar(1.0f, phase);
    }
    return std::abs(acc) / buf.size();
}

double apply_gaussian_point(const std::deque<double>& history, int window) {
    if (history.size() < (size_t)window) return history.back();
    double std_dev = std::max(window / 6.0, 1.0);
    double weight_sum = 0, val_sum = 0;
    int center = window / 2;
    for (int i = 0; i < window; ++i) {
        int idx = history.size() - window + i;
        double weight = std::exp(-0.5 * std::pow((double)i - center, 2) / (2 * std_dev * std_dev));
        val_sum += history[idx] * weight;
        weight_sum += weight;
    }
    return val_sum / weight_sum;
}

// ───────────── Main Program ─────────────
int main(int argc, char** argv) {
    // UPDATED DEFAULTS: 2000 MHz, OBJ: NO, Height: 1.25m
    long long freq_mhz = (argc > 1) ? std::stoll(argv[1]) : 2000;
    std::string obj_status = (argc > 2) ? argv[2] : "NO";
    double height_m = (argc > 3) ? std::stod(argv[3]) : 1.25;

    long long center_freq = freq_mhz * 1000000;

    // 1. Directory & File Setup
    std::string folder = "VKA/" + std::to_string(freq_mhz) + "MHz";
    fs::create_directories(folder);
    
    auto now_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss; ss << std::put_time(std::localtime(&now_t), "%d-%H%M%S");
    std::string csv_path = folder + "/scan_" + ss.str() + ".csv";

    std::ofstream csv(csv_path);
    csv << "Time,Frequency,S11_Raw,S21_Raw,S11_Smooth,S21_Smooth,OBJ,Height\n";

    // 2. Hardware Init
    iio_context *ctx = iio_create_network_context("192.168.2.1");
    if (!ctx) { std::cerr << "SDR not found at ip:192.168.2.1\n"; return -1; }
    
    iio_device *phy = iio_context_find_device(ctx, "ad9361-phy");
    iio_device *rx_dev = iio_context_find_device(ctx, "cf-ad9361-lpc");

    iio_channel_attr_write_longlong(iio_device_find_channel(phy, "altvoltage0", true), "frequency", center_freq);
    iio_channel *rx0 = iio_device_find_channel(rx_dev, "voltage0", false);
    iio_channel *rx1 = iio_device_find_channel(rx_dev, "voltage1", false);
    iio_channel_enable(rx0); iio_channel_enable(rx1);

    iio_buffer *rxbuf = iio_device_create_buffer(rx_dev, NUM_S, false);

    // 3. Main Loop
    std::deque<double> h11, h21;
    auto start_time = std::chrono::steady_clock::now();

    std::cout << "--- VNA Scanning (Headless) ---\n";
    std::cout << "Target: " << freq_mhz << " MHz | OBJ: " << obj_status << " | H: " << height_m << "m\n";
    std::cout << "Saving to: " << csv_path << "\n";

    while (true) {
        if (iio_buffer_refill(rxbuf) < 0) break;

        std::vector<cf32> s0, s1;
        char *p_dat, *p_end = (char *)iio_buffer_end(rxbuf);
        ptrdiff_t p_inc = iio_buffer_step(rxbuf);

        for (p_dat = (char *)iio_buffer_first(rxbuf, rx0); p_dat < p_end; p_dat += p_inc)
            s0.push_back(cf32(((int16_t *)p_dat)[0], ((int16_t *)p_dat)[1]) / 2048.0f);
        for (p_dat = (char *)iio_buffer_first(rxbuf, rx1); p_dat < p_end; p_dat += p_inc)
            s1.push_back(cf32(((int16_t *)p_dat)[0], ((int16_t *)p_dat)[1]) / 2048.0f);

        double s21_raw = to_dB(lockin(s0));
        double s11_raw = to_dB(lockin(s1));
        if (s11_raw > 0) s11_raw = 0.0;

        h21.push_back(s21_raw); h11.push_back(s11_raw);
        if (h21.size() > 50) { h21.pop_front(); h11.pop_front(); }

        double s21_smooth = apply_gaussian_point(h21, SMOOTH_WIN);
        double s11_smooth = apply_gaussian_point(h11, SMOOTH_WIN);
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();

        csv << elapsed << "," << center_freq << "," << s11_raw << "," << s21_raw << "," 
            << s11_smooth << "," << s21_smooth << "," << obj_status << "," << height_m << "\n";

        // Terminal feedback
        std::cout << "\r[" << std::fixed << std::setprecision(1) << elapsed << "s] "
                  << "S21: " << std::setprecision(2) << s21_smooth << " dB " << std::flush;
        
        // Safety flush every 20 samples
        static int f_count = 0;
        if (++f_count % 20 == 0) csv.flush();
    }

    iio_buffer_destroy(rxbuf);
    iio_context_destroy(ctx);
    return 0;
}
