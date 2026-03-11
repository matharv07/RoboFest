#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <omp.h>
#include <iio.h>
#include <string>
#include <ctime>
#include <algorithm>

namespace fs = std::filesystem;
typedef std::complex<float> cf32;

// ───────────── Configuration ─────────────
const double FS           = 8e6;
const size_t NUM_S        = 131072;
const double TONE         = 543e3;
const double EPS          = 1e-15;
const double CAL_REF_DBFS = 1.8;   // ⚠️ Update with your direct-cable calibration value

struct SdrData {
    std::vector<cf32> rx_complex;
    double timestamp;
};

// Threading & Optimization Globals
std::deque<SdrData>     data_queue;
std::mutex              queue_mutex;
std::condition_variable queue_cv;
std::atomic<bool>       keep_running{true};
std::vector<cf32>       PRECOMPUTED_OSC;

// ───────────── Mathematical Helpers ─────────────
std::vector<double> generate_gaussian_kernel(int window_size, double sigma) {
    std::vector<double> coeffs(window_size);
    double sum = 0.0;
    int half = window_size / 2;
    for (int i = 0; i < window_size; ++i) {
        int x = i - half;
        coeffs[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += coeffs[i];
    }
    for (int i = 0; i < window_size; ++i) coeffs[i] /= sum;
    return coeffs;
}

float lockin_omp(const std::vector<cf32>& buf) {
    if (buf.empty()) return 0.0f;
    float real_acc = 0.0f, imag_acc = 0.0f;
    int n = static_cast<int>(buf.size());
    for (int i = 0; i < n; ++i) {
        cf32 mixed = buf[i] * PRECOMPUTED_OSC[i];
        real_acc += mixed.real();
        imag_acc += mixed.imag();
    }
    return std::sqrt(real_acc * real_acc + imag_acc * imag_acc) / n;
}

// ───────────── Acquisition Thread ─────────────
void acquisition_thread(const std::string& ip, long long freq) {
    std::cout << "[Acq] Connecting to Pluto at " << ip << "..." << std::endl;
    iio_context *ctx = iio_create_network_context(ip.c_str());
    if (!ctx) {
        std::cerr << "[Fatal] Could not connect to SDR." << std::endl;
        keep_running = false;
        return;
    }

    iio_device *phy    = iio_context_find_device(ctx, "ad9361-phy");
    iio_device *rx_dev = iio_context_find_device(ctx, "cf-ad9361-lpc");
    iio_device *tx_dev = iio_context_find_device(ctx, "cf-ad9361-dds-core-lpc");

    if (!phy || !rx_dev || !tx_dev) {
        std::cerr << "[Fatal] Hardware devices not found." << std::endl;
        keep_running = false;
        return;
    }

    iio_channel *rx_i    = iio_device_find_channel(rx_dev, "voltage0", false);
    iio_channel *rx_q    = iio_device_find_channel(rx_dev, "voltage1", false);
    iio_channel *rx_lo   = iio_device_find_channel(phy,    "altvoltage0", true);
    iio_channel *tx_i    = iio_device_find_channel(tx_dev, "voltage0", true);
    iio_channel *tx_q    = iio_device_find_channel(tx_dev, "voltage1", true);
    iio_channel *tx_lo   = iio_device_find_channel(phy,    "altvoltage1", true);
    iio_channel *tx_phy  = iio_device_find_channel(phy,    "voltage0", true);
    iio_channel *rx_phy  = iio_device_find_channel(phy,    "voltage0", false);

    if (rx_lo)  iio_channel_attr_write_longlong(rx_lo,  "frequency", freq);
    if (tx_lo)  iio_channel_attr_write_longlong(tx_lo,  "frequency", freq);
    if (tx_phy) iio_channel_attr_write_double(tx_phy,   "hardwaregain", -10.0);
    if (rx_phy) {
        iio_channel_attr_write(rx_phy, "gain_control_mode", "manual");
        iio_channel_attr_write_double(rx_phy, "hardwaregain", 30.0); // ⚠️ 5.0 for cable, 30.0+ for antenna
    }

    iio_channel_enable(rx_i); iio_channel_enable(rx_q);
    iio_channel_enable(tx_i); iio_channel_enable(tx_q);

    // Push cyclic TX tone
    iio_buffer *txbuf = iio_device_create_buffer(tx_dev, 8000, true);
    if (!txbuf) {
        std::cerr << "[Fatal] TX buffer creation failed." << std::endl;
        keep_running = false;
        return;
    }
    char    *p_start = (char *)iio_buffer_first(txbuf, tx_i);
    ptrdiff_t p_inc  = iio_buffer_step(txbuf);
    for (int i = 0; i < 8000; ++i) {
        float t = (float)i / FS;
        ((int16_t*)(p_start + i * p_inc))[0] = (int16_t)(1432.0f * std::cos(2.0f * M_PI * TONE * t));
        ((int16_t*)(p_start + i * p_inc))[1] = (int16_t)(1432.0f * std::sin(2.0f * M_PI * TONE * t));
    }
    iio_buffer_push(txbuf);
    std::cout << "[Acq] TX cyclic buffer pushed (" << TONE / 1000 << " kHz, -10 dB)." << std::endl;

    iio_buffer *rxbuf = iio_device_create_buffer(rx_dev, NUM_S, false);
    if (!rxbuf) {
        std::cerr << "[Fatal] RX buffer creation failed." << std::endl;
        keep_running = false;
        return;
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "[Acq] Hardware ready. Starting capture." << std::endl;

    while (keep_running) {
        if (iio_buffer_refill(rxbuf) < 0) break;

        SdrData data;
        data.timestamp = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        data.rx_complex.resize(NUM_S);

        char    *b0    = (char *)iio_buffer_first(rxbuf, rx_i);
        ptrdiff_t r_inc = iio_buffer_step(rxbuf);
        for (int i = 0; i < (int)NUM_S; ++i) {
            float iv = ((int16_t *)(b0 + i * r_inc))[0] / 2048.0f;
            float qv = ((int16_t *)(b0 + i * r_inc))[1] / 2048.0f;
            data.rx_complex[i] = cf32(iv, qv);
        }

        std::lock_guard<std::mutex> lock(queue_mutex);
        if (data_queue.size() > 10) data_queue.pop_front();
        data_queue.push_back(std::move(data));
        queue_cv.notify_one();
    }

    iio_buffer_destroy(rxbuf);
    iio_buffer_destroy(txbuf);
    iio_context_destroy(ctx);
    std::cout << "\n[Acq] Hardware released." << std::endl;
}

// ───────────── Processing Thread ─────────────
void processing_thread(long long freq_mhz, std::string obj, double height) {
    // Setup output file
    std::string folder = "VCA/" + std::to_string(freq_mhz) + "MHz";
    fs::create_directories(folder);
    std::time_t rawtime; std::time(&rawtime);
    char t_buf[20]; std::strftime(t_buf, 20, "%d-%m-%H%M%S", std::localtime(&rawtime));
    std::string csv_path = folder + "/scan_cpp-" + std::string(t_buf) + ".csv";
    std::ofstream csv(csv_path);
    csv << "Time,Raw_dBFS,S21_dB,S21_Gaussian,S21_SavGol,Baseline,Diff_dB,Detect_Flag,Minima_Flag,OBJ,Height\n";

    // ── DSP parameters ──
    const int    GAUSS_WIN     = 15;
    const double GAUSS_SIGMA   = 2.0;
    const int    SAVGOL_WIN    = 21;
    const int    MEDIAN_WIN    = 100;   // rolling baseline window (samples)
    const int    BASELINE_MIN  = 30;    // minimum samples before baseline is trusted
    const int    SG_HIST_WIN   = 5;     // window for local minima detection
    const double DETECT_THRESH = 8.0;   // dB — abs(SG - baseline) to trigger detection

    std::vector<double> GAUSS_COEFFS = generate_gaussian_kernel(GAUSS_WIN, GAUSS_SIGMA);
    const double SG_COEFFS[21] = {
        -0.0559006, -0.0248447,  0.0029421,  0.0274600,  0.0487087,
         0.0666885,  0.0813991,  0.0928408,  0.1010134,  0.1059170,
         0.1075515,  0.1059170,  0.1010134,  0.0928408,  0.0813991,
         0.0666885,  0.0487087,  0.0274600,  0.0029421, -0.0248447,
        -0.0559006
    };

    struct RawPoint   { double t, dbfs, s21; };
    struct GaussPoint { double t, dbfs, s21, g_val; };

    std::deque<RawPoint>   r_buf;
    std::deque<GaussPoint> g_buf;
    std::deque<double>     baseline_buf;  // only clean (non-triggered) samples
    std::deque<double>     sg_history;    // last SG_HIST_WIN SG values for minima detection

    bool   is_detecting = false;
    double baseline     = 0.0;

    // Pipeline flush helper — called after keep_running goes false
    auto flush_pipeline = [&](double pad_dbfs, double pad_s21, double last_t) {
        const int flush_count = (GAUSS_WIN / 2) + (SAVGOL_WIN / 2);
        const double time_step = static_cast<double>(NUM_S) / FS;

        for (int f = 0; f < flush_count; ++f) {
            last_t += time_step;
            r_buf.push_back({last_t, pad_dbfs, pad_s21});
            if (r_buf.size() > (size_t)GAUSS_WIN) r_buf.pop_front();

            if (r_buf.size() == (size_t)GAUSS_WIN) {
                double g_val = 0.0;
                for (int i = 0; i < GAUSS_WIN; ++i) g_val += r_buf[i].s21 * GAUSS_COEFFS[i];
                const auto& mid = r_buf[GAUSS_WIN / 2];
                g_buf.push_back({mid.t, mid.dbfs, mid.s21, g_val});
                if (g_buf.size() > (size_t)SAVGOL_WIN) g_buf.pop_front();

                if (g_buf.size() == (size_t)SAVGOL_WIN) {
                    double sg = 0.0;
                    for (int i = 0; i < SAVGOL_WIN; ++i) sg += g_buf[i].g_val * SG_COEFFS[i];
                    double diff    = std::abs(sg - baseline);
                    bool triggered = (diff > DETECT_THRESH) && (baseline_buf.size() >= (size_t)BASELINE_MIN);
                    const auto& pt = g_buf[SAVGOL_WIN / 2];
                    csv << pt.t << "," << pt.dbfs << "," << pt.s21 << "," << pt.g_val << ","
                        << sg << "," << baseline << "," << diff << ","
                        << (triggered ? 1 : 0) << ",0," << obj << "," << height << "\n";
                }
            }
        }
    };

    std::cout << "[Proc] Logging to " << csv_path << std::endl;

    double last_dbfs = 0.0, last_s21 = 0.0, last_t = 0.0;

    while (keep_running || !data_queue.empty()) {
        SdrData data;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait_for(lock, std::chrono::milliseconds(100),
                              []{ return !data_queue.empty(); });
            if (data_queue.empty()) continue;
            data = std::move(data_queue.front());
            data_queue.pop_front();
        }

        double raw_dbfs = 20.0 * std::log10(std::max((double)lockin_omp(data.rx_complex), EPS));
        double s21_db   = raw_dbfs - CAL_REF_DBFS;
        last_dbfs = raw_dbfs; last_s21 = s21_db; last_t = data.timestamp;

        // ── Stage 1: Gaussian ──
        r_buf.push_back({data.timestamp, raw_dbfs, s21_db});
        if (r_buf.size() > (size_t)GAUSS_WIN) r_buf.pop_front();
        if (r_buf.size() < (size_t)GAUSS_WIN) continue;

        double g_val = 0.0;
        for (int i = 0; i < GAUSS_WIN; ++i) g_val += r_buf[i].s21 * GAUSS_COEFFS[i];
        const auto& r_mid = r_buf[GAUSS_WIN / 2];
        g_buf.push_back({r_mid.t, r_mid.dbfs, r_mid.s21, g_val});

        // ── Stage 2: Savitzky-Golay ── (cap at top, no pop at bottom)
        if (g_buf.size() > (size_t)SAVGOL_WIN) g_buf.pop_front();
        if (g_buf.size() < (size_t)SAVGOL_WIN) continue;

        double sg = 0.0;
        for (int i = 0; i < SAVGOL_WIN; ++i) sg += g_buf[i].g_val * SG_COEFFS[i];

        // ── Baseline: rolling median over CLEAN samples only ──
        // Only update when not currently detecting, so mine dips don't pollute baseline
        if (!is_detecting) {
            baseline_buf.push_back(sg);
            if (baseline_buf.size() > (size_t)MEDIAN_WIN) baseline_buf.pop_front();
        }

        if (baseline_buf.size() >= (size_t)BASELINE_MIN) {
            std::vector<double> tmp(baseline_buf.begin(), baseline_buf.end());
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
            baseline = tmp[tmp.size() / 2];
        }

        double diff      = std::abs(sg - baseline);
        bool   triggered = (diff > DETECT_THRESH) && (baseline_buf.size() >= (size_t)BASELINE_MIN);
        const auto& pt   = g_buf[SAVGOL_WIN / 2];

        // ── Detection state transitions ──
        if (triggered && !is_detecting) {
            std::cout << "\n[!!!] TARGET DETECTED at T=" << std::fixed << std::setprecision(3)
                      << pt.t << "s  |  Diff: " << std::setprecision(2) << diff
                      << " dB  |  SG: " << sg << " dB  |  Baseline: " << baseline << " dB" << std::endl;
            is_detecting = true;
        } else if (!triggered && is_detecting) {
            std::cout << "\n[ - ] Signal returned to baseline at T=" << std::fixed
                      << std::setprecision(3) << pt.t << "s" << std::endl;
            is_detecting = false;
        }

        // ── Local Minima Detection (mine center confirmation) ──
        // Uses a 5-point window: checks centre is below ALL four neighbours (robust V-shape)
        sg_history.push_back(sg);
        if (sg_history.size() > (size_t)SG_HIST_WIN) sg_history.pop_front();

        bool is_minima = false;
        if (sg_history.size() == (size_t)SG_HIST_WIN && triggered) {
            double c = sg_history[2]; // centre point
            if (c < sg_history[0] && c < sg_history[1] &&
                c < sg_history[3] && c < sg_history[4]) {
                is_minima = true;
                std::cout << "[>>>] MINIMA — MINE CENTER at T=" << std::fixed
                          << std::setprecision(3) << pt.t << "s  |  Depth: "
                          << std::setprecision(2) << diff << " dB below baseline" << std::endl;
            }
        }

        // ── Write CSV row ──
        csv << pt.t     << ","
            << pt.dbfs  << ","
            << pt.s21   << ","
            << pt.g_val << ","
            << sg        << ","
            << baseline  << ","
            << diff      << ","
            << (triggered ? 1 : 0) << ","
            << (is_minima ? 1 : 0) << ","
            << obj       << ","
            << height    << "\n";

        // ── Console throttle: ~10 updates/sec ──
        static int p_cnt = 0;
        if (++p_cnt % 6 == 0) {
            std::cout << "\r" << (triggered ? "[DETECT] " : "[SCAN]   ")
                      << "T: "       << std::fixed << std::setprecision(1) << pt.t
                      << "s | SG: "  << std::setprecision(2) << sg
                      << " dB | Base: " << baseline
                      << " dB | Diff: " << diff << " dB   " << std::flush;
        }
    }

    // ── Flush DSP pipeline tail ──
    std::cout << "\n[Proc] Flushing DSP pipeline..." << std::endl;
    flush_pipeline(last_dbfs, last_s21, last_t);

    csv.flush();
    csv.close();
    std::cout << "[Proc] Log saved: " << csv_path << std::endl;
}

// ───────────── Main ─────────────
int main(int argc, char** argv) {
    long long f_mhz = (argc > 1) ? std::stoll(argv[1]) : 2000;

    // Precompute local oscillator
    PRECOMPUTED_OSC.resize(NUM_S);
    for (size_t i = 0; i < NUM_S; ++i)
        PRECOMPUTED_OSC[i] = std::polar(1.0f, (float)(-2.0 * M_PI * TONE * (i / FS)));
    std::cout << "[Init] Precomputed " << NUM_S << " oscillator states." << std::endl;

    std::thread acq(acquisition_thread, "192.168.2.1", f_mhz * 1000000LL);
    std::thread proc(processing_thread, f_mhz, "MINE_TEST", 1.25);

    std::cout << "Dual-Threaded VNA Mine Detector Active." << std::endl;
    std::cout << "Press Enter to stop gracefully..." << std::endl;
    std::cin.get();

    keep_running = false;
    queue_cv.notify_all();
    if (acq.joinable())  acq.join();
    if (proc.joinable()) proc.join();

    std::cout << "\nScan complete." << std::endl;
    return 0;
}
