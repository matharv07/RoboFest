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

namespace fs = std::filesystem;
typedef std::complex<float> cf32;

// ───────────── Configuration ─────────────
const double FS = 8e6;
const size_t NUM_S = 131072; // ~40-61 Hz Refresh Rate (Max USB 2.0 Throughput)
const double TONE = 543e3;
const double EPS = 1e-15;

// ⚠️ UPDATE THIS WITH YOUR DIRECT CABLE CALIBRATION VALUE
const double CAL_REF_DBFS = -14.4; 

struct SdrData {
    std::vector<cf32> rx_complex;
    double timestamp;
};

// Threading & Optimization Globals
std::deque<SdrData> data_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
std::atomic<bool> keep_running{true};

// Precomputed Local Oscillator for lightning-fast lock-in math
std::vector<cf32> PRECOMPUTED_OSC;

// ───────────── Mathematical Helpers ─────────────
std::vector<double> generate_gaussian_kernel(int window_size, double sigma) {
    std::vector<double> coeffs(window_size);
    double sum = 0.0;
    int half_window = window_size / 2;

    for (int i = 0; i < window_size; ++i) {
        int x = i - half_window;
        coeffs[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += coeffs[i];
    }

    for (int i = 0; i < window_size; ++i) {
        coeffs[i] /= sum;
    }
    return coeffs;
}

// ───────────── Optimized DSP Logic ─────────────
float lockin_omp(const std::vector<cf32>& buf) {
    if (buf.empty()) return 0.0f;
    float real_acc = 0.0f, imag_acc = 0.0f;
    int n = static_cast<int>(buf.size());

    // Single-core array multiplication (Faster than OpenMP overhead at this size)
    for (int i = 0; i < n; ++i) {
        cf32 mixed = buf[i] * PRECOMPUTED_OSC[i]; 
        real_acc += mixed.real();
        imag_acc += mixed.imag();
    }
    return std::sqrt(real_acc * real_acc + imag_acc * imag_acc) / n;
}

// ───────────── Threads ─────────────
void acquisition_thread(const std::string& ip, long long freq) {
    std::cout << "[Acq] Connecting to Pluto at " << ip << "..." << std::endl;
    iio_context *ctx = iio_create_network_context(ip.c_str());
    if (!ctx) {
        std::cerr << "[Fatal] Could not connect to SDR." << std::endl;
        keep_running = false;
        return;
    }

    iio_device *phy = iio_context_find_device(ctx, "ad9361-phy");
    iio_device *rx_dev = iio_context_find_device(ctx, "cf-ad9361-lpc");
    iio_device *tx_dev = iio_context_find_device(ctx, "cf-ad9361-dds-core-lpc"); 
    
    if (!phy || !rx_dev || !tx_dev) {
        std::cerr << "[Fatal] Hardware devices not found." << std::endl;
        keep_running = false;
        return;
    }

    iio_channel *rx_i = iio_device_find_channel(rx_dev, "voltage0", false);
    iio_channel *rx_q = iio_device_find_channel(rx_dev, "voltage1", false);
    iio_channel *rx_lo = iio_device_find_channel(phy, "altvoltage0", true); 

    iio_channel *tx_i = iio_device_find_channel(tx_dev, "voltage0", true);
    iio_channel *tx_q = iio_device_find_channel(tx_dev, "voltage1", true);
    iio_channel *tx_lo = iio_device_find_channel(phy, "altvoltage1", true); 
    iio_channel *tx_phy_i = iio_device_find_channel(phy, "voltage0", true); 
    
    // NEW: Grab the physical RX channel to control the amplifier
    iio_channel *rx_phy_i = iio_device_find_channel(phy, "voltage0", false); 

    if (rx_lo) iio_channel_attr_write_longlong(rx_lo, "frequency", freq);
    if (tx_lo) iio_channel_attr_write_longlong(tx_lo, "frequency", freq);
    if (tx_phy_i) iio_channel_attr_write_double(tx_phy_i, "hardwaregain", -10.0);

    // NEW: Disable AGC and lock RX gain to a manual value
    if (rx_phy_i) {
        iio_channel_attr_write(rx_phy_i, "gain_control_mode", "manual");
        // ⚠️ SET THIS: 5.0 for loopback cable, 30.0+ for antennas
        iio_channel_attr_write_double(rx_phy_i, "hardwaregain", 30.0); 
    }

    iio_channel_enable(rx_i); iio_channel_enable(rx_q);
    iio_channel_enable(tx_i); iio_channel_enable(tx_q);

    const size_t TX_NUM_S = 8000; 
    iio_buffer *txbuf = iio_device_create_buffer(tx_dev, TX_NUM_S, true); 
    if (!txbuf) {
        std::cerr << "[Fatal] TX Buffer creation failed." << std::endl;
        keep_running = false; return;
    }

    char *p_dat, *p_end = (char *)iio_buffer_end(txbuf);
    ptrdiff_t p_inc = iio_buffer_step(txbuf);
    char *p_start = (char *)iio_buffer_first(txbuf, tx_i);

    int idx = 0;
    for (p_dat = p_start; p_dat < p_end; p_dat += p_inc, ++idx) {
        float t = (float)idx / FS;
        ((int16_t*)p_dat)[0] = (int16_t)(1432.0f * std::cos(2.0f * M_PI * TONE * t));
        ((int16_t*)p_dat)[1] = (int16_t)(1432.0f * std::sin(2.0f * M_PI * TONE * t));
    }
    iio_buffer_push(txbuf);
    std::cout << "[Acq] TX Cyclic Buffer Pushed (" << TONE/1000 << " kHz tone, -10 dB Attenuation)." << std::endl;

    iio_buffer *rxbuf = iio_device_create_buffer(rx_dev, NUM_S, false);
    if (!rxbuf) {
        std::cerr << "[Fatal] RX Buffer creation failed." << std::endl;
        keep_running = false; return;
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "[Acq] Hardware Ready. Starting capture." << std::endl;

    while (keep_running) {
        if (iio_buffer_refill(rxbuf) < 0) break;

        SdrData data;
        data.timestamp = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        
        char *r_dat, *r_end = (char *)iio_buffer_end(rxbuf);
        ptrdiff_t r_inc = iio_buffer_step(rxbuf);
        char *b0 = (char *)iio_buffer_first(rxbuf, rx_i);

        if (b0) {
            data.rx_complex.resize(NUM_S); 
            int fill_idx = 0;
            
            for (r_dat = b0; r_dat < r_end; r_dat += r_inc) {
                float i_val = ((int16_t *)r_dat)[0] / 2048.0f;
                float q_val = ((int16_t *)r_dat)[1] / 2048.0f;
                data.rx_complex[fill_idx++] = cf32(i_val, q_val); 
            }

            std::lock_guard<std::mutex> lock(queue_mutex);
            if (data_queue.size() > 10) data_queue.pop_front();
            data_queue.push_back(std::move(data));
            queue_cv.notify_one();
        }
    }
    
    iio_buffer_destroy(rxbuf); iio_buffer_destroy(txbuf); iio_context_destroy(ctx);
}

void processing_thread(long long freq_mhz, std::string obj, double height) {
    std::string folder = "VCA/" + std::to_string(freq_mhz) + "MHz";
    fs::create_directories(folder);
    
    std::time_t rawtime; std::time(&rawtime);
    std::tm* timeinfo = std::localtime(&rawtime);
    char buffer[20]; std::strftime(buffer, sizeof(buffer), "%d-%m-%H%M%S", timeinfo);
    std::string csv_path = folder + "/scan_cpp-" + std::string(buffer) + ".csv";
    std::ofstream csv(csv_path);
    
    csv << "Time,Raw_dBFS,S21_dB,S21_Gaussian,S21_Gauss_SavGol,OBJ,Height\n";

    const int GAUSS_WIN = 15;
    const double GAUSS_SIGMA = 2.0;
    std::vector<double> GAUSS_COEFFS = generate_gaussian_kernel(GAUSS_WIN, GAUSS_SIGMA);
    
    const int SAVGOL_WIN = 21;
    const double SG_COEFFS[21] = {
        -0.0559006, -0.0248447,  0.0029421,  0.0274600,  0.0487087, 
         0.0666885,  0.0813991,  0.0928408,  0.1010134,  0.1059170, 
         0.1075515,  0.1059170,  0.1010134,  0.0928408,  0.0813991,  
         0.0666885,  0.0487087,  0.0274600,  0.0029421, -0.0248447, -0.0559006
    };

    struct RawPoint { double timestamp; double raw_dbfs; double s21_db; };
    struct GaussPoint { double timestamp; double raw_dbfs; double s21_db; double gauss_val; };

    std::deque<RawPoint> raw_buffer;     
    std::deque<GaussPoint> gauss_buffer; 

    std::cout << "[Proc] Logging to " << csv_path << std::endl;

    while (keep_running || !data_queue.empty()) {
        SdrData data;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait_for(lock, std::chrono::milliseconds(100), []{ return !data_queue.empty(); });
            if (data_queue.empty()) continue;
            data = std::move(data_queue.front());
            data_queue.pop_front();
        }

        double raw_dbfs = 20.0 * std::log10(std::max((double)lockin_omp(data.rx_complex), EPS));
        double s21_db = raw_dbfs - CAL_REF_DBFS;

        raw_buffer.push_back({data.timestamp, raw_dbfs, s21_db});
        if (raw_buffer.size() > GAUSS_WIN) raw_buffer.pop_front();

        if (raw_buffer.size() == GAUSS_WIN) {
            double current_gauss = 0.0;
            for (int i = 0; i < GAUSS_WIN; ++i) {
                current_gauss += raw_buffer[i].s21_db * GAUSS_COEFFS[i];
            }

            const auto& center_raw = raw_buffer[GAUSS_WIN / 2];

            gauss_buffer.push_back({center_raw.timestamp, center_raw.raw_dbfs, center_raw.s21_db, current_gauss});
            if (gauss_buffer.size() > SAVGOL_WIN) gauss_buffer.pop_front();

            if (gauss_buffer.size() == SAVGOL_WIN) {
                double final_savgol = 0.0;
                for (int i = 0; i < SAVGOL_WIN; ++i) {
                    final_savgol += gauss_buffer[i].gauss_val * SG_COEFFS[i];
                }

                const auto& final_pt = gauss_buffer[SAVGOL_WIN / 2];

                // Write to RAM buffer without forcing immediate disk write
                csv << final_pt.timestamp << "," << final_pt.raw_dbfs << "," << final_pt.s21_db << "," 
                    << final_pt.gauss_val << "," << final_savgol << "," << obj << "," << height << "\n";

                // Print Throttle: Only update the screen every 6 frames (~10 updates/sec)
                static int print_counter = 0;
                if (++print_counter % 6 == 0) {
                    std::cout << "\r[DSP] T: " << std::fixed << std::setprecision(1) << final_pt.timestamp 
                              << "s | Raw: " << std::setprecision(2) << final_pt.s21_db 
                              << " dB | Final (G+SG): " << std::setprecision(2) << final_savgol << " dB   " << std::flush;
                }
            }
        }
    }
    
    if (!raw_buffer.empty() && gauss_buffer.size() > 0) {
        std::cout << "\n[Proc] SDR Stopped. Flushing remaining samples from DSP pipeline..." << std::endl;
        
        int flush_count = (GAUSS_WIN / 2) + (SAVGOL_WIN / 2); 
        double pad_raw = raw_buffer.back().raw_dbfs;
        double pad_s21 = raw_buffer.back().s21_db;
        double last_time = raw_buffer.back().timestamp;
        
        // Dynamic time step automatically handles the 131072 buffer math
        double time_step = static_cast<double>(NUM_S) / FS; 

        for (int flush = 0; flush < flush_count; ++flush) {
            last_time += time_step;
            raw_buffer.push_back({last_time, pad_raw, pad_s21});
            raw_buffer.pop_front(); 

            double current_gauss = 0.0;
            for (int i = 0; i < GAUSS_WIN; ++i) {
                current_gauss += raw_buffer[i].s21_db * GAUSS_COEFFS[i];
            }

            const auto& center_raw = raw_buffer[GAUSS_WIN / 2];
            gauss_buffer.push_back({center_raw.timestamp, center_raw.raw_dbfs, center_raw.s21_db, current_gauss});
            if (gauss_buffer.size() > SAVGOL_WIN) gauss_buffer.pop_front();

            if (gauss_buffer.size() == SAVGOL_WIN) {
                double final_savgol = 0.0;
                for (int i = 0; i < SAVGOL_WIN; ++i) {
                    final_savgol += gauss_buffer[i].gauss_val * SG_COEFFS[i];
                }
                const auto& final_pt = gauss_buffer[SAVGOL_WIN / 2];

                csv << final_pt.timestamp << "," << final_pt.raw_dbfs << "," << final_pt.s21_db << "," 
                    << final_pt.gauss_val << "," << final_savgol << "," << obj << "," << height << "\n";
            }
        }
    }
    
    // Final save
    csv.flush(); csv.close();
    std::cout << "[Proc] Log file saved and closed safely." << std::endl;
}

int main(int argc, char** argv) {
    long long f_mhz = (argc > 1) ? std::stoll(argv[1]) : 2000;
    
    // --- PRECOMPUTE DSP ARRAYS ---
    PRECOMPUTED_OSC.resize(NUM_S);
    for (size_t i = 0; i < NUM_S; ++i) {
        float phase = -2.0f * M_PI * TONE * (i / FS);
        PRECOMPUTED_OSC[i] = std::polar(1.0f, phase);
    }
    std::cout << "[Init] Precomputed " << NUM_S << " oscillator states." << std::endl;
    
    std::thread acq(acquisition_thread, "192.168.2.1", f_mhz * 1000000);
    std::thread proc(processing_thread, f_mhz, "NO", 1.25);

    std::cout << "Dual-Threaded VNA Active." << std::endl;
    std::cout << "Press Enter to gracefully stop..." << std::endl;
    std::cin.get();
    
    keep_running = false;
    queue_cv.notify_all(); 
    
    if (acq.joinable()) acq.join();
    if (proc.joinable()) proc.join();

    std::cout << "\nScan Complete." << std::endl;
    return 0;
}
