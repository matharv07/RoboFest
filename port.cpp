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
const size_t NUM_S = 262144; // 2^18
const double TONE = 543e3;
const double EPS = 1e-15;
const int SAVGOL_WIN = 21;

// ⚠️ UPDATE THIS WITH YOUR DIRECT CABLE CALIBRATION VALUE
const double CAL_REF_DBFS = 1.62; 

// Savitzky-Golay Coefficients (Window=21, PolyOrder=2)
const double SG_COEFFS[21] = {
    -0.051948, -0.021645, 0.004329, 0.025974, 0.043290, 0.056277, 0.064935, 0.069264, 0.073593, 0.074458, 0.077922, // Center Point (i=0)
     0.074458, 0.073593, 0.069264, 0.064935, 0.056277, 0.043290, 0.025974, 0.004329, -0.021645, -0.051948
};

struct SdrData {
    std::vector<cf32> rx_complex;
    double timestamp;
};

// Threading Globals
std::deque<SdrData> data_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
std::atomic<bool> keep_running{true};

// ───────────── DSP Logic ─────────────

float lockin_omp(const std::vector<cf32>& buf) {
    if (buf.empty()) return 0.0f;
    float real_acc = 0.0f, imag_acc = 0.0f;
    int n = static_cast<int>(buf.size());

    #pragma omp parallel for reduction(+:real_acc, imag_acc)
    for (int i = 0; i < n; ++i) {
        float phase = -2.0f * M_PI * TONE * (i / FS);
        cf32 osc = std::polar(1.0f, phase);
        cf32 mixed = buf[i] * osc;
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

    // 1. Find Devices
    iio_device *phy = iio_context_find_device(ctx, "ad9361-phy");
    iio_device *rx_dev = iio_context_find_device(ctx, "cf-ad9361-lpc");
    iio_device *tx_dev = iio_context_find_device(ctx, "cf-ad9361-dds-core-lpc"); // TX Core
    
    if (!phy || !rx_dev || !tx_dev) {
        std::cerr << "[Fatal] Hardware devices not found." << std::endl;
        keep_running = false;
        return;
    }

    // 2. Setup RX Channels
    iio_channel *rx_i = iio_device_find_channel(rx_dev, "voltage0", false);
    iio_channel *rx_q = iio_device_find_channel(rx_dev, "voltage1", false);
    iio_channel *rx_lo = iio_device_find_channel(phy, "altvoltage0", true); // RX LO

    // 3. Setup TX Channels
    iio_channel *tx_i = iio_device_find_channel(tx_dev, "voltage0", true);
    iio_channel *tx_q = iio_device_find_channel(tx_dev, "voltage1", true);
    iio_channel *tx_lo = iio_device_find_channel(phy, "altvoltage1", true); // TX LO
    iio_channel *tx_phy_i = iio_device_find_channel(phy, "voltage0", true); // TX Attenuation

    // 4. Configure Hardware Frequencies and Gains
    if (rx_lo) iio_channel_attr_write_longlong(rx_lo, "frequency", freq);
    if (tx_lo) iio_channel_attr_write_longlong(tx_lo, "frequency", freq);
    
    // Set TX Attenuation to -10 dB to avoid saturating the RX during a direct cable connection
    if (tx_phy_i) iio_channel_attr_write_double(tx_phy_i, "hardwaregain", -10.0);

    // Enable all channels
    iio_channel_enable(rx_i);
    iio_channel_enable(rx_q);
    iio_channel_enable(tx_i);
    iio_channel_enable(tx_q);

    // 5. Build and Push the TX Cyclic Buffer
    const size_t TX_NUM_S = 8000; // Perfect multiple for 543kHz @ 8MSPS
    iio_buffer *txbuf = iio_device_create_buffer(tx_dev, TX_NUM_S, true); // true = cyclic buffer
    if (!txbuf) {
        std::cerr << "[Fatal] TX Buffer creation failed." << std::endl;
        keep_running = false;
        return;
    }

    char *p_dat, *p_end = (char *)iio_buffer_end(txbuf);
    ptrdiff_t p_inc = iio_buffer_step(txbuf);
    char *p_start = (char *)iio_buffer_first(txbuf, tx_i);

    // Generate complex sine wave with 0.7 amplitude (1432 / 2048) to prevent DAC clipping
    int idx = 0;
    for (p_dat = p_start; p_dat < p_end; p_dat += p_inc, ++idx) {
        float t = (float)idx / FS;
        int16_t i_val = (int16_t)(1432.0f * std::cos(2.0f * M_PI * TONE * t));
        int16_t q_val = (int16_t)(1432.0f * std::sin(2.0f * M_PI * TONE * t));
        ((int16_t*)p_dat)[0] = i_val;
        ((int16_t*)p_dat)[1] = q_val;
    }
    iio_buffer_push(txbuf);
    std::cout << "[Acq] TX Cyclic Buffer Pushed (" << TONE/1000 << " kHz tone, -10 dB Attenuation)." << std::endl;

    // 6. Start RX Acquisition
    iio_buffer *rxbuf = iio_device_create_buffer(rx_dev, NUM_S, false);
    if (!rxbuf) {
        std::cerr << "[Fatal] RX Buffer creation failed." << std::endl;
        keep_running = false;
        return;
    }

    auto start_time = std::chrono::steady_clock::now();
    std::cout << "[Acq] Hardware Ready. Starting capture." << std::endl;

    while (keep_running) {
        if (iio_buffer_refill(rxbuf) < 0) break;

        SdrData data;
        data.timestamp = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
        data.rx_complex.reserve(NUM_S);

        char *r_dat, *r_end = (char *)iio_buffer_end(rxbuf);
        ptrdiff_t r_inc = iio_buffer_step(rxbuf);
        char *b0 = (char *)iio_buffer_first(rxbuf, rx_i);

        if (b0) {
            for (r_dat = b0; r_dat < r_end; r_dat += r_inc) {
                float i_val = ((int16_t *)r_dat)[0] / 2048.0f;
                float q_val = ((int16_t *)r_dat)[1] / 2048.0f;
                data.rx_complex.push_back(cf32(i_val, q_val));
            }

            std::lock_guard<std::mutex> lock(queue_mutex);
            if (data_queue.size() > 10) data_queue.pop_front();
            data_queue.push_back(std::move(data));
            queue_cv.notify_one();
        }
    }
    
    // Clean up
    iio_buffer_destroy(rxbuf);
    iio_buffer_destroy(txbuf);
    iio_context_destroy(ctx);
}

// ----------------------------------------------------------------------------
// Dynamically generates a normalized Gaussian kernel
// ----------------------------------------------------------------------------
std::vector<double> generate_gaussian_kernel(int window_size, double sigma) {
    std::vector<double> coeffs(window_size);
    double sum = 0.0;
    int half_window = window_size / 2;

    for (int i = 0; i < window_size; ++i) {
        int x = i - half_window;
        coeffs[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        sum += coeffs[i];
    }

    // Normalize so the sum is exactly 1.0 to prevent artificial dB shifts
    for (int i = 0; i < window_size; ++i) {
        coeffs[i] /= sum;
    }
    return coeffs;
}

// ----------------------------------------------------------------------------
// DSP Processing Thread: Raw -> Gaussian -> Savitzky-Golay -> CSV
// ----------------------------------------------------------------------------
void processing_thread(long long freq_mhz, std::string obj, double height) {
    std::string folder = "VCA/" + std::to_string(freq_mhz) + "MHz";
    fs::create_directories(folder);
    
    // Setup CSV
    std::time_t rawtime;
    std::time(&rawtime);
    std::tm* timeinfo = std::localtime(&rawtime);
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%d-%m-%H%M%S", timeinfo);
    std::string csv_path = folder + "/scan_cpp-" + std::string(buffer) + ".csv";
    std::ofstream csv(csv_path);
    
    csv << "Time,Raw_dBFS,S21_dB,S21_Gaussian,S21_Gauss_SavGol,OBJ,Height\n";

    // --- Filter Configuration ---
    const int GAUSS_WIN = 15;
    const double GAUSS_SIGMA = 2.0;
    std::vector<double> GAUSS_COEFFS = generate_gaussian_kernel(GAUSS_WIN, GAUSS_SIGMA);
    
    const int SAVGOL_WIN = 21;
    // Correctly Normalized SavGol (Window 21, Poly 2) - Sums exactly to 1.0
    const double SG_COEFFS[21] = {
        -0.0559006, -0.0248447,  0.0029421,  0.0274600,  0.0487087, 
         0.0666885,  0.0813991,  0.0928408,  0.1010134,  0.1059170, 
         0.1075515,  0.1059170,  0.1010134,  0.0928408,  0.0813991,  
         0.0666885,  0.0487087,  0.0274600,  0.0029421, -0.0248447, -0.0559006
    };

    // --- Data Structures ---
    struct RawPoint { double timestamp; double raw_dbfs; double s21_db; };
    struct GaussPoint { double timestamp; double raw_dbfs; double s21_db; double gauss_val; };

    std::deque<RawPoint> raw_buffer;     // Stage 1: Feeds the Gaussian Filter
    std::deque<GaussPoint> gauss_buffer; // Stage 2: Feeds the SavGol Filter

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

        // 1. Base Calculations
        double raw_dbfs = 20.0 * std::log10(std::max((double)lockin_omp(data.rx_complex), EPS));
        double s21_db = raw_dbfs - CAL_REF_DBFS;

        // 2. Stage One: Gaussian Filter
        raw_buffer.push_back({data.timestamp, raw_dbfs, s21_db});
        if (raw_buffer.size() > GAUSS_WIN) raw_buffer.pop_front();

        if (raw_buffer.size() == GAUSS_WIN) {
            double current_gauss = 0.0;
            for (int i = 0; i < GAUSS_WIN; ++i) {
                current_gauss += raw_buffer[i].s21_db * GAUSS_COEFFS[i];
            }

            // Get the center point of the Gaussian window
            const auto& center_raw = raw_buffer[GAUSS_WIN / 2];

            // 3. Stage Two: Savitzky-Golay Filter
            gauss_buffer.push_back({center_raw.timestamp, center_raw.raw_dbfs, center_raw.s21_db, current_gauss});
            if (gauss_buffer.size() > SAVGOL_WIN) gauss_buffer.pop_front();

            if (gauss_buffer.size() == SAVGOL_WIN) {
                double final_savgol = 0.0;
                for (int i = 0; i < SAVGOL_WIN; ++i) {
                    final_savgol += gauss_buffer[i].gauss_val * SG_COEFFS[i];
                }

                // Get the center point of the SavGol window
                const auto& final_pt = gauss_buffer[SAVGOL_WIN / 2];

                // 4. Log everything in perfect time-alignment
                csv << final_pt.timestamp << "," << final_pt.raw_dbfs << "," << final_pt.s21_db << "," 
                    << final_pt.gauss_val << "," << final_savgol << "," << obj << "," << height << "\n";
                
                // Force flush during the live loop to prevent empty files if force-quit
                csv.flush(); 

                std::cout << "\r[OMP] T: " << std::fixed << std::setprecision(1) << final_pt.timestamp 
                          << "s | Raw: " << std::setprecision(2) << final_pt.s21_db 
                          << " dB | Final (G+SG): " << std::setprecision(2) << final_savgol << " dB   " << std::flush;
            }
        }
    }

    // ==========================================
    // --- CLEANUP / FLUSH PIPELINE ---
    // ==========================================
    
    // Only flush if we actually collected enough data to start the pipeline
    if (!raw_buffer.empty() && gauss_buffer.size() > 0) {
        std::cout << "\n[Proc] SDR Stopped. Flushing remaining samples from DSP pipeline..." << std::endl;
        
        // Calculate total delay: (15 / 2) + (21 / 2) = 7 + 10 = 17 samples
        int flush_count = (GAUSS_WIN / 2) + (SAVGOL_WIN / 2); 
        
        // Grab the last valid data to pad the filter and prevent edge artifacts
        double pad_raw = raw_buffer.back().raw_dbfs;
        double pad_s21 = raw_buffer.back().s21_db;
        double last_time = raw_buffer.back().timestamp;
        
        // Approximate the time step to keep the CSV timeline moving forward evenly (~32ms per buffer)
        double time_step = 0.032; 

        for (int flush = 0; flush < flush_count; ++flush) {
            last_time += time_step;
            
            // 1. Push the padded data into the Gaussian window
            raw_buffer.push_back({last_time, pad_raw, pad_s21});
            raw_buffer.pop_front(); 

            double current_gauss = 0.0;
            for (int i = 0; i < GAUSS_WIN; ++i) {
                current_gauss += raw_buffer[i].s21_db * GAUSS_COEFFS[i];
            }

            const auto& center_raw = raw_buffer[GAUSS_WIN / 2];

            // 2. Push the Gaussian output into the SavGol window
            gauss_buffer.push_back({center_raw.timestamp, center_raw.raw_dbfs, center_raw.s21_db, current_gauss});
            if (gauss_buffer.size() > SAVGOL_WIN) gauss_buffer.pop_front();

            if (gauss_buffer.size() == SAVGOL_WIN) {
                double final_savgol = 0.0;
                for (int i = 0; i < SAVGOL_WIN; ++i) {
                    final_savgol += gauss_buffer[i].gauss_val * SG_COEFFS[i];
                }

                const auto& final_pt = gauss_buffer[SAVGOL_WIN / 2];

                // 3. Log the flushed data
                csv << final_pt.timestamp << "," << final_pt.raw_dbfs << "," << final_pt.s21_db << "," 
                    << final_pt.gauss_val << "," << final_savgol << "," << obj << "," << height << "\n";
            }
        }
    }
    
    // Final force save to disk
    csv.flush(); 
    csv.close();
    std::cout << "[Proc] Log file saved and closed safely." << std::endl;
}

int main(int argc, char** argv) {
    long long f_mhz = (argc > 1) ? std::stoll(argv[1]) : 2000;
    
    std::thread acq(acquisition_thread, "192.168.2.1", f_mhz * 1000000);
    std::thread proc(processing_thread, f_mhz, "NO", 1.25);

    std::cout << "Dual-Threaded VNA Active. OpenMP Cores: " << omp_get_max_threads() << std::endl;
    std::cout << "Press Enter to gracefully stop..." << std::endl;
    std::cin.get();
    
    keep_running = false;
    queue_cv.notify_all(); // Wake up processor thread if it's waiting
    
    if (acq.joinable()) acq.join();
    if (proc.joinable()) proc.join();

    std::cout << "\nScan Complete." << std::endl;
    return 0;
}
