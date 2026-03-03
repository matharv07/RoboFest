#!/usr/bin/env python3

import sys, os, time, traceback
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT, FigureCanvasQTAgg as FigureCanvas)
import adi
from scipy.signal import kaiserord, firwin, lfilter, fftconvolve, oaconvolve
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QFrame, QMessageBox, QCheckBox, QComboBox)
from PyQt6.QtCore import QThread, pyqtSignal

FILTER_MODE, FFT_METHOD = 'fft', 'fftconvolve'
FILT_RIPPLE_DB, FILT_CUTOFF_HZ, FILT_TRANS_WIDTH_HZ = 70, 500, 100
SMOOTH_WIN21, SMOOTH_WIN11 = 7, 5
SDR_URI = "ip:192.168.2.1"
FS, NUM_S, TONE = 8e6, 2**18, 543e3
RF_F = 4e6
DWELL, CLR_READS, EPS = 0.1, 1, 1e-15
MIN_FREQ, MAX_FREQ = 0.3e9, 3e9
DEFAULT_CENTER_FREQ = 1e9

# ───────────── hardware initialisation ─────────────
sdr = adi.ad9361(uri=SDR_URI)
sdr.sample_rate             = int(FS)
sdr.tx_enabled_channels     = [0]
sdr.rx_enabled_channels     = [0, 1]
sdr.rx_rf_bandwidth         = int(RF_F)
sdr.tx_rf_bandwidth         = int(RF_F)
sdr.rx_buffer_size          = NUM_S
sdr.tx_buffer_size          = NUM_S
sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.tx_cyclic_buffer        = True
sdr.tx_hardwaregain_chan0   = -10
sdr.rx_hardwaregain_chan0   = 50
sdr.rx_hardwaregain_chan1   = 50

_t = np.arange(NUM_S) / FS
sdr.tx((np.exp(2j * np.pi * TONE * _t) * (2**14)).astype(np.complex64))

# ───────────── FIR filter for lock-in ─────────────
nyq     = FS / 2
N, beta = kaiserord(FILT_RIPPLE_DB, FILT_TRANS_WIDTH_HZ/nyq)
b_fir   = firwin(N, FILT_CUTOFF_HZ/nyq, window=('kaiser', beta))

def apply_filter(x):
    if FILTER_MODE == 'direct':
        return lfilter(b_fir, 1.0, x)
    func = fftconvolve if FFT_METHOD == 'fftconvolve' else oaconvolve
    return func(x, b_fir, mode='same')

def lockin(buf: np.ndarray) -> float:
    if np.allclose(buf, 0):
        return 0.0
    t = np.arange(len(buf)) / FS
    y = apply_filter(buf * np.exp(-2j * np.pi * TONE * t))
    y = y[N//2:]  # discard FIR transient
    return np.abs(y).mean()

def to_dB(x):
    return 20 * np.log10(np.maximum(x, EPS))

def smooth_trace(y, k):
    if k <= 1 or y.size < k:
        return y
    mask = np.isfinite(y).astype(float)
    win = np.ones(k)
    num = np.convolve(np.nan_to_num(y), win, 'same')
    den = np.convolve(mask, win, 'same')
    out = np.full_like(y, np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out

# ───────────── worker thread ─────────────
class TimeSweepThread(QThread):
    update  = pyqtSignal(float, float, float) # (time, s21, s11)
    started = pyqtSignal()
    error   = pyqtSignal(str)

    def __init__(self, dev):
        super().__init__()
        self.dev       = dev
        self.center_f  = DEFAULT_CENTER_FREQ
        self.stop_flag = True
        self.pause_flag = False

    def set_freq(self, f):
        self.center_f = f

    def stop(self):
        self.stop_flag = True
        self.pause_flag = False

    def _safe_rx(self):
        try:
            return self.dev.rx()
        except Exception as e:
            self.error.emit(f"SDR RX error: {e}")
            self.stop_flag = True
            raise

    def run(self):
        self.started.emit()
        f = self.center_f
        NUM_R = 4 if f < 1e9 else 1

        try:
            self.dev.tx_lo = self.dev.rx_lo = int(f)
        except Exception as e:
            self.error.emit(f"SDR tune error: {e}")
            self.stop_flag = True
            return

        time.sleep(0.5) # Let PLL settle
        
        start_time = time.time()
        total_pause_time = 0.0
        is_paused_internal = False
        pause_start = 0.0

        while not self.stop_flag:
            if self.pause_flag:
                if not is_paused_internal:
                    pause_start = time.time()
                    is_paused_internal = True
                time.sleep(0.1)
                continue
            else:
                if is_paused_internal:
                    total_pause_time += (time.time() - pause_start)
                    is_paused_internal = False

            time.sleep(DWELL)
            for _ in range(CLR_READS):
                self._safe_rx()

            acc0 = np.zeros(NUM_S * NUM_R, np.complex64)
            acc1 = np.zeros_like(acc0)
            
            for j in range(NUM_R):
                r = self._safe_rx()
                acc0[j*NUM_S:(j+1)*NUM_S] = (r[0]/2**12) * 7
                acc1[j*NUM_S:(j+1)*NUM_S] = (r[1]/2**12) * 7

            s21 = to_dB(lockin(acc0))
            s11 = to_dB(lockin(acc1))

            if not np.isnan(s11) and s11 > 0:
                s11 = 0.0

            t_elapsed = time.time() - start_time - total_pause_time
            self.update.emit(t_elapsed, s21, s11)

# ───────────── GUI ─────────────
class VNA(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlutoSDR VNA Pro – Time Domain")
        
        self.meta_obj = "NO"
        self.meta_height = 0.0
        self.last_t = 0.0 # Track the current time for the marker
        self.meta_lines = []

        self._build_ui()
        self._init_plot()
        
        self.wk = TimeSweepThread(sdr)
        self.wk.update.connect(self._update)
        self.wk.started.connect(self._reset)
        self.wk.error.connect(lambda m: QMessageBox.critical(self, "Worker error", m))

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        v = QVBoxLayout(cw)

        top = QFrame()
        top.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        h = QHBoxLayout(top)

        h.addWidget(QLabel("Freq (MHz):"))
        self.le_cf = QLineEdit(str(int(DEFAULT_CENTER_FREQ/1e6)))
        self.le_cf.setFixedWidth(60)
        h.addWidget(self.le_cf)

        self.btn_toggle_scan = QPushButton("Start Scan")
        self.btn_toggle_scan.clicked.connect(self.toggle_scan)
        h.addWidget(self.btn_toggle_scan)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_pause.setEnabled(False)
        h.addWidget(self.btn_pause)

        h.addSpacing(20)

        h.addWidget(QLabel("OBJ:"))
        self.combo_obj = QComboBox()
        self.combo_obj.addItems(["NO", "YES"])
        h.addWidget(self.combo_obj)

        h.addWidget(QLabel("Height (m):"))
        self.le_height = QLineEdit("0.0")
        self.le_height.setFixedWidth(50)
        h.addWidget(self.le_height)

        self.btn_save_meta = QPushButton("Save Metadata")
        self.btn_save_meta.clicked.connect(lambda: self.save_metadata(quiet=False))
        h.addWidget(self.btn_save_meta)

        h.addSpacing(20)

        self.btn_export = QPushButton("Save Data & Plot")
        self.btn_export.clicked.connect(lambda: self.export_data(auto=False))
        h.addWidget(self.btn_export)

        h.addStretch()

        h.addWidget(QLabel("Show:"))
        self.cbSmooth = QCheckBox("Smoothed")
        self.cbSmooth.setChecked(True)
        h.addWidget(self.cbSmooth)
        
        self.cbRaw = QCheckBox("Raw")
        self.cbRaw.setChecked(True)
        h.addWidget(self.cbRaw)
        
        self.cbSmooth.stateChanged.connect(self._vis_toggle)
        self.cbRaw.stateChanged.connect(self._vis_toggle)

        v.addWidget(top)

        self.fig, (self.ax21, self.ax11) = plt.subplots(2, 1, figsize=(12, 9))
        self.canvas = FigureCanvas(self.fig)
        v.addWidget(self.canvas)
        v.addWidget(NavigationToolbar2QT(self.canvas, self))

    def _init_plot(self):
        self.x_time, self.y21r, self.y11r = [], [], []
        self.obj_history, self.height_history = [], []

        self.l21, = self.ax21.plot([], [], 'b-', lw=1.2, label='S21 smoothed')
        self.d21, = self.ax21.plot([], [], 'ro', ms=4,   label='S21 raw')
        self.l11, = self.ax11.plot([], [], 'g-', lw=1.2, label='S11 smoothed')
        self.d11, = self.ax11.plot([], [], 'mo', ms=4,   label='S11 raw')

        self.ax21.set_title("S21 vs Time")
        self.ax11.set_title("S11 vs Time")

        self.ax21.set_xlim(0, 10)
        self.ax11.set_xlim(0, 10)
        self.ax21.set_ylabel("S21 (dB)")
        self.ax11.set_ylabel("S11 (dB)")
        self.ax11.set_xlabel("Elapsed Time (s)")

        for ax in (self.ax21, self.ax11):
            ax.grid(True)

        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.markers = []
        self._vis_toggle()

    def save_metadata(self, quiet=False):
        try:
            new_obj = self.combo_obj.currentText()
            new_height = float(self.le_height.text())
            
            # Check if values actually changed
            is_changed = (new_obj != self.meta_obj) or (new_height != self.meta_height)
            
            self.meta_obj = new_obj
            self.meta_height = new_height
            
            # Draw a visual marker if a change occurred during an active scan
            if is_changed and not self.wk.stop_flag and self.last_t > 0:
                l1 = self.ax21.axvline(x=self.last_t, color='orange', linestyle='--', alpha=0.8)
                l2 = self.ax11.axvline(x=self.last_t, color='orange', linestyle='--', alpha=0.8)
                
                # Transform to keep text at the top of the axes regardless of Y-scale
                t1 = self.ax21.text(self.last_t, 0.95, ' Meta', color='orange', rotation=90, 
                                    transform=self.ax21.get_xaxis_transform(), va='top', ha='right', fontsize=8)
                t2 = self.ax11.text(self.last_t, 0.95, ' Meta', color='orange', rotation=90, 
                                    transform=self.ax11.get_xaxis_transform(), va='top', ha='right', fontsize=8)
                
                self.meta_lines.extend([l1, l2, t1, t2])
                self.canvas.draw_idle()

            if not quiet:
                self.btn_save_meta.setText("Saved!")
                QApplication.processEvents()
                time.sleep(0.5)
                self.btn_save_meta.setText("Save Metadata")
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Height must be a valid number.")

    def export_data(self, auto=False):
        if not self.x_time:
            if not auto:
                QMessageBox.information(self, "Export", "No data to export. Run a scan first.")
            return

        dt_str = datetime.now().strftime("%d-%H%M%S")
        cf_hz = self.wk.center_f
        cf_mhz = cf_hz / 1e6
        cf_ghz = cf_hz / 1e9

        base_dir = "VNA"
        sub_dir = os.path.join(base_dir, f"{int(cf_mhz)}MHz")
        os.makedirs(sub_dir, exist_ok=True)

        base_filename = f"scan_{cf_ghz}GHz_{dt_str}"
        csv_path = os.path.join(sub_dir, base_filename + ".csv")
        img_path = os.path.join(sub_dir, base_filename + ".png")

        df = pd.DataFrame({
            'Elapsed Time (s)': self.x_time,
            'Frequency (Hz)': [cf_hz] * len(self.x_time),
            'S11 (dB)': self.y11r,
            'S21 (dB)': self.y21r,
            'OBJ': self.obj_history,
            'Height (m)': self.height_history
        })

        try:
            df.to_csv(csv_path, index=False)
            self.fig.savefig(img_path, dpi=300, bbox_inches='tight')
            if not auto:
                QMessageBox.information(self, "Export Successful", f"Data and plots saved to:\n{sub_dir}")
            else:
                print(f"✔️ Auto-saved data and plots to {sub_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save files:\n{e}")

    def toggle_scan(self):
        if self.wk.stop_flag:
            self.start_scan()
        else:
            self.stop_scan()

    def start_scan(self):
        try:
            cf = float(self.le_cf.text()) * 1e6
            if not (MIN_FREQ <= cf <= MAX_FREQ):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Input Error", f"Center frequency must be between {MIN_FREQ/1e6} and {MAX_FREQ/1e6} MHz.")
            return

        self.save_metadata(quiet=True)

        self.btn_toggle_scan.setText("Stop Scan")
        self.btn_pause.setEnabled(True)
        self.le_cf.setEnabled(False) 

        self.ax21.set_title(f"S21 vs Time ({cf/1e6} MHz)")
        self.ax11.set_title(f"S11 vs Time ({cf/1e6} MHz)")
        self.canvas.draw()

        self.wk.set_freq(cf)
        self.wk.stop_flag = False
        self.wk.pause_flag = False
        self.btn_pause.setText("Pause")
        self.wk.start()
        
    def stop_scan(self):
        was_running = not self.wk.stop_flag
        
        self.wk.stop()
        self.wk.wait()
        
        self.btn_toggle_scan.setText("Start Scan")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.le_cf.setEnabled(True)

        if was_running:
            self.export_data(auto=True)

    def toggle_pause(self):
        if self.wk.stop_flag:
            return
            
        self.save_metadata(quiet=True)
            
        self.wk.pause_flag = not self.wk.pause_flag
        if self.wk.pause_flag:
            self.btn_pause.setText("Resume")
        else:
            self.btn_pause.setText("Pause")

    def _reset(self):
        self.last_t = 0.0
        self.x_time.clear(); self.y21r.clear(); self.y11r.clear()
        self.obj_history.clear(); self.height_history.clear()
        
        for ln in (self.l21, self.d21, self.l11, self.d11):
            ln.set_data([], [])
            
        for line in self.meta_lines:
            try: line.remove()
            except: pass
        self.meta_lines.clear()
        
        self._clear_markers()
        self.ax21.set_xlim(0, 10)
        self.ax11.set_xlim(0, 10)
        self.canvas.draw()

    def _update(self, t, s21, s11):
        self.last_t = t
        self.x_time.append(t)
        self.y21r.append(s21)
        self.y11r.append(s11)
        
        self.obj_history.append(self.meta_obj)
        self.height_history.append(self.meta_height)

        x_arr = np.array(self.x_time)
        
        self.l21.set_data(x_arr, smooth_trace(np.array(self.y21r), SMOOTH_WIN21))
        self.d21.set_data(x_arr, self.y21r)

        y_raw = np.array(self.y11r)
        valid = np.isfinite(y_raw)
        
        if valid.sum() >= 2:
            y_filled = y_raw.copy()
            y_filled[~valid] = np.interp(x_arr[~valid], x_arr[valid], y_raw[valid])
        else:
            y_filled = y_raw.copy()
            
        if self.cbSmooth.isChecked():
            y_plot = smooth_trace(y_filled, SMOOTH_WIN11)
        else:
            y_plot = y_filled

        y_plot = np.minimum(y_plot, 0.0)

        self.l11.set_data(x_arr, y_plot)
        if self.cbRaw.isChecked():
            self.d11.set_data(x_arr[valid], y_raw[valid])
        else:
            self.d11.set_data([], [])

        max_t = max(10, t * 1.05)
        for ax in (self.ax21, self.ax11):
            ax.set_xlim(0, max_t)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            
        self.canvas.draw_idle()

    def _vis_toggle(self):
        showS = self.cbSmooth.isChecked()
        showR = self.cbRaw.isChecked()
        self.l21.set_visible(showS); self.l11.set_visible(showS)
        self.d21.set_visible(showR); self.d11.set_visible(showR)
        self.canvas.draw_idle()

    def _clear_markers(self):
        for m in self.markers: m.remove()
        self.markers.clear()

    def _on_click(self, event):
        if event.inaxes not in (self.ax21, self.ax11):
            return
        ax = event.inaxes
        if event.button == 1:
            xdata = self.x_time
            if ax is self.ax21:
                ydata = self.y21r if self.d21.get_visible() else list(self.l21.get_ydata())
            else:
                ydata = self.y11r if self.d11.get_visible() else list(self.l11.get_ydata())
                
            if not xdata:
                return
            idx = int(np.argmin(np.abs(np.array(xdata) - event.xdata)))
            x = xdata[idx]; y = ydata[idx]
            
            if np.isnan(y):
                return
                
            mrk = ax.plot(x, y, 'kx', ms=8, mew=2)[0]
            txt = ax.annotate(f"{y:.2f} dB\n{x:.1f} s",
                              (x, y),
                              textcoords="offset points",
                              xytext=(5, 5),
                              fontsize=8,
                              color='k',
                              bbox=dict(boxstyle="round,pad=0.2", fc='w', alpha=0.7))
            self.markers.extend([mrk, txt])
            self.canvas.draw_idle()
        elif event.button == 3:
            self._clear_markers()
            self.canvas.draw_idle()

    def closeEvent(self, e):
        self.stop_scan()
        sdr.tx_destroy_buffer(); sdr.rx_destroy_buffer()
        e.accept()

# ───────────── main ─────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        win = VNA()
    except Exception:
        traceback.print_exc()
        QMessageBox.critical(None, "Init error", "Failed to start:\n" + traceback.format_exc())
        sys.exit(1)
    win.show()
    sys.exit(app.exec())
