#!/usr/bin/env python3
"""
PlutoSDR VNA Data Plotter
=========================
Reads the generated CSV files and reproduces the S21 and S11 vs Time plots,
including reconstructing the metadata change markers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys

# --- Smoothing Function (same as the main app) ---
def smooth_trace(y, k):
    if k <= 1 or len(y) < k:
        return y
    y_arr = np.array(y)
    mask = np.isfinite(y_arr).astype(float)
    win = np.ones(k)
    num = np.convolve(np.nan_to_num(y_arr), win, 'same')
    den = np.convolve(mask, win, 'same')
    out = np.full_like(y_arr, np.nan)
    good = den > 0
    out[good] = num[good] / den[good]
    return out

def main():
    # Hide the main Tkinter window
    root = tk.Tk()
    root.withdraw()

    # Open file dialog to pick the CSV
    print("Please select a VNA CSV file to plot...")
    file_path = filedialog.askopenfilename(
        title="Select VNA Data CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit(0)

    print(f"Loading data from: {file_path}")
    
    # Read the data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    # Validate columns
    req_cols = ['Elapsed Time (s)', 'Frequency (Hz)', 'S11 (dB)', 'S21 (dB)', 'OBJ', 'Height (m)']
    if not all(col in df.columns for col in req_cols):
        print(f"Error: CSV is missing required columns. Expected: {req_cols}")
        sys.exit(1)

    # Extract data
    t = df['Elapsed Time (s)'].values
    freq_hz = df['Frequency (Hz)'].iloc[0]
    freq_mhz = freq_hz / 1e6
    
    s21_raw = df['S21 (dB)'].values
    s11_raw = df['S11 (dB)'].values
    
    s21_smooth = smooth_trace(s21_raw, 7)
    s11_smooth = smooth_trace(s11_raw, 5)

    # Identify where metadata changed
    meta_change_indices = []
    for i in range(1, len(df)):
        if (df['OBJ'].iloc[i] != df['OBJ'].iloc[i-1]) or (df['Height (m)'].iloc[i] != df['Height (m)'].iloc[i-1]):
            meta_change_indices.append(i)

    # Setup the plot
    fig, (ax21, ax11) = plt.subplots(2, 1, figsize=(12, 9))

    # Plot S21
    ax21.plot(t, s21_raw, 'ro', ms=3, alpha=0.3, label='S21 raw')
    ax21.plot(t, s21_smooth, 'b-', lw=1.5, label='S21 smoothed')
    ax21.set_title(f"S21 vs Time ({freq_mhz} MHz)")
    ax21.set_ylabel("S21 (dB)")
    ax21.grid(True)
    ax21.legend(loc='lower right')

    # Plot S11
    ax11.plot(t, s11_raw, 'mo', ms=3, alpha=0.3, label='S11 raw')
    ax11.plot(t, s11_smooth, 'g-', lw=1.5, label='S11 smoothed')
    ax11.set_title(f"S11 vs Time ({freq_mhz} MHz)")
    ax11.set_ylabel("S11 (dB)")
    ax11.set_xlabel("Elapsed Time (s)")
    ax11.grid(True)
    ax11.legend(loc='lower right')

    # Add Metadata Markers
    for idx in meta_change_indices:
        change_t = t[idx]
        obj_val = df['OBJ'].iloc[idx]
        h_val = df['Height (m)'].iloc[idx]
        label_text = f" Meta\n OBJ:{obj_val}\n H:{h_val}m"

        for ax in (ax21, ax11):
            ax.axvline(x=change_t, color='orange', linestyle='--', alpha=0.8)
            # Add text annotation at the top of the graph
            ax.text(change_t, 0.95, label_text, color='orange', 
                    transform=ax.get_xaxis_transform(), va='top', ha='right', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc='w', ec='orange', alpha=0.7))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
