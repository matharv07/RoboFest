#!/usr/bin/env python3
"""
PlutoSDR VNA Data Plotter - Real-Time DSP Viewer (Auto-Save)
=========================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
import os

def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select VNA Data CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )

    if not file_path:
        sys.exit(0)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 1. Extract columns matching the new C++ CSV format
    t = df['Time'].values
    s21_raw = df['S21_dB'].values
    s21_gauss = df['S21_Gaussian'].values
    s21_savgol = df['S21_Gauss_SavGol'].values

    # 2. Rolling Std Dev (Center=True uses 5 points before and 5 points after)
    s21_std = pd.Series(s21_raw).rolling(window=11, center=True, min_periods=1).std().values

    # 3. Fixed Metadata Detection: Keep dimensions aligned
    obj_changes = df['OBJ'].ne(df['OBJ'].shift())
    h_changes = df['Height'].ne(df['Height'].shift())
    
    combined_changes = obj_changes | h_changes
    meta_change_indices = df.index[combined_changes].tolist()
    if 0 in meta_change_indices:
        meta_change_indices.remove(0)

    # 4. Plotting (2 subplots instead of 3, since we only have S21)
    fig, (ax21, ax_std) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # --- Top Plot: S21 Processing Stages ---
    ax21.plot(t, s21_raw, 'o', color='lightgray', ms=3, alpha=0.5, label='S21 Raw')
    ax21.plot(t, s21_gauss, '-', color='cornflowerblue', lw=1.5, alpha=0.8, label='S21 Gaussian')
    ax21.plot(t, s21_savgol, '-', color='crimson', lw=2, label='S21 Gauss+SavGol')
    
    ax21.set_ylabel("S21 (dB)")
    ax21.set_title(f"Real-Time DSP: {os.path.basename(file_path)}")
    ax21.grid(True, alpha=0.3)
    ax21.legend(loc='lower right')

    # --- Bottom Plot: Standard Deviation ---
    ax_std.plot(t, s21_std, 'r-', lw=1, label='Raw S21 Std Dev (±5 pts)')
    ax_std.set_ylabel("Std Dev (dB)")
    ax_std.set_xlabel("Time (seconds)")
    ax_std.grid(True, alpha=0.3)
    ax_std.legend(loc='upper right')

    # 5. Add Markers for Object/Height Changes
    for idx in meta_change_indices:
        change_t = t[idx]
        label_text = f" OBJ:{df['OBJ'].iloc[idx]}\n H:{df['Height'].iloc[idx]}m"
        
        for i, ax in enumerate((ax21, ax_std)):
            ax.axvline(x=change_t, color='orange', linestyle='--', alpha=0.7)
            if i == 0:
                ax.text(change_t, 0.98, label_text, color='darkorange', transform=ax.get_xaxis_transform(),
                        va='top', ha='right', fontsize=8, bbox=dict(boxstyle="round", fc='w', ec='orange', alpha=0.8))

    plt.tight_layout()

    # --- AUTO-SAVE LOGIC ---
    # Strip the .csv extension and add .png
    base_path = os.path.splitext(file_path)[0]
    img_path = f"{base_path}.png"

    # Save only if the image doesn't already exist
    if not os.path.exists(img_path):
        print(f"Saving plot image to: {img_path}")
        plt.savefig(img_path, dpi=300, bbox_inches='tight') # dpi=300 keeps it crisp
    else:
        print(f"Image already exists at: {img_path}. Skipping save.")

    # Display the interactive window
    plt.show()

if __name__ == "__main__":
    main()
