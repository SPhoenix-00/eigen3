"""Process single-instrument CSV: compute technical indicators, output
production-ready CSV and PKL.

Replicates the exact Eigen2 VBA indicator calculations for three base series
extracted from the input CSV:
  - Close price  (Column B)
  - PE NTM       (Column D)
  - VIX          (Column AF)

For each series the full indicator stack is computed:
  RSI, MACD, MACD_Signal, TRIX, diff20DMA
using the same EMA initialisation (SMA seed) and smoothing constants as Eigen2.

Input CSV layout (Excel-style references):
    Rows 1-3   : headers / metadata (skipped)
    Row 4+     : trading-day data
    Column A   : date
    Column B   : price at close
    Column D   : PE NTM (next-twelve-months P/E)
    Column AF  : VIX

Output: flat CSV / PKL with date index and 18 data columns (6 per series):
        raw, RSI, MACD, MACD_Signal, TRIX, diff20DMA
"""

import pandas as pd
import numpy as np
import time
import sys
import os
import argparse
from datetime import datetime

# --- Configuration ---
INPUT_FILE = 'Eigen3mono_master_v1.csv'
OUTPUT_FILE_PKL = 'Eigen3_Processed_OUTPUT.pkl'
OUTPUT_FILE_CSV = 'Eigen3_Processed_OUTPUT.csv'

# CSV structure (Excel-style references)
CSV_HEADER_ROW = 2    # 0-indexed; Excel row 3 is the header, data starts row 4

# Column positions (Excel column letter -> 0-based index)
COL_DATE   = 0     # A
COL_CLOSE  = 1     # B
COL_PE_NTM = 3     # D
COL_VIX    = 31    # AF

BASE_SERIES = [
    ('close',  COL_CLOSE),
    ('pe_ntm', COL_PE_NTM),
    ('vix',    COL_VIX),
]

WARMUP_ROWS = 34   # Ramp-up rows discarded from output (same as Eigen2)


# ---------------------------------------------------------------------------
# Indicator computation — exact Eigen2 / VBA logic
# ---------------------------------------------------------------------------

def compute_indicators(v):
    """Compute the full Eigen2 indicator stack for a single value series.

    Replicates the VBA calculation order and initialisation:
      1. RSI   (7-period EMA of up/down moves, alpha = 2/8, SMA seed at row 7)
      2. EMA12 (alpha = 2/13,  SMA seed at row 11)
      3. EMA26 (alpha = 2/27,  SMA seed at row 25)
      4. MACD  = EMA12 - EMA26
      5. MACD Signal (9-period EMA of MACD, alpha = 2/10, SMA seed at row 33)
      6. EMA(EMA12)       — "EMA12EMA"    (alpha = 2/13, SMA seed at row 22)
      7. EMA(EMA(EMA12))  — "EMA12EMAEMA" (alpha = 2/13, SMA seed at row 33)
      8. TRIX  = 100 * (EMA12EMAEMA_curr / EMA12EMAEMA_prev - 1)
      9. x20DMA  = 20-day SMA
     10. DiffDMA = value - x20DMA

    Args:
        v: 1-D array-like of raw series values (length N).

    Returns:
        dict with keys 'rsi', 'macd', 'macd_signal', 'trix', 'diff20dma',
        each a 1-D numpy array of length N (NaN where not yet computable).
    """
    v = np.asarray(v, dtype=float)
    n = len(v)

    def _sma(arr, start, end):
        """Mean of non-NaN values in arr[start:end]."""
        chunk = arr[start:end]
        valid = chunk[~np.isnan(chunk)]
        return np.mean(valid) if len(valid) > 0 else np.nan

    # ---- RSI_U / RSI_D ----
    rsi_u = np.zeros(n)
    rsi_d = np.zeros(n)
    for i in range(1, n):
        if np.isnan(v[i]) or np.isnan(v[i - 1]):
            rsi_u[i] = 0.0
            rsi_d[i] = 0.0
        else:
            diff = v[i] - v[i - 1]
            rsi_u[i] = max(0.0, diff)
            rsi_d[i] = max(0.0, -diff)

    # ---- RSI_U_EMA7 / RSI_D_EMA7  (alpha = 2/8) ----
    rsi_u_ema = np.full(n, np.nan)
    rsi_d_ema = np.full(n, np.nan)
    if n > 7:
        rsi_u_ema[7] = np.mean(rsi_u[1:8])   # 7 values: transitions 0->1 .. 6->7
        rsi_d_ema[7] = np.mean(rsi_d[1:8])
    for i in range(8, n):
        rsi_u_ema[i] = rsi_u_ema[i - 1] + (2.0 / 8) * (rsi_u[i] - rsi_u_ema[i - 1])
        rsi_d_ema[i] = rsi_d_ema[i - 1] + (2.0 / 8) * (rsi_d[i] - rsi_d_ema[i - 1])

    # ---- RSI ----
    rsi = np.full(n, np.nan)
    for i in range(7, n):
        u, d = rsi_u_ema[i], rsi_d_ema[i]
        if np.isnan(u) or np.isnan(d) or (u + d) == 0:
            rsi[i] = 0.0
        else:
            rsi[i] = 100.0 * u / (u + d)

    # ---- EMA12  (alpha = 2/13) ----
    ema12 = np.full(n, np.nan)
    if n > 11:
        ema12[11] = _sma(v, 0, 12)
    for i in range(12, n):
        if np.isnan(ema12[i - 1]):
            ema12[i] = _sma(v, max(0, i - 11), i + 1)
        else:
            ema12[i] = (v[i] - ema12[i - 1]) * (2.0 / 13) + ema12[i - 1]

    # ---- EMA26  (alpha = 2/27, SMA seed only at row 25) ----
    ema26 = np.full(n, np.nan)
    if n > 25:
        ema26[25] = _sma(v, 0, 26)
    for i in range(26, n):
        if np.isnan(ema26[i - 1]):
            pass                    # no fallback — stays NaN (matches VBA)
        else:
            ema26[i] = (v[i] - ema26[i - 1]) * (2.0 / 27) + ema26[i - 1]

    # ---- MACD ----
    macd = np.full(n, np.nan)
    for i in range(25, n):
        if not np.isnan(ema12[i]) and not np.isnan(ema26[i]):
            macd[i] = ema12[i] - ema26[i]

    # ---- MACD Signal  (alpha = 2/10, 9-period EMA of MACD) ----
    macd_signal = np.full(n, np.nan)
    if n > 33:
        macd_signal[33] = _sma(macd, 25, 34)
    for i in range(34, n):
        if np.isnan(macd_signal[i - 1]) or np.isnan(macd[i]):
            macd_signal[i] = _sma(macd, max(0, i - 8), i + 1)
        else:
            macd_signal[i] = (macd[i] - macd_signal[i - 1]) * (2.0 / 10) + macd_signal[i - 1]

    # ---- EMA(EMA12) — "EMA12EMA"  (alpha = 2/13) ----
    ema12ema = np.full(n, np.nan)
    if n > 22:
        ema12ema[22] = _sma(ema12, 11, 23)
    for i in range(23, n):
        if np.isnan(ema12ema[i - 1]) or np.isnan(ema12[i]):
            ema12ema[i] = _sma(ema12, max(0, i - 11), i + 1)
        else:
            ema12ema[i] = (ema12[i] - ema12ema[i - 1]) * (2.0 / 13) + ema12ema[i - 1]

    # ---- EMA(EMA(EMA12)) — "EMA12EMAEMA"  (alpha = 2/13) ----
    ema12emaema = np.full(n, np.nan)
    if n > 33:
        ema12emaema[33] = _sma(ema12ema, 22, 34)
    for i in range(34, n):
        if np.isnan(ema12emaema[i - 1]) or np.isnan(ema12ema[i]):
            ema12emaema[i] = _sma(ema12ema, max(0, i - 11), i + 1)
        else:
            ema12emaema[i] = (ema12ema[i] - ema12emaema[i - 1]) * (2.0 / 13) + ema12emaema[i - 1]

    # ---- TRIX ----
    trix = np.full(n, np.nan)
    for i in range(34, n):
        curr = ema12emaema[i]
        prev = ema12emaema[i - 1]
        if not np.isnan(curr) and not np.isnan(prev) and prev != 0:
            trix[i] = 100.0 * (curr / prev - 1.0)

    # ---- x20DMA (20-day SMA) ----
    x20dma = np.full(n, np.nan)
    for i in range(19, n):
        x20dma[i] = _sma(v, i - 19, i + 1)

    # ---- DiffDMA ----
    diff_dma = np.full(n, np.nan)
    for i in range(19, n):
        if not np.isnan(v[i]) and not np.isnan(x20dma[i]):
            diff_dma[i] = v[i] - x20dma[i]

    return {
        'rsi':         rsi,
        'macd':        macd,
        'macd_signal': macd_signal,
        'trix':        trix,
        'diff20dma':   diff_dma,
    }


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

def print_progress(current, total, start_time, label='Series'):
    elapsed = time.time() - start_time
    pct = current / total * 100
    sys.stdout.write(
        f'\r  {label} {current}/{total} ({pct:.0f}%) | '
        f'Elapsed: {time.strftime("%H:%M:%S", time.gmtime(elapsed))}'
    )
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_csv(input_file, output_csv, output_pkl):
    """Read the input CSV, compute indicators for each base series, write output."""

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return

    # ---- Step 1: Read CSV ----
    print(f"Step 1: Loading {input_file} ...")
    df_raw = pd.read_csv(input_file, header=CSV_HEADER_ROW)
    num_rows = len(df_raw)
    print(f"  {num_rows} data rows loaded (Excel rows 4–{num_rows + 3})")

    # Extract dates (column A) for the output index
    dates = df_raw.iloc[:, COL_DATE].values

    # ---- Step 2: Extract base series & compute indicators ----
    print(f"Step 2: Computing indicators for {len(BASE_SERIES)} series ...")
    start_time = time.time()
    result_columns = {}

    for idx, (name, col_idx) in enumerate(BASE_SERIES):
        print_progress(idx + 1, len(BASE_SERIES), start_time)

        raw_values = pd.to_numeric(df_raw.iloc[:, col_idx], errors='coerce').values
        indicators = compute_indicators(raw_values)

        result_columns[name]                    = raw_values
        result_columns[f'{name}_rsi']           = indicators['rsi']
        result_columns[f'{name}_macd']          = indicators['macd']
        result_columns[f'{name}_macd_signal']   = indicators['macd_signal']
        result_columns[f'{name}_trix']          = indicators['trix']
        result_columns[f'{name}_diff20dma']     = indicators['diff20dma']

    sys.stdout.write('\n')
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.2f}s")

    # ---- Step 3: Build output DataFrame ----
    print("Step 3: Building output DataFrame ...")
    df_out = pd.DataFrame(result_columns, index=dates)
    df_out.index.name = 'date'

    # ---- Step 4: Slice off warm-up rows ----
    print(f"Step 4: Slicing off top {WARMUP_ROWS} warm-up rows ...")
    df_out = df_out.iloc[WARMUP_ROWS:]
    print(f"  {len(df_out)} rows remaining "
          f"(dates {df_out.index[0]} – {df_out.index[-1]})")

    # ---- Step 5: Save outputs ----
    print(f"Step 5: Saving outputs ...")

    # PKL
    print(f"  Pickle -> {output_pkl}")
    df_out.to_pickle(output_pkl)

    # CSV
    print(f"  CSV    -> {output_csv}")
    df_out.to_csv(output_csv)

    # ---- Step 6: Sanity check ----
    print("\n--- Sanity Check ---")
    df_reload = pd.read_pickle(output_pkl)
    print(f"  Rows in PKL on disk : {len(df_reload)}")
    print(f"  Columns             : {list(df_reload.columns)}")

    print(f"\n  Per-column stats (non-NaN rows):")
    for col in df_reload.columns:
        s = df_reload[col].dropna()
        if len(s) > 0:
            print(f"    {col:>25s}:  n={len(s):5d}  "
                  f"min={s.min():12.4f}  max={s.max():12.4f}  "
                  f"mean={s.mean():12.4f}")
        else:
            print(f"    {col:>25s}:  (all NaN)")

    print(f"\nSuccess!  Pickle: {output_pkl}  |  CSV: {output_csv}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Process mono CSV: compute indicators and output production CSV/PKL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_eigen_data.py
  python process_eigen_data.py --input my_data.csv
  python process_eigen_data.py --input my_data.csv --output-csv out.csv --output-pkl out.pkl
        """,
    )
    parser.add_argument('--input', type=str, default=INPUT_FILE,
                        help=f'Input CSV file (default: {INPUT_FILE})')
    parser.add_argument('--output-csv', type=str, default=OUTPUT_FILE_CSV,
                        help=f'Output CSV file (default: {OUTPUT_FILE_CSV})')
    parser.add_argument('--output-pkl', type=str, default=OUTPUT_FILE_PKL,
                        help=f'Output PKL file (default: {OUTPUT_FILE_PKL})')

    args = parser.parse_args()
    process_csv(args.input, args.output_csv, args.output_pkl)


if __name__ == '__main__':
    main()
