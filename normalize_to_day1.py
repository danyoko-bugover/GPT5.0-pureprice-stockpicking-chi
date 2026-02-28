#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalize_to_day1.py - 以「第一個有效交易日的收盤價 = 100」標準化

功能：
- 找到第一個 Close 不為 NaN 且 > 0 的交易日作為基準
- 只正規化 Open/High/Low/Close/Volume（不影響 Ticker 等欄位）
- 若整檔無有效 Close → 警告並輸出原檔
- 支援 A股 6 位數字 ticker（如 600519.csv）

使用：
    python normalize_to_day1.py
"""

import os
import sys
import glob
import pandas as pd

INPUT_DIR = "data"
OUTPUT_DIR = "data_processed"
CSV_GLOB = "*.csv"

TARGET_COLS = ["Open", "High", "Low", "Close", "Volume"]


def find_case_insensitive_cols(df, target_cols):
    """不區分大小寫找到欄位對應"""
    cols_lower = {c.lower(): c for c in df.columns}
    return {t: cols_lower.get(t.lower()) for t in target_cols if t.lower() in cols_lower}


def normalize_df_day1(df, col_map, filename):
    """
    以第一個有效 Close 為基準正規化
    返回處理後的 DataFrame
    """
    out = df.copy()

    # 先確認有沒有 Close 欄位
    close_col = col_map.get("Close")
    if not close_col:
        print(f"  警告：{filename} 沒有 Close 欄位，無法正規化")
        return out

    # 轉成 numeric，錯誤變 NaN
    close_series = pd.to_numeric(out[close_col], errors='coerce')

    # 找到第一個有效 Close（非 NaN 且 > 0）
    valid_mask = (close_series.notna()) & (close_series > 0)
    if not valid_mask.any():
        print(f"  警告：{filename} 全檔 Close 無有效值（全 NaN 或 ≤0），跳過正規化")
        return out

    # 第一個有效交易日的索引
    first_valid_idx = close_series[valid_mask].index[0]
    base_value = close_series.loc[first_valid_idx]

    print(f"  {filename} 使用第 {first_valid_idx + 1} 筆資料作為基準 (Close = {base_value})")

    # 正規化所有目標欄位
    for std_col, actual_col in col_map.items():
        series = pd.to_numeric(out[actual_col], errors='coerce')
        # 只對有效值做正規化，無效值保持 NaN
        out[actual_col] = (series / base_value) * 100.0

    return out


if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"錯誤：輸入資料夾 '{INPUT_DIR}' 不存在")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(INPUT_DIR, CSV_GLOB)))
    if not files:
        print(f"在 {INPUT_DIR} 找不到任何 CSV 檔案")
        sys.exit(1)

    print(f"找到 {len(files)} 個 CSV 檔案，開始處理...\n")

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"處理：{filename}")

        try:
            df = pd.read_csv(
                filepath,
                parse_dates=['Date'],
                date_format='%Y-%m-%d',
                low_memory=False
            )
        except Exception as e:
            print(f"  錯誤：讀取失敗 {e}，跳過")
            continue

        if df.empty:
            print(f"  警告：檔案為空，跳過")
            continue

        col_map = find_case_insensitive_cols(df, TARGET_COLS)
        if not col_map:
            print(f"  警告：沒有找到任何 OHLCV 欄位，直接複製原檔")
            out_df = df
        else:
            out_df = normalize_df_day1(df, col_map, filename)

        # 寫出
        out_path = os.path.join(OUTPUT_DIR, filename)
        try:
            out_df.to_csv(out_path, index=False)
            print(f"  已寫入：{out_path}")
        except Exception as e:
            print(f"  寫入失敗：{e}")

    print("\n全部處理完成。")