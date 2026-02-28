#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rename_and_merge_csvs.py - 分批合併正規化後的 CSV

功能：
- 讀取 data_processed/ 裡的所有 CSV
- 為每檔隨機產生 5 位英文字母代號（匿名）
- 產生 mapping.txt 對照表
- 分批合併（每批 10 檔）
- 輸出到 input/merged_part_xx.csv

使用：
    python rename_and_merge_csvs.py
"""

import os
import sys
import glob
import random
import string
import pandas as pd

INPUT_DIR = "data_processed"
OUTPUT_DIR = "input"
MAPPING_FILE = "mapping.txt"
BATCH_SIZE = 10  # 每批合併 10 檔，防記憶體問題

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_anonymous_code():
    """產生隨機 5 位大寫英文字母代號"""
    return ''.join(random.choices(string.ascii_uppercase, k=5))


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
    if not files:
        print(f"在 {INPUT_DIR} 找不到任何 CSV 檔案")
        sys.exit(1)

    total_files = len(files)
    print(f"找到 {total_files} 個正規化 CSV 檔案，開始匿名化與分批合併...\n")

    mapping = {}  # 原檔名 -> 匿名代號
    batch_dfs = []
    part_num = 1
    processed = 0

    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"[{processed + 1}/{total_files}] 處理 {filename}", end=" ")
        sys.stdout.flush()

        try:
            df = pd.read_csv(
                filepath,
                parse_dates=['Date'],
                date_format='%Y-%m-%d',
                low_memory=False
            )
        except Exception as e:
            print(f"讀取失敗：{e}")
            continue

        if df.empty:
            print("空檔，跳過")
            continue

        # 產生匿名代號
        anon_code = generate_anonymous_code()
        while anon_code in mapping.values():  # 避免重複
            anon_code = generate_anonymous_code()
        mapping[filename.replace('.csv', '')] = anon_code

        # 重命名欄位為 匿名代號_原欄位
        rename_dict = {col: f"{anon_code}_{col}" for col in df.columns if col != 'Date' and col != 'Ticker'}
        df = df.rename(columns=rename_dict)

        batch_dfs.append(df)

        processed += 1

        # 每 BATCH_SIZE 檔合併一次
        if len(batch_dfs) >= BATCH_SIZE or processed == total_files:
            print(" → 合併批次...")
            sys.stdout.flush()

            # 一次 concat axis=1
            merged = batch_dfs[0].copy()
            for d in batch_dfs[1:]:
                merged = pd.merge(merged, d, on='Date', how='outer', suffixes=('', '_dup'))
                # 移除重複欄位（如果有）
                merged = merged.loc[:, ~merged.columns.str.endswith('_dup')]

            # 排序日期
            merged = merged.sort_values('Date').reset_index(drop=True)

            out_path = os.path.join(OUTPUT_DIR, f"merged_part_{part_num:02d}.csv")
            merged.to_csv(out_path, index=False)
            print(f"輸出：{out_path}")
            sys.stdout.flush()

            batch_dfs = []  # 清空批次
            part_num += 1

    # 寫 mapping.txt
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        for orig, anon in sorted(mapping.items()):
            f.write(f"{orig} -> {anon}\n")
    print(f"\nmapping.txt 已產生：{MAPPING_FILE}")

    print("\n" + "="*60)
    print(f"全部完成！總共產生 {part_num-1} 個 merged_part_xx.csv")
    print("輸出資料夾：", OUTPUT_DIR)
    print("="*60)
    sys.stdout.flush()

    sys.exit(0)


if __name__ == "__main__":
    main()