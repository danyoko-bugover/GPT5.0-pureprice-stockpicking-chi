#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_csi300_prices.py - A股滬深300成分股價格下載工具（一檔一檔存版）
檔名格式：純6位數字.csv（例如 600000.csv），直接存放在 data/ 底下
執行：python save_csi300_prices.py [--days 365] [--sleep 0.1]
"""

import os
import time
import sys
from datetime import datetime, timedelta
import argparse

import pandas as pd
import yfinance as yf

# 只在需要時 import akshare
DATA_DIR = "data"
# 不再使用 STOCK_DIR 子資料夾，直接用 DATA_DIR

DEFAULT_SLEEP = 0.1
DEFAULT_DAYS = 365
MAX_RETRY = 2


def get_csi300_tickers():
    """取得最新滬深300成分股代碼"""
    print("正在取得最新滬深300成分股...")
    try:
        import akshare as ak
        df = ak.index_stock_cons(symbol="000300")
        tickers = df['品种代码'].astype(str).str.zfill(6).tolist()
        print(f"成功取得 {len(tickers)} 檔成分股")
        return tickers
    except Exception as e:
        print(f"akshare 取得成分股失敗: {e}")
        print("請 pip install akshare --upgrade，或從中證官網手動下載成分股清單")
        raise


def download_with_yfinance(code: str, days: int):
    """yfinance 下載主邏輯"""
    for attempt in range(MAX_RETRY + 1):
        try:
            end = datetime.now().date()
            start = end - timedelta(days=days)
            suffix = ".SZ" if code[0] in ('0', '3', '4') else ".SS"
            ticker = code + suffix
            
            df = yf.download(ticker, start=start, end=end + timedelta(days=1), progress=False)
            if not df.empty:
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            if attempt == MAX_RETRY:
                raise
            time.sleep(2.0)
    return None


def main(days: int = DEFAULT_DAYS, sleep: float = DEFAULT_SLEEP):
    os.makedirs(DATA_DIR, exist_ok=True)
    
    tickers = get_csi300_tickers()
    print(f"\n開始下載 {len(tickers)} 檔 A股（一檔一檔存，預計耗時約 {len(tickers) * sleep / 60:.1f} 分鐘）\n")
    
    failed = []
    success_count = 0

    for i, code in enumerate(tickers, 1):
        print(f"[{i:3d}/{len(tickers)}] {code} ... ", end="")
        sys.stdout.flush()

        try:
            df = download_with_yfinance(code, days)
            
            if df is None or df.empty:
                print("失敗（資料為空）")
                failed.append(code)
                time.sleep(sleep)
                continue

            # 重設索引並加入 Ticker 欄位
            df_reset = df.reset_index()
            df_reset.insert(0, "Ticker", code)

            # 存檔：直接放在 data/ 底下，純數字檔名
            filename = os.path.join(DATA_DIR, f"{code}.csv")
            df_reset.to_csv(filename, index=False)

            print(f"成功 ({len(df)} 列) → {filename}")
            success_count += 1

        except Exception as e:
            print(f"失敗: {e}")
            failed.append(code)

        time.sleep(sleep)
    
    # 最終統計
    print("\n" + "="*60)
    print(f"下載完成！成功 {success_count} / {len(tickers)} 檔")
    print(f"檔案儲存位置：{DATA_DIR}")
    if failed:
        print(f"失敗 {len(failed)} 檔：{failed}")
    else:
        print("全部成功！")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下載滬深300成分股價格（一檔一檔存版）")
    parser.add_argument("--days", "-d", type=int, default=DEFAULT_DAYS, help="下載天數")
    parser.add_argument("--sleep", "-s", type=float, default=DEFAULT_SLEEP, help="每檔間隔秒數")
    args = parser.parse_args()
    
    main(days=args.days, sleep=args.sleep)