#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_csi300_prices.py - A股滬深300成分股價格下載工具（優先 yfinance 版）
優化合併與寫檔速度，適合大表
執行：python save_csi300_prices.py [--days 365] [--sleep 1.0]
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
DEFAULT_SLEEP = 1.0          # 加大防 rate limit
DEFAULT_DAYS = 365
MAX_RETRY = 2
CHUNKSIZE_WRITE = 50000      # to_csv 分塊寫入，加速大檔

def get_csi300_tickers():
    """取得最新滬深300成分股代碼"""
    print("正在取得最新滬深300成分股...")
    try:
        import akshare as ak
        df = ak.index_stock_cons(symbol="000300")  # 穩定接口
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
            suffix = ".SZ" if code[0] in ('0', '3') else ".SS"
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
    print(f"\n開始下載 {len(tickers)} 檔 A股（優先 yfinance，預計耗時約 {len(tickers) * sleep / 60:.1f} 分鐘）\n")
    
    failed = []
    all_dfs = []  # 用 list 收集
    
    for i, code in enumerate(tickers, 1):
        print(f"[{i:3d}/{len(tickers)}] {code} ... ", end="")
        sys.stdout.flush()
        
        df = None
        source = ""
        
        try:
            df = download_with_yfinance(code, days)
            source = "yfinance"
        except Exception as e:
            print(f"yfinance 失敗: {e}", end=" ")
        
        if df is None or df.empty:
            print("失敗")
            failed.append(code)
            time.sleep(sleep)
            continue
        
        df_reset = df.reset_index()
        df_reset.insert(0, "Ticker", code)
        all_dfs.append(df_reset)
        
        print(f"成功 ({len(df)} 列) [{source}]")
        sys.stdout.flush()
        
        time.sleep(sleep)
    
    if not all_dfs:
        print("\n沒有任何資料下載成功。")
        return
    
    print("\n所有下載完成，開始合併資料（這可能需要 30秒～幾分鐘）...")
    sys.stdout.flush()
    
    # 分批 concat 減少記憶體峰值
    combined = pd.concat(all_dfs, ignore_index=True, copy=False)
    print("合併完成，開始寫入合併檔（分塊寫入，較快）...")
    sys.stdout.flush()
    
    combined_path = os.path.join(DATA_DIR, "all_csi300_last_year.csv")
    combined.to_csv(combined_path, index=False, chunksize=CHUNKSIZE_WRITE)
    
    print(f"合併檔已儲存：{combined_path}")
    print(f"總行數：{len(combined)}")
    
    if failed:
        print(f"\n下載失敗 {len(failed)} 檔：{failed}")
    else:
        print("\n全部完成！腳本結束。")
    
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下載滬深300成分股近一年價格（優先 yfinance）")
    parser.add_argument("--days", "-d", type=int, default=DEFAULT_DAYS, help="下載天數")
    parser.add_argument("--sleep", "-s", type=float, default=DEFAULT_SLEEP, help="每檔間隔秒數")
    args = parser.parse_args()
    
    main(days=args.days, sleep=args.sleep)