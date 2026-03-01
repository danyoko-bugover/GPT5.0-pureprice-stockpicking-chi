#!/usr/bin/env python3
# pick_top20.py
"""
多因子選股（保留 ticket 欄位；完全忽略 'Ticker' 欄位）
- 讀取 data_dir 下所有 CSV（可能多個檔案），合併以 ticket 區分資產時間序列
- 計算每個 ticket 的因子（需要至少 min_history_days）
- 標準化、加權計分，輸出包含 ticket 的結果 CSV

執行：
python pick_top20.py --data_dir ./input --out_dir ./output
"""
import os
import glob
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------------- CONFIG ----------------
CFG = {
    "momentum_windows": [252, 90, 60, 20, 5],
    "ma_windows": [20, 60, 200],
    "vol_windows": [90, 30],
    "rsi_window": 14,
    "atr_window": 14,
    "min_history_days": 120,
    "top_n": 20,
    "weights": {
        "mom_252": 0.12,
        "mom_90": 0.10,
        "mom_60": 0.06,
        "mom_20": 0.06,
        "mom_5": 0.03,
        "ma_ratio_s_l": 0.06,
        "price_vs_ma_long": 0.04,
        "ann_vol_90": 0.08,
        "vol_30": 0.03,
        "max_drawdown": 0.08,
        "downside_dev": 0.03,
        "avg_vol_60": 0.03,
        "vol_spike": 0.02,
        "mom_accel": 0.04,
        "volatility_change": 0.02,
        "rsi_14": 0.02,
        "atr_14": 0.02,
        "sharpe_like": 0.10,
        "calmar": 0.05,
        "trend_r2": 0.05,
        "consec_up_10": 0.02,
        "skewness": 0.0,
        "kurtosis": 0.0
    }
}

# ---------------- UTIL ----------------
def read_all_csvs(data_dir):
    """
    Read all CSV files in data_dir and concatenate into a single DataFrame.
    Requirements: columns Date, ticket, Close, Volume. Adj Close optional. Ignore any 'Ticker' column.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    dfs = []
    for f in files:
        try:
            # read as strings first, parse Date afterwards to avoid mixed types issues
            df = pd.read_csv(f, parse_dates=["Date"], dayfirst=False, dtype=str)
        except Exception as e:
            print(f"Warning: failed reading {f}: {e}")
            continue
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        # Drop 'Ticker' column entirely if present (case-insensitive)
        df = df.drop(columns=[c for c in df.columns if c.lower() == "ticker"], errors="ignore")
        # Ensure required columns exist
        required = {"Date", "ticket", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            print(f"Skipping file {f}: missing required cols {required - set(df.columns)}")
            continue
        # convert types: Date parsed, numeric columns
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        # keep ticket as string
        df["ticket"] = df["ticket"].astype(str)
        # numeric cols
        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # If Adj Close missing, fill from Close to maintain compatibility
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]
            print(f"Info: file {os.path.basename(f)} missing 'Adj Close' - filled from 'Close'")
        # drop rows with no Date or Close
        df = df.dropna(subset=["Date", "Close"])
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No valid CSV data found")
    big = pd.concat(dfs, ignore_index=True)
    # sort by ticket and date
    big = big.sort_values(["ticket", "Date"]).reset_index(drop=True)
    return big

def pct_change_safe(s):
    return s.pct_change().replace([np.inf, -np.inf], np.nan)

def rolling_downside_deviation(returns, window):
    def dd(x):
        neg = x[x < 0]
        return float(np.sqrt((neg ** 2).mean())) if len(neg) > 0 else 0.0
    return returns.rolling(window, min_periods=1).apply(dd, raw=False)

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window, min_periods=1).mean()
    ma_down = down.rolling(window, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, window=14):
    high = df["High"] if "High" in df.columns else df["Close"]
    low = df["Low"] if "Low" in df.columns else df["Close"]
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=1).mean()
    return atr

def stability_r2(log_price, window):
    if len(log_price) < 2:
        return 0.0
    y = log_price.iloc[-window:] if len(log_price) >= window else log_price
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return float(r_value ** 2)

# ---------------- FACTORS PER TICKET ----------------
def compute_factors_for_ticket(df_ticket, cfg):
    """
    df_ticket: DataFrame of rows for one ticket sorted by Date ascending.
    Returns dict of factors including 'ticket', or None if insufficient history.
    """
    if df_ticket.shape[0] < cfg["min_history_days"]:
        return None
    # price series (prefer Adj Close)
    # use safe access in case Adj Close not present (but read_all_csvs should have filled it)
    if "Adj Close" in df_ticket.columns:
        price_series = df_ticket["Adj Close"]
    else:
        price_series = df_ticket["Close"]
    price = price_series.fillna(df_ticket["Close"]).astype(float).reset_index(drop=True)
    vol = df_ticket["Volume"].astype(float).reset_index(drop=True).fillna(0.0)
    ret = pct_change_safe(price).fillna(0.0)
    logp = np.log(price.replace(0, np.nan)).dropna()
    factors = {}
    factors["ticket"] = str(df_ticket["ticket"].iloc[0])
    factors["last_date"] = df_ticket["Date"].iloc[-1]
    factors["price"] = float(price.iloc[-1])
    # momentum windows
    for w in cfg["momentum_windows"]:
        if w <= 0:
            factors[f"mom_{w}"] = 0.0
        else:
            if len(price) >= w:
                factors[f"mom_{w}"] = float(price.iloc[-1] / price.iloc[-w] - 1.0)
            else:
                factors[f"mom_{w}"] = float(price.iloc[-1] / price.iloc[0] - 1.0)
    # moving averages
    for w in cfg["ma_windows"]:
        factors[f"ma_{w}"] = float(price.rolling(window=w, min_periods=1).mean().iloc[-1])
    factors["ma_ratio_s_l"] = factors.get(f"ma_{cfg['ma_windows'][0]}", 0.0) / (factors.get(f"ma_{cfg['ma_windows'][-1]}", 1.0) + 1e-9)
    factors["price_vs_ma_long"] = factors["price"] / (factors.get(f"ma_{cfg['ma_windows'][-1]}", 1.0) + 1e-9)
    # volatility
    for w in cfg["vol_windows"]:
        sigma = ret.rolling(window=w, min_periods=2).std().iloc[-1]
        factors[f"ann_vol_{w}"] = float(sigma * np.sqrt(252)) if not pd.isna(sigma) else np.nan
        factors[f"vol_{w}"] = float(sigma) if not pd.isna(sigma) else np.nan
    # drawdown
    cummax = price.cummax()
    dd = (price / cummax - 1.0)
    factors["max_drawdown"] = float(dd.min()) if len(dd) > 0 else 0.0
    # downside deviation (90)
    factors["downside_dev"] = float(rolling_downside_deviation(ret, 90).iloc[-1])
    # volume metrics
    factors["avg_vol_20"] = float(vol.rolling(20, min_periods=1).mean().iloc[-1])
    factors["avg_vol_60"] = float(vol.rolling(60, min_periods=1).mean().iloc[-1])
    factors["vol_spike"] = float((vol.iloc[-1]) / (factors["avg_vol_20"] + 1e-9))
    # momentum accel & volatility change
    factors["mom_accel"] = factors.get("mom_20", 0.0) - factors.get("mom_60", 0.0)
    v30 = factors.get("ann_vol_30", np.nan)
    v90 = factors.get("ann_vol_90", np.nan)
    factors["volatility_change"] = float((v30 / (v90 + 1e-9)) if (not pd.isna(v30) and not pd.isna(v90)) else 0.0)
    # rsi & atr
    factors["rsi_14"] = float(compute_rsi(price, cfg["rsi_window"]).iloc[-1])
    factors["atr_14"] = float(compute_atr(df_ticket, cfg["atr_window"]).iloc[-1])
    # ann_return (from up to last 90 days)
    look = min(len(price), 90)
    total_ret_look = float(price.iloc[-1] / price.iloc[-look] - 1.0) if look > 1 else 0.0
    ann_return = float((1 + total_ret_look) ** (252.0 / look) - 1.0) if total_ret_look > -0.999 else -0.999
    factors["ann_return"] = ann_return
    # sharpe-like
    ann_vol = factors.get("ann_vol_90", np.nan)
    factors["sharpe_like"] = float(ann_return / ann_vol) if (not pd.isna(ann_vol) and ann_vol > 0) else 0.0
    # calmar
    md = factors["max_drawdown"]
    factors["calmar"] = float(ann_return / (-md)) if (md < 0 and ann_return > -0.999) else 0.0
    # trend stability r2
    try:
        factors["trend_r2"] = float(stability_r2(logp, 180))
    except Exception:
        factors["trend_r2"] = 0.0
    # consecutive up days in last 10
    last10 = price.pct_change().fillna(0).tail(10)
    consec = 0
    for x in last10.iloc[::-1]:
        if x > 0:
            consec += 1
        else:
            break
    factors["consec_up_10"] = int(consec)
    # skewness / kurtosis on last 90 returns
    rt90 = ret.tail(90)
    factors["skewness"] = float(rt90.skew()) if len(rt90) > 2 else 0.0
    factors["kurtosis"] = float(rt90.kurtosis()) if len(rt90) > 3 else 0.0
    factors["history_days"] = len(price)
    return factors

# ---------------- AGGREGATE & SCORE ----------------
def build_factor_table(big_df, cfg):
    """
    big_df: concatenated dataframe with ticket, Date, Close, etc.
    Returns factor_df (each row per ticket, ticket as a column)
    """
    rows = []
    tickets = []
    for ticket, group in big_df.groupby("ticket"):
        group = group.sort_values("Date").reset_index(drop=True)
        fac = compute_factors_for_ticket(group, cfg)
        if fac is None:
            print(f"Ticket {ticket} skipped: insufficient history ({len(group)})")
            continue
        rows.append(fac)
        tickets.append(ticket)
    if not rows:
        raise RuntimeError("No tickets with sufficient history")
    factor_df = pd.DataFrame(rows)
    # ensure ticket column exists and is first
    cols = list(factor_df.columns)
    if "ticket" in cols:
        cols = ["ticket"] + [c for c in cols if c != "ticket"]
        factor_df = factor_df[cols]
    return factor_df

def winsorize_series(s, p=0.01):
    low = s.quantile(p)
    high = s.quantile(1 - p)
    return s.clip(lower=low, upper=high)

def standardize_and_score(factor_df, cfg):
    df = factor_df.copy()
    weights = cfg["weights"]
    # define direction: 1 bigger better, -1 smaller better
    direction = {}
    for w in cfg["momentum_windows"]:
        direction[f"mom_{w}"] = 1
    direction.update({
        "ma_ratio_s_l": 1,
        "price_vs_ma_long": 1,
        "sharpe_like": 1,
        "ann_return": 1,
        "calmar": 1,
        "trend_r2": 1,
        "consec_up_10": 1,
        "rsi_14": 0,  # nearer 50 typically better; we'll transform
        "atr_14": -1,
        "ann_vol_90": -1,
        "vol_30": -1,
        "max_drawdown": -1,
        "downside_dev": -1,
        "vol_spike": -1,
        "mom_accel": 1,
        "volatility_change": -1,
        "skewness": 1,
        "kurtosis": -1,
        "avg_vol_60": 1,
        "avg_vol_20": 1
    })
    # prepare factor columns from weights keys (only keep existing)
    factor_cols = [f for f in weights.keys() if f in df.columns]
    if not factor_cols:
        raise RuntimeError("No overlap between configured weights and computed factors")
    zscores = pd.DataFrame(index=df.index)
    for col in factor_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.fillna(s.median())  # fill missing with median
        # special transform for RSI: distance to 50 (smaller is better)
        if col == "rsi_14":
            # transform so that higher is better: 100 - abs(rsi-50)*2 -> closer to 100 better
            trans = 100 - np.abs(s - 50) * 2
            s = trans
        # winsorize
        s = winsorize_series(s, p=0.01)
        mean = s.mean()
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            z = (s - mean) * 0.0
        else:
            z = (s - mean) / std
        # apply direction
        dir_sign = direction.get(col, 1)
        z = z * dir_sign
        zscores[col] = z
    # weighted score (normalize weights among available factors)
    avail_weights = {k: v for k, v in weights.items() if k in zscores.columns}
    total_w = sum(abs(v) for v in avail_weights.values()) or 1.0
    score = pd.Series(0.0, index=zscores.index)
    for col, w in avail_weights.items():
        score += zscores[col] * (w / total_w)
    df["score"] = score
    # sort by score desc and pick top_n
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df, zscores

# ---------------- MAIN ----------------
def main(args):
    big = read_all_csvs(args.data_dir)
    factor_df = build_factor_table(big, CFG)
    scored_df, z = standardize_and_score(factor_df, CFG)
    os.makedirs(args.out_dir, exist_ok=True)
    out_all = os.path.join(args.out_dir, "factors_with_scores.csv")
    scored_df.to_csv(out_all, index=False)
    print(f"Wrote full factor table with scores to {out_all}")
    topn = scored_df.head(CFG["top_n"])
    out_top = os.path.join(args.out_dir, f"top_{CFG['top_n']}.csv")
    topn.to_csv(out_top, index=False)
    print(f"Wrote top {CFG['top_n']} to {out_top}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick top assets by multi-factor (keeping ticket; ignoring 'Ticker')")
    parser.add_argument("--data_dir", required=True, help="Input directory with CSV files")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()
    main(args)