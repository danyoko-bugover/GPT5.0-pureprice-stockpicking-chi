#!/usr/bin/env python3
# pick_top20.py
"""
擴充版多因子選股腳本（修正版）
- 讀取資料夾中所有 csv（支援單檔多票）
- 計算大量因子（動量 / 趨勢 / 波動 / 量能 / 技術指標 / 風險調整收益 / 穩定性等）
- z-score 標準化 + 權重合成 -> 選出 top N

執行方式：
python pick_top20.py --data_dir ./input --out_dir ./output
"""

import os, glob, argparse
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import zscore, linregress

# -------------- CONFIG --------------
CFG = {
    "momentum_windows": [252, 90, 60, 20, 5],   # days
    "ma_windows": [20, 60, 200],
    "vol_windows": [90, 30],
    "rsi_window": 14,
    "atr_window": 14,
    "min_history_days": 120,
    "top_n": 20,
    # 預設權重（可調）
    "weights": {
        # 動量類
        "mom_252": 0.12,
        "mom_90": 0.10,
        "mom_60": 0.06,
        "mom_20": 0.06,
        "mom_5": 0.03,
        # 趨勢 / 均線
        "ma_ratio_s_l": 0.06,
        "price_vs_ma_long": 0.04,
        # 風險 / 波動（低越好）
        "ann_vol_90": 0.08,
        "vol_30": 0.03,
        "max_drawdown": 0.08,
        "downside_dev": 0.03,
        # 流動性 / 量能
        "avg_vol_60": 0.03,
        "vol_spike": 0.02,
        # 動量動態 / 確認
        "mom_accel": 0.04,
        "volatility_change": 0.02,
        # 技術指標
        "rsi_14": 0.02,
        "atr_14": 0.02,
        # 風險調整收益 / 穩定性
        "sharpe_like": 0.10,
        "calmar": 0.05,
        "trend_r2": 0.05,
        # 品質 / 其他
        "consec_up_10": 0.02,
        "skewness": 0.0,
        "kurtosis": 0.0
    }
}

# -------------- UTIL --------------
def read_all_csvs(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError("No CSV files found in %s" % data_dir)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=["Date"], dtype={"ticket": str})
        except Exception as e:
            print("Read fail", f, e)
            continue
        required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "ticket"}
        if not required.issubset(set(df.columns)):
            print("Skip %s: missing columns" % f)
            continue
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[["Date", "ticket", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    all_df.sort_values(["ticket", "Date"], inplace=True)
    return all_df

def pct_change_safe(s):
    return s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

def rolling_downside_deviation(returns, window):
    # downside deviation: std of negative returns in window
    def dd(x):
        neg = x[x < 0]
        return float(np.sqrt((neg ** 2).mean())) if len(neg) > 0 else 0.0
    return returns.rolling(window, min_periods=1).apply(dd, raw=False)

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window, min_periods=1).mean()
    ma_down = down.rolling(window, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df, window=14):
    high = df["High"]
    low = df["Low"]
    close = df["Adj Close"].fillna(df["Close"])
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=1).mean()
    return atr

def stability_r2(log_price, window):
    # linear regression of log price over window, return R^2 for last window
    if len(log_price) < 2:
        return 0.0
    if len(log_price) < window:
        y = log_price
    else:
        y = log_price.iloc[-window:]
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return float(r_value ** 2)

# -------------- FACTOR COMPUTE --------------
def compute_factors_for_ticket(df_ticket, cfg):
    df = df_ticket.copy().reset_index(drop=True)
    if len(df) < cfg["min_history_days"]:
        return None
    price = df["Adj Close"].fillna(df["Close"]).astype(float)
    vol = df["Volume"].astype(float).fillna(0.0)
    ret = pct_change_safe(price)
    logp = np.log(price.replace(0, np.nan)).dropna()

    factors = {}
    factors["ticket"] = df["ticket"].iloc[0]
    factors["last_date"] = df["Date"].iloc[-1]
    factors["price"] = price.iloc[-1]
    # momentum windows
    for w in cfg["momentum_windows"]:
        if len(price) >= w and w > 0:
            mom = price.iloc[-1] / price.iloc[-w] - 1
        else:
            mom = price.iloc[-1] / price.iloc[0] - 1
        factors[f"mom_{w}"] = float(mom)
    # ma and ratios
    ma = {}
    for w in cfg["ma_windows"]:
        ma[w] = price.rolling(window=w, min_periods=1).mean().iloc[-1]
        factors[f"ma_{w}"] = float(ma[w])
    factors["ma_ratio_s_l"] = float(ma[cfg["ma_windows"][0]] / (ma[cfg["ma_windows"][-1]] + 1e-9))
    factors["price_vs_ma_long"] = float(price.iloc[-1] / (ma[cfg["ma_windows"][-1]] + 1e-9))
    # vol / volatility
    for w in cfg["vol_windows"]:
        sigma = ret.rolling(window=w, min_periods=2).std().iloc[-1]
        factors[f"ann_vol_{w}"] = float(sigma * np.sqrt(252)) if not pd.isna(sigma) else np.nan
        factors[f"vol_{w}"] = float(sigma) if not pd.isna(sigma) else np.nan
    # max drawdown
    cummax = price.cummax()
    dd = (price / cummax - 1.0)
    factors["max_drawdown"] = float(dd.min())  # negative
    # downside deviation (90)
    dd5 = rolling_downside_deviation(ret, 90).iloc[-1]
    factors["downside_dev"] = float(dd5)
    # volume
    factors["avg_vol_20"] = float(vol.rolling(20, min_periods=1).mean().iloc[-1])
    factors["avg_vol_60"] = float(vol.rolling(60, min_periods=1).mean().iloc[-1])
    factors["vol_spike"] = float(vol.iloc[-1] / (factors["avg_vol_20"] + 1e-9))
    # momentum acceleration
    factors["mom_accel"] = float(factors.get("mom_20", 0.0) - factors.get("mom_60", 0.0))
    # volatility change
    v30 = factors.get("ann_vol_30", np.nan)
    v90 = factors.get("ann_vol_90", np.nan)
    factors["volatility_change"] = float((v30 / (v90 + 1e-9)) if (not pd.isna(v30) and not pd.isna(v90)) else 0.0)
    # rsi and atr
    rsi = compute_rsi(price, window=cfg["rsi_window"]).iloc[-1]
    factors["rsi_14"] = float(rsi)
    atr = compute_atr(df, window=cfg["atr_window"]).iloc[-1]
    factors["atr_14"] = float(atr)
    # ann return estimate (from 90d)
    look = min(len(price), 90)
    total_ret_look = (price.iloc[-1] / price.iloc[-look] - 1) if look > 1 else 0.0
    ann_return = (1 + total_ret_look) ** (252 / look) - 1 if total_ret_look > -0.999 else -0.999
    factors["ann_return"] = float(ann_return)
    # sharpe-like
    ann_vol = factors.get("ann_vol_90", np.nan)
    if not pd.isna(ann_vol) and ann_vol > 0:
        factors["sharpe_like"] = float(ann_return / ann_vol)
    else:
        factors["sharpe_like"] = 0.0
    # calmar ratio
    md = factors["max_drawdown"]
    factors["calmar"] = float(ann_return / (-md) if md < 0 and ann_return> -0.999 else 0.0)
    # trend stability R^2 over 180 days
    try:
        factors["trend_r2"] = float(stability_r2(logp, 180))
    except Exception:
        factors["trend_r2"] = 0.0
    # consecutive up days in past 10
    last10 = price.pct_change().fillna(0).tail(10)
    consec = 0
    for x in last10.iloc[::-1]:
        if x > 0:
            consec += 1
        else:
            break
    factors["consec_up_10"] = int(consec)
    # skewness / kurtosis
    rt90 = ret.tail(90)
    factors["skewness"] = float(rt90.skew()) if len(rt90)>2 else 0.0
    factors["kurtosis"] = float(rt90.kurtosis()) if len(rt90)>3 else 0.0
    # avg volume for liquidity filter
    factors["avg_volume"] = float(vol.tail(60).mean()) if len(vol)>=1 else float(vol.mean())
    factors["history_days"] = len(price)
    return factors

# -------------- AGGREGATE & SCORE --------------
def build_factor_table(all_df, cfg):
    rows = []
    for ticker, g in all_df.groupby("ticket"):
        res = compute_factors_for_ticket(g, cfg)
        if res:
            rows.append(res)
    if not rows:
        raise RuntimeError("No tickers with sufficient history")
    df = pd.DataFrame(rows).set_index("ticket")
    return df

def winsorize_series(s, p=0.01):
    low = s.quantile(p)
    high = s.quantile(1-p)
    return s.clip(lower=low, upper=high)

def standardize_and_score(factor_df, cfg):
    df = factor_df.copy()
    all_weights = cfg["weights"]

    # define directionality: 1 means larger is better, -1 means smaller is better
    direction = {}
    for w in cfg["momentum_windows"]:
        direction[f"mom_{w}"] = 1
    direction["ma_ratio_s_l"] = 1
    direction["price_vs_ma_long"] = 1
    direction["sharpe_like"] = 1
    direction["calmar"] = 1
    direction["trend_r2"] = 1
    direction["consec_up_10"] = 1
    direction["vol_spike"] = 1
    direction["avg_vol_60"] = 1
    direction["mom_accel"] = 1
    direction["volatility_change"] = 1

    # risk / volatility / drawdown: lower better
    direction["ann_vol_90"] = -1
    direction["vol_30"] = -1
    direction["max_drawdown"] = -1
    direction["downside_dev"] = -1
    direction["atr_14"] = -1

    # RSI: penalize distance from 50 (we'll treat transformed series with smaller-is-better)
    # build list of factors to score from the weights dict
    factor_keys = list(all_weights.keys())

    # prepare score_mat with renamed columns score__{factor}
    score_mat = pd.DataFrame(index=df.index)
    for f in factor_keys:
        colname = f
        if colname in df.columns:
            raw = df[colname].fillna(df[colname].median())
        else:
            # if missing factor, fill with zeros (neutral)
            raw = pd.Series(0.0, index=df.index)
        # special handling: RSI -> distance from 50 (smaller better)
        if f == "rsi_14":
            raw = -np.abs(raw - 50.0)  # larger is better here (closer to 50 gives value nearer to 0, but we invert)
        # For max_drawdown which is negative, keep raw (direction handles sign)
        # Winsorize
        raw_w = winsorize_series(raw, p=0.01)
        # standardize (z-score). If constant, set zeros.
        if raw_w.std() == 0 or np.isnan(raw_w.std()):
            z = pd.Series(0.0, index=raw_w.index)
        else:
            z = (raw_w - raw_w.mean()) / (raw_w.std() + 1e-9)
        # apply direction (flip sign if lower-is-better)
        dir_ct = direction.get(f, 1)
        z = z * dir_ct
        score_col_name = f"score__{f}"
        score_mat[score_col_name] = z

    if score_mat.shape[1] == 0:
        raise RuntimeError("No scoring columns were generated. Check CFG weights keys and factor names in data.")

    # Normalize weights among available scored factors
    used_factors = [f for f in factor_keys if f in all_weights]
    # but we should normalize according to those we actually computed in score_mat
    score_cols = score_mat.columns.tolist()
    # map back to weight keys
    norm_weights = {}
    total = 0.0
    for sc in score_cols:
        orig = sc.replace("score__", "")
        w = float(all_weights.get(orig, 0.0))
        norm_weights[sc] = w
        total += abs(w)
    if total == 0:
        # avoid divide by zero
        for k in norm_weights:
            norm_weights[k] = 0.0
    else:
        for k in norm_weights:
            norm_weights[k] = norm_weights[k] / total

    # compute composite
    composite = np.zeros(len(score_mat.index))
    for sc in score_cols:
        composite += score_mat[sc].values * norm_weights.get(sc, 0.0)
    score_mat["composite_score"] = composite

    # join back — score_mat columns are prefixed so no overlap
    out = df.join(score_mat, how="left")
    out = out.sort_values("composite_score", ascending=False)
    return out

# -------------- MAIN --------------
def main(data_dir, out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    print("Reading CSVs from", data_dir)
    all_df = read_all_csvs(data_dir)
    print("Loaded rows:", len(all_df), "tickers:", all_df["ticket"].nunique())
    print("Computing factors ...")
    factor_df = build_factor_table(all_df, cfg)
    print("Factor table shape:", factor_df.shape)
    print("Scoring ...")
    scored = standardize_and_score(factor_df, cfg)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scored.to_csv(os.path.join(out_dir, f"scored_all_more_factors_{ts}.csv"))
    topn = scored.head(cfg["top_n"])
    topn.to_csv(os.path.join(out_dir, f"top{cfg['top_n']}_more_factors_{ts}.csv"))
    print("Top", cfg["top_n"])
    # show composite and the score columns (a few)
    score_cols = [c for c in scored.columns if c.startswith("score__")]
    print(topn[["price", "composite_score"] + score_cols].head(cfg["top_n"]))
    return scored, topn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="./output")
    args = ap.parse_args()
    scored, top = main(args.data_dir, args.out_dir, CFG)