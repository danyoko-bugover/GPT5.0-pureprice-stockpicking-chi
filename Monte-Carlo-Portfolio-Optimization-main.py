import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import matplotlib.font_manager as fm

# ── 中文字型設定 ───────────────────────────────────────
# 優先嘗試系統字型，macOS 上 PingFang TC / Heiti TC 常有 bug，可先清 Matplotlib 快取：
# 終端機執行： rm -rf ~/.cache/matplotlib   （然後重跑）
mpl.rcParams['font.family'] = ['Heiti TC', 'PingFang TC', 'Hiragino Sans', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False  # 避免負號變方塊

# ── A股設定 ────────────────────────────────────────────
#在輸入 A 股 ticker 時，請使用 Yahoo Finance 的格式：
#- 上海證券交易所：600000.SS、681872.SS 等
#- 深圳證券交易所：000023.SZ、302028.SZ 等
# 這樣才能正確下載資料。

# -----------------------------------------------------------------------------------
# Author: Hunter Gould (原作者) + 修改版
# Description: Monte Carlo Portfolio Optimization - Efficient Frontier 
# -----------------------------------------------------------------------------------

# ── 參數設定 ───────────────────────────────────────────────────────────────
ASSETS = ["601872.SS","002028.SZ","600875.SS","000338.SZ","600522.SS","600026.SS","600176.SS","600219.SS","000792.SZ","603993.SS"]  # 資產列表
START_DATE = '2025-02-28'
END_DATE = '2026-02-28'
MARKET_REPRESENTATION = '510300.SS'  # 市場基準
NUM_PORTFOLIOS = 100_000
RISK_FREE_RATE = 0.01310 # 無風險利率


# ── 資料下載 ────────────────────────────────────────────────────────────────
print("下載資產資料...")
data = yf.download(ASSETS, start=START_DATE, end=END_DATE, auto_adjust=False)['Close']  # 用 Close 避免 Adj Close 問題

print("下載市場基準...")
market_data = yf.download(MARKET_REPRESENTATION, start=START_DATE, end=END_DATE)['Close']

# 強制確保是單一 Series，並取出數值
if isinstance(market_data, pd.DataFrame):
    market_data = market_data.iloc[:, 0]   # 取第一欄（通常只有一欄）

# 檢查資料是否有缺漏
if data.isnull().values.any():
    print("警告：部分資產有 NaN 值，正在移除...")
    data = data.dropna()

daily_returns = data.pct_change().dropna()
if daily_returns.empty:
    raise ValueError("日報酬資料為空，請檢查 ticker 或日期範圍！")


# ── 共變異數矩陣（年化） ───────────────────────────────────────────────────
cov_matrix = daily_returns.cov() * 252


# ── 市場基準表現 ────────────────────────────────────────────────────────────
market_daily_returns = market_data.pct_change().dropna()
market_return = float(market_daily_returns.mean() * 252)
market_volatility = float(market_daily_returns.std() * np.sqrt(252))
market_sharpe_ratio = float((market_return - RISK_FREE_RATE) / market_volatility) if market_volatility != 0 else 0.0


# ── 蒙地卡洛模擬 ────────────────────────────────────────────────────────────
print(f"開始模擬 {NUM_PORTFOLIOS:,} 個投資組合...")
results = np.zeros((4, NUM_PORTFOLIOS))
weights_record = np.zeros((len(ASSETS), NUM_PORTFOLIOS))

np.random.seed(42)  # 固定隨機種子，可重現結果

for i in range(NUM_PORTFOLIOS):
    weights = np.random.random(len(ASSETS))
    weights /= np.sum(weights)
    weights_record[:, i] = weights

    port_return = np.sum(weights * daily_returns.mean()) * 252
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (port_return - RISK_FREE_RATE) / port_std if port_std != 0 else 0

    results[0, i] = port_return
    results[1, i] = port_std
    results[2, i] = sharpe
    results[3, i] = i


# ── 找出最佳組合 ───────────────────────────────────────────────────────────
simulated_portfolios = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio', 'Simulation'])

optimal_idx = simulated_portfolios['Sharpe Ratio'].idxmax()
optimal_portfolio = simulated_portfolios.loc[optimal_idx]
optimal_weights = weights_record[:, optimal_idx]


# ── 繪圖 ────────────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 8))

plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

plt.scatter(
    simulated_portfolios['Volatility'],
    simulated_portfolios['Return'],
    c=simulated_portfolios['Sharpe Ratio'],
    cmap='YlGnBu',
    alpha=0.6,
    s=10
)
plt.colorbar(label='Sharpe Ratio')

plt.scatter(market_volatility, market_return, color='red', marker='o', s=120, label=f'Market ({MARKET_REPRESENTATION})')
plt.scatter(optimal_portfolio.iloc[1], optimal_portfolio.iloc[0], color='lime', marker='*', s=200, label='最佳組合')

plt.xlabel('年化波動率 (Volatility)')
plt.ylabel('年化報酬 (Return)')
plt.title('蒙地卡洛模擬 - 有效邊界 (Efficient Frontier)')
plt.legend()
plt.grid(True, alpha=0.3)


# ── 文字框 ──────────────────────────────────────────────────────────────────
weight_text = "最佳權重 (Optimal Weights):\n" + \
              "\n".join([f"{asset:8} : {w*100:6.2f}%" for asset, w in zip(ASSETS, optimal_weights)])

plt.gcf().text(0.02, 0.98, weight_text, fontsize=10,
               va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9))

opt_text = f"最佳組合\n報酬: {optimal_portfolio['Return']*100:.2f}%\n波動: {optimal_portfolio['Volatility']*100:.2f}%\nSharpe: {optimal_portfolio['Sharpe Ratio']:.2f}"
plt.gcf().text(0.02, 0.98 - 0.20, opt_text, fontsize=10,
               va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="green", alpha=0.9))

market_text = f"市場基準 ({MARKET_REPRESENTATION})\n報酬: {market_return*100:.2f}%\n波動: {market_volatility*100:.2f}%\nSharpe: {market_sharpe_ratio:.2f}"
plt.gcf().text(0.02, 0.98 - 0.36, market_text, fontsize=10,
               va='top', ha='left',
               bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="red", alpha=0.9))

plt.tight_layout()
plt.show()


# ── 終端機輸出 ──────────────────────────────────────────────────────────────
print("\n模擬完成！")
print("最佳 Sharpe Ratio 組合權重：")
for asset, w in zip(ASSETS, optimal_weights):
    print(f"{asset:10} : {w*100:8.4f}%")

print(f"\n預期年化報酬: {optimal_portfolio['Return']*100:.2f}%")
print(f"年化波動率: {optimal_portfolio['Volatility']*100:.2f}%")
print(f"Sharpe Ratio: {optimal_portfolio['Sharpe Ratio']:.3f}")