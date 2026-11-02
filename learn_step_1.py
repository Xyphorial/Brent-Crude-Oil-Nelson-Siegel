"""学习入口：直接读取你的真实 CSV，显示第一条价格曲线。"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. 找到与这个脚本相邻的 data 文件夹里的 CSV。
ROOT = Path(__file__).resolve().parent
file_path = ROOT / 'data' / 'brent_futures.csv'

# 2. 读取 CSV。每一行是一个日期、一个合约的观察。
df = pd.read_csv(file_path)

# 3. 把很长的供应商列名换成容易理解的名称。
df = df.rename(columns={
    'query_date': 'date',
    'ID': 'ticker',
    'FUT_MONTH_YR': 'contract_month',
    'fut_last_trade_dt()': 'last_trade_date',
    'px_last(dates=#dt)': 'price'
})

# 4. 日期文本转换成 pandas 日期，价格列原本就是数字。
df['date'] = pd.to_datetime(df['date'])
df['last_trade_date'] = pd.to_datetime(df['last_trade_date'])

# 5. 删除没有价格的观察，只保留最后交易日之前/当天的数据。
df = df.dropna(subset=['price'])
df = df.loc[df['date'] <= df['last_trade_date']]

# 6. 找到文件中第一个有实际价格的日期。
first_date = df['date'].min()
first_curve = df.loc[df['date'] == first_date].copy()

# 7. 这份 CSV 的标签按月份列出；按最后交易日排序选最近 12 个。
first_curve = first_curve.sort_values('last_trade_date').head(12)
print('第一个有价格的日期：', first_date.date())
print(first_curve[['date', 'ticker', 'contract_month', 'last_trade_date', 'price']].to_string(index=False))

# 8. 先按合约月份画出价格，不涉及优化和 Jacobian。
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(first_curve['contract_month'], first_curve['price'], marker='o')
ax.set_title(f'Uploaded Brent quotes: {first_date.date()}')
ax.set_xlabel('Contract month')
ax.set_ylabel('PX_LAST (USD/barrel)')
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
plt.show()
