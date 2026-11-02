# 从你的真实 CSV 开始学

这一版不再模拟行情。先打开 learn_step_1.py，逐段对照下面的解释。

## 1 定位文件

```python
ROOT = Path(__file__).resolve().parent
file_path = ROOT / 'data' / 'brent_futures.csv'
```

__file__ 表示当前脚本的位置，parent 是它所在的文件夹。后面的 / 用来拼接路径。整个 file_path 指向项目里的真实 CSV。

## 2 读取成表格

```python
df = pd.read_csv(file_path)
```

pd 是 pandas 的简称。read_csv 读取 CSV，得到 DataFrame；df 是保存这张表的普通变量名。你的原始表包含 query_date、ID、FUT_MONTH_YR 和价格等列。

## 3 改列名

```python
df = df.rename(columns={'query_date': 'date', 'ID': 'ticker'})
```

字典左边是旧列名，右边是新列名。改名字不改变价格。完整代码还把很长的价格字段改成 price，把最后交易日改成 last_trade_date。

## 4 日期转换

```python
df['date'] = pd.to_datetime(df['date'])
```

CSV 中日期原本是文本。to_datetime 转成可排序、可比较、可计算相差天数的日期。右边转换，左边写回原列。

## 5 删除没有价格的行

```python
df = df.dropna(subset=['price'])
```

NaN 表示缺失。dropna 删除指定列为空的行。你的文件第一天 2026-01-01 全部无价；代码跳过，不把下一天价格复制过去。

```python
df = df.loc[df['date'] <= df['last_trade_date']]
```

比较每一行的观察日期与该合约最后交易日；只保留日期不超过最后交易日的行。loc 用条件筛选行。

## 6 找到第一天

```python
first_date = df['date'].min()
first_curve = df.loc[df['date'] == first_date].copy()
```

min 找到最早有有效价格的日期，结果是 2026-01-02。第二行取出那一天的所有合约。copy 创建一张独立表，便于后续操作。

## 7 选最近 12 个

```python
first_curve = first_curve.sort_values('last_trade_date').head(12)
```

对这份连续月度 Brent 数据，按最后交易日排序可以展示近月顺序。完整模型则明确按经济期限排序。head(12) 保留前 12 行。

第一行是真实的 COH26 Comdty（三月合约），PX_LAST=60.75；实际最后交易日为文件所列 2026-01-30。这个日期来自你的文件，还不是我们独立核验后的官方日期。

## 8 画原始价格曲线

```python
ax.plot(first_curve['contract_month'], first_curve['price'], marker='o')
```

横轴用合约月份标签，纵轴用 PX_LAST。这里只展示原始价格，不拟合。

完整模型会把月份标签转换为该月第一天，再计算：经济期限（月）=(合约月锚点-观察日期).days/(365.25/12)。之后才调用 curve、objective、fit。

## 推荐学习顺序

先理解 learn_step_1.py，再学 data.py 中的 read_quotes 和 make_panel，然后进入 model.py 的曲线公式与拟合。最后学 research.py 的每天循环和按时间验证。先看懂真实表格，再学优化器。
