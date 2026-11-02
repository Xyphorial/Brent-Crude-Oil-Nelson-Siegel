# Input data (not redistributed)

This research was originally run using the author's supplied `brent_futures.csv` containing vendor-derived Brent contract quotes. The original CSV is intentionally **not published** due to potential market-data licensing and redistribution restrictions.

To rerun, place an appropriately licensed `brent_futures.csv` in this directory. The expected original source columns are:

- `query_date`
- `ID`
- `FUT_MONTH_YR`
- `fut_last_trade_dt()`
- `px_last(dates=#dt)`
- `px_volume(dates=#dt)`
- `px_bid(dates=#dt)`
- `px_ask(dates=#dt)`

`brent_ns/data.py` documents the schema and performs data validation. Data/expiry provenance and limitations are recorded in `output/real/data_provenance.json` and the report. No quote filling is performed.
