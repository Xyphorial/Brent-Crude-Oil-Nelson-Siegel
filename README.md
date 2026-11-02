# Brent Crude Oil Futures Curve Fitting | Nelson–Siegel

**Python • SciPy optimisation • Nelson–Siegel • Time-series validation**

Research study of fitting daily Brent futures term structures while balancing cross-sectional fit accuracy and day-to-day parameter stability. The archived outputs were computed using the author's real quoted futures data, **not synthetic market results**. Raw vendor quotes are **not redistributed** here.

## Research questions and methods

- Convert contract expiries into remaining tenors and construct a daily nearest-12-contract panel.
- Fit a four-parameter Nelson–Siegel curve to **log futures prices** using bounded optimisation and analytic Jacobians.
- Compare fixed/random/multi-start initialisation, previous-day warm starts, Ridge penalties on parameter increments, smooth-L1 increments and elastic-net variants.
- Tune penalty strength using a chronological training/validation split; assess an untouched test period and withheld-tenor predictions.
- Evaluate fitting errors, forecast-style yesterday-parameter errors, optimisation failures, stability, parameter paths and robustness checks.

## Archived findings (real-data run, October 2026)

- **106 daily curves**, split into **53 training, 26 validation and 27 test days**.
- The validation selection rule favoured **Ridge**, prioritising parameter stability subject to validation fit constraints.
- On the test period, Ridge's curve-fit RMSE was **0.2129 USD/barrel** compared with **0.1937** for unpenalised warm start and **0.1023** for smooth-L1. This highlights the explicit **stability–fit trade-off**; the selection is not a claim that Ridge wins every test metric.
- Withheld-tenor prediction RMSE was **0.1976 USD/barrel** in the archived run.
- These are **curve calibration diagnostics**, not trading returns or evidence of profitability.

## Explore results without market data

Open [archived HTML report](output/real/report.html) or [Markdown report](output/real/report.md). Charts are in [output/real/figures](output/real/figures/), and aggregate comparisons are in [scores.csv](output/real/scores.csv), [robustness.csv](output/real/robustness.csv) and [seed_sensitivity.csv](output/real/seed_sensitivity.csv).

## Reproduce (requires your own licensed data)

1. Install Python 3.10+ and packages from `requirements.txt`.
2. Supply `data/brent_futures.csv` in the schema described at [data/README.md](data/README.md).
3. Run the project entry point `main.py` with a fresh `--output` folder (default: `output/my_run`).
4. Run tests with `python -m unittest discover -s tests -v`. Math tests work without quotes; vendor-data integration tests are skipped if the CSV is absent.

Core implementation: `brent_ns/data.py`, `brent_ns/model.py`, `brent_ns/research.py`, `brent_ns/report.py`. There is also a beginner-friendly `learn_step_1.py` and a [Chinese learning guide](docs/LEARNING_GUIDE_CN.md).

## Limitations

Observed PX_LAST is not asserted to be official exchange settlement. Historical expiry dates supplied by the vendor were not independently verified. Missing-price days are not automatically treated as exchange holidays. Parameter regularisation can slow adaptation to regime changes; lower in-sample error does not imply better forecasting or trading outcomes. The attached report and aggregate diagnostics reflect the original data run, and cannot be identically reproduced without the licensed source quotes.

**Data provenance:** original vendor quotes (not redistributed). [ICE Brent specifications](https://www.ice.com/products/219/Brent-Crude-Futures).
