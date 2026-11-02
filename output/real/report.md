# Brent Nelson–Siegel Curve Research

用户上传的真实报价 / UPLOADED MARKET QUOTES — PX_LAST, vendor expiry metadata

## Scope and conclusion

This reproducible run fits daily cross-sectional futures-price curves. The validation-selected model is **ridge**. Its untouched chronological test-period price RMSE is **0.2129 USD/barrel**, versus warm-start-only **0.1937**. Mean scaled parameter movement changes by **90.3% reduction**, while RMSE changes by **+9.9%**. A zero-strength penalty or selection of warm start is an admissible result; stability improvements are not assumed.

These figures describe the uploaded quotes only. Prices and last-trade dates are read directly from the supplied CSV. No synthetic prices or expiry dates are generated. Vendor provenance and historical expiry dates have not been independently authenticated. Test performance does not determine the selected model. Repeatedly editing the model after inspecting test results would contaminate that period.

## Design

- Sample: 2026-01-01 to 2026-06-01; 106 dates with observed prices in the uploaded file, 106 complete curve slices.
- Contract design: fixed_k; target K=12; expiry buffer=0 observed quote sessions (primary buffer=0; not an official exchange calendar).
- Economic tenor: actual days to contract-month day 1, divided by 365.25/12. Tradability uses supplied fut_last_trade_dt() metadata, not expiry inferred from month codes.
- Target: log prices. Observation weights: equal. No returns are fitted.
- Model: f(tau)=beta0+beta1*A+beta2*(A-exp(-tau/lambda)); A=(1-exp(-tau/lambda))/(tau/lambda).
- phi=(beta0,beta1,beta2,log(lambda)); lambda is in months, bounded to 0.25–60 months. Target-specific beta bounds are in model.py.
- Loss: weighted MSE + strength * [(1-alpha) sum(delta²) + alpha sum(sqrt(delta²+epsilon²)-epsilon)], where delta=(phi_today-phi_previous)/training_scales.
- smooth L1 increment approximation; epsilon=0.01. This implementation does not produce guaranteed exact zero increments. It must not be presented as an exact sparse Lasso solution.
- Training-only robust scales: [0.03428478257364718, 0.048396953779529, 0.08422211648554072, 0.6637866829988114]. Initial seeds and complete config are saved.
- Burn-in: 20 slices; penalty ramps linearly. Stability criterion: 5 consecutive successful fits with scaled step <= 0.25; diagnostic only. This criterion is descriptive and not evidence of accuracy.
- Four static/warm benchmarks plus Ridge, smooth L1-change, and elastic-net (alpha=0.5).
- Tuning: validation RMSE <= warm baseline * 1.10, movement-tracking error <= warm baseline * 1.20, no optimiser failures; select least mean scaled movement among eligible candidates.
- Failed optimiser estimates are flagged and are not propagated as the next warm start. Metrics include all attempted fits; inspect failures.csv before interpreting comparisons.

## Validation interpretation

Daily test RMSE is a **contemporaneous calibration** metric using that day's quotes, not tomorrow-price prediction. The separately recorded forecast_rmse evaluates yesterday's frozen parameter vector on today's tenors before updating. The cross-sectional holdout removes every fourth tenor from that day's optimisation; test holdout price RMSE is **0.1976 USD/barrel** over 81 withheld observations, with 0 failed-fit observations flagged.

Response error compares fitted daily curve changes with changes in interpolated market curves on the fixed observed 3–12 month grid. Dates not covering that grid are missing, never extrapolated. It is a tracking diagnostic; interpolation itself contains price noise. Roster changes are logged separately. There is no random train/test split.

## Robustness and limitations

robustness.csv compares target, tenor universe, anchor, buffer, weights, optimiser and smooth-L1 epsilon. Each variant estimates scales on its own training set and tunes using its own validation period; test results remain evaluation only. The unit check in tests confirms months-to-years invariance when decay is converted consistently. Seed sensitivity is separately recorded.

Skipped optional comparisons: []. Missing volume does not block equal-weight real-data calibration.

Parameter non-identifiability can yield different parameters with similar prices. Bounds, projected-gradient norms, optimiser messages, seeds and data/penalty terms are retained. Nelson–Siegel here is a descriptive commodity-curve basis, not a structural storage or no-arbitrage model. Strong penalties can mask shifts; lower parameter movement alone is not proof of model quality. A missing day is skipped, and the next warm start is the last successful eligible observation, not necessarily the previous calendar day.

## Data provenance

{
  "data_kind": "real",
  "source": "User-uploaded brent_futures.csv; Bloomberg-style field names",
  "price_field": "px_last(dates=#dt)",
  "price_definition": "Vendor last/close field; not asserted to be official settlement",
  "retrieval_timestamp": "Not provided in uploaded CSV",
  "received_date": "2026-10-05",
  "overrides": "Not provided",
  "calendar_verified": false,
  "expiry_source": "fut_last_trade_dt() from uploaded CSV; independent historical ICE cross-check outstanding",
  "raw_sha256": "7125de871be8225b2b15806946f9a7c53dd7e477600419864895038be5af9c0f",
  "rows_in_sample": 14263,
  "contracts_in_sample": 157,
  "price_observations": 9946,
  "dates_with_prices": 106,
  "calendar_days_without_prices": 46
}

The CSV supplies explicit last-trade dates; this run uses those vendor dates directly. Independent historical ICE verification remains outstanding. The file contains 152 calendar dates, with prices on 106 dates. All-empty days are logged as no_prices_in_uploaded_file, not automatically called holidays. No official calendar is fabricated. There are 157 distinct contracts and an expansion of far-dated listings in May. The primary nearest-12 roster uses relative tenors. PX_LAST is used consistently and is not asserted to be official settlement. Some bid/ask spreads are extreme, so mid is not substituted for PX_LAST. Missing volume is retained; optional liquidity checks may be skipped.

## Sources checked 2026-10-05

- ICE product specifications: https://www.ice.com/products/219/Brent-Crude-Futures
- ICE expiry details: https://www.ice.com/products/219/Brent-Crude-Futures/expiry
- SciPy bounded minimisation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- User-provided project specification: Nelson Siegel for Brent Futures Curve Fitting.docx

## Reproducibility

See data_provenance.json, observation_calendar.csv, run_config.json, run_metadata.json, environment.json and input_hashes.json. The input CSV is never forward-filled. There is no demo mode in this project. Generated outputs include contract audit, long panel, coverage, roster changes, fits, residuals, train/validation/test scores, tuning, withheld predictions, robustness, seed sensitivity and figures.
