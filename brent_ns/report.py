"""Static charts and self-contained, offline HTML research report."""
from pathlib import Path
import base64
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .model import curve

COLORS={'fixed':'#9a9a9a','random':'#c49c94','multistart':'#8c564b','warm':'#22577a',
        'ridge':'#179c86','lasso_smooth':'#df8b24','elastic':'#8b5fbf'}


def charts(out,results,residuals,tuning,coverage,preferred,target):
    out=Path(out); folder=out/'figures'; folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':130,'savefig.dpi':150})
    names=[]
    def save(fig,name):
        fig.tight_layout(); fig.savefig(folder/name,bbox_inches='tight'); plt.close(fig); names.append(name)
    focus=['warm','ridge','lasso_smooth','elastic']
    fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ax,col,title in zip(axs.flat,['beta0','beta1','beta2','decay_months'],['Level beta0','Slope beta1','Curvature beta2','Decay (months)']):
        for method in focus:
            group=results.loc[results.method.eq(method)]
            ax.plot(group.date,group[col],label=method,color=COLORS[method],lw=1.2)
        ax.set_title(title); ax.grid(alpha=.15); ax.tick_params(axis='x',rotation=25)
    axs.flat[0].legend(fontsize=8,ncol=2)
    save(fig,'parameters.png')
    fig,axs=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for method in focus:
        group=results.loc[results.method.eq(method)]
        axs[0].plot(group.date,group.rmse,label=method,color=COLORS[method])
        axs[1].plot(group.date,group.scaled_step,label=method,color=COLORS[method])
    axs[0].set_ylabel('Daily price RMSE ($/barrel)'); axs[0].legend(ncol=4)
    axs[1].set_yscale('symlog',linthresh=.1)
    axs[1].set_ylabel('Scaled movement (symlog)'); axs[1].tick_params(axis='x',rotation=25)
    # Roster markers visible alongside numerical diagnostics.
    for date in coverage.loc[coverage.roster_change,'date']:
        for ax in axs:
            ax.axvline(date,color='grey',alpha=.25,ls=':')
    save(fig,'fit_and_stability.png')
    chosen=residuals.loc[residuals.method.eq(preferred)]
    dates=sorted(chosen.date.unique())
    picks=np.linspace(0,len(dates)-1,4,dtype=int)
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for ax,ix in zip(axs.flat,picks):
        group=chosen.loc[chosen.date.eq(dates[ix])].sort_values('tenor')
        ax.scatter(group.tenor,group.observed,label='Observed',color='#253858',s=25)
        row=results.loc[results.method.eq(preferred)&results.date.eq(dates[ix])].iloc[0]
        dense=np.linspace(group.tenor.min(),group.tenor.max(),200)
        values=curve(dense,row[['beta0','beta1','beta2','log_lambda']].to_numpy(dtype=float))
        ax.plot(dense,np.exp(values) if target=='log' else values,label='NS fitted',color=COLORS[preferred])
        ax.set_title(str(pd.Timestamp(dates[ix]).date())); ax.set_xlabel('Economic tenor (months)')
        ax.set_ylabel('USD per barrel'); ax.legend(fontsize=8)
    save(fig,'curve_examples.png')
    # Relative ranks keep expiry-induced ticker replacement from breaking the heatmap.
    chosen=chosen.copy(); chosen['rank']=chosen.groupby('date').tenor.rank(method='first').astype(int)
    pivot=chosen.pivot(index='rank',columns='date',values='residual')
    fig,ax=plt.subplots(figsize=(12,4.5)); lim=max(float(np.nanquantile(np.abs(pivot),.98)),.01)
    img=ax.imshow(pivot.to_numpy(),aspect='auto',origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim)
    ix=np.linspace(0,len(pivot.columns)-1,6,dtype=int)
    ax.set_xticks(ix,[str(pd.Timestamp(pivot.columns[i]).date()) for i in ix],rotation=20)
    ax.set_yticks(np.arange(len(pivot)),pivot.index); ax.set_ylabel('Relative contract rank')
    ax.set_title(f'Residual heatmap — {preferred}'); fig.colorbar(img,ax=ax,label='Fitted minus observed ($/barrel)')
    save(fig,'residual_heatmap.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for method,group in tuning.groupby('method'):
        axs[0].plot(group.mean_scaled_step,group.rmse,'o-',label=method,color=COLORS[method])
        axs[1].plot(group.multiplier,group.response_rmse,'o-',label=method,color=COLORS[method])
    axs[0].set_xlabel('Validation mean scaled movement'); axs[0].set_ylabel('Validation price RMSE')
    axs[1].set_xscale('symlog',linthresh=.01); axs[1].set_xlabel('Penalty / training loss unit')
    axs[1].set_ylabel('Curve-movement tracking error ($/barrel)'); axs[0].legend(fontsize=8)
    save(fig,'validation_tradeoff.png')
    return names


def write_report(out,config,manifest,results,scores,tuning,metadata,coverage,holdout,robustness):
    out=Path(out); preferred=metadata['preferred_model']
    pictures=charts(out,results,pd.read_csv(out/'residuals.csv',parse_dates=['date']),tuning,coverage,preferred,config['target'])
    test=scores.loc[scores.split.eq('test')]
    chosen=test.loc[test.method.eq(preferred)].iloc[0]
    warm=test.loc[test.method.eq('warm')].iloc[0]
    fit_change=100*(chosen.rmse/warm.rmse-1)
    smooth_change=100*(1-chosen.mean_scaled_step/warm.mean_scaled_step)
    truth='模拟数据演示 / SYNTHETIC DATA — NOT HISTORICAL BRENT RESULTS' if manifest['data_kind']=='synthetic' else '用户上传的真实报价 / UPLOADED MARKET QUOTES — PX_LAST, vendor expiry metadata'
    dates=results.loc[results.method.eq('warm')].groupby('split').date.agg(['min','max','count']).reset_index()
    holdout_rmse=float(np.sqrt(np.mean(holdout.error**2)))
    text=f'''# Brent Nelson–Siegel Curve Research

{truth}

## Scope and conclusion

This reproducible run fits daily cross-sectional futures-price curves. The validation-selected model is **{preferred}**. Its untouched chronological test-period price RMSE is **{chosen.rmse:.4f} USD/barrel**, versus warm-start-only **{warm.rmse:.4f}**. Mean scaled parameter movement changes by **{smooth_change:.1f}% reduction**, while RMSE changes by **{fit_change:+.1f}%**. A zero-strength penalty or selection of warm start is an admissible result; stability improvements are not assumed.

These figures describe the uploaded quotes only. Prices and last-trade dates are read directly from the supplied CSV. No synthetic prices or expiry dates are generated. Vendor provenance and historical expiry dates have not been independently authenticated. Test performance does not determine the selected model. Repeatedly editing the model after inspecting test results would contaminate that period.

## Design

- Sample: {config['start']} to {config['end']}; {len(coverage)} dates with observed prices in the uploaded file, {int(coverage.complete.sum())} complete curve slices.
- Contract design: {config['design']}; target K={config['k']}; expiry buffer={config['expiry_buffer']} observed quote sessions (primary buffer=0; not an official exchange calendar).
- Economic tenor: actual days to contract-month day {config['anchor_day']}, divided by 365.25/12. Tradability uses supplied fut_last_trade_dt() metadata, not expiry inferred from month codes.
- Target: {config['target']} prices. Observation weights: {config['weighting']}. No returns are fitted.
- Model: f(tau)=beta0+beta1*A+beta2*(A-exp(-tau/lambda)); A=(1-exp(-tau/lambda))/(tau/lambda).
- phi=(beta0,beta1,beta2,log(lambda)); lambda is in months, bounded to 0.25–60 months. Target-specific beta bounds are in model.py.
- Loss: weighted MSE + strength * [(1-alpha) sum(delta²) + alpha sum(sqrt(delta²+epsilon²)-epsilon)], where delta=(phi_today-phi_previous)/training_scales.
- {metadata['lasso']}. This implementation does not produce guaranteed exact zero increments. It must not be presented as an exact sparse Lasso solution.
- Training-only robust scales: {metadata['scales']}. Initial seeds and complete config are saved.
- Burn-in: {config['burn_in']} slices; penalty ramps linearly. Stability criterion: {metadata['stability_definition']}. This criterion is descriptive and not evidence of accuracy.
- Four static/warm benchmarks plus Ridge, smooth L1-change, and elastic-net (alpha=0.5).
- Tuning: validation RMSE <= warm baseline * {1+config['fit_tolerance']:.2f}, movement-tracking error <= warm baseline * {1+config['response_tolerance']:.2f}, no optimiser failures; select least mean scaled movement among eligible candidates.
- Failed optimiser estimates are flagged and are not propagated as the next warm start. Metrics include all attempted fits; inspect failures.csv before interpreting comparisons.

## Validation interpretation

Daily test RMSE is a **contemporaneous calibration** metric using that day's quotes, not tomorrow-price prediction. The separately recorded forecast_rmse evaluates yesterday's frozen parameter vector on today's tenors before updating. The cross-sectional holdout removes every fourth tenor from that day's optimisation; test holdout price RMSE is **{holdout_rmse:.4f} USD/barrel** over {len(holdout)} withheld observations, with {int((~holdout.success).sum())} failed-fit observations flagged.

Response error compares fitted daily curve changes with changes in interpolated market curves on the fixed observed 3–12 month grid. Dates not covering that grid are missing, never extrapolated. It is a tracking diagnostic; interpolation itself contains price noise. Roster changes are logged separately. There is no random train/test split.

## Robustness and limitations

robustness.csv compares target, tenor universe, anchor, buffer, weights, optimiser and smooth-L1 epsilon. Each variant estimates scales on its own training set and tunes using its own validation period; test results remain evaluation only. The unit check in tests confirms months-to-years invariance when decay is converted consistently. Seed sensitivity is separately recorded.

Skipped optional comparisons: {json.dumps(metadata.get('skipped_robustness',[]))}. Missing volume does not block equal-weight real-data calibration.

Parameter non-identifiability can yield different parameters with similar prices. Bounds, projected-gradient norms, optimiser messages, seeds and data/penalty terms are retained. Nelson–Siegel here is a descriptive commodity-curve basis, not a structural storage or no-arbitrage model. Strong penalties can mask shifts; lower parameter movement alone is not proof of model quality. A missing day is skipped, and the next warm start is the last successful eligible observation, not necessarily the previous calendar day.

## Data provenance

{json.dumps(manifest,ensure_ascii=False,indent=2)}

The CSV supplies explicit last-trade dates; this run uses those vendor dates directly. Independent historical ICE verification remains outstanding. The file contains 152 calendar dates, with prices on 106 dates. All-empty days are logged as no_prices_in_uploaded_file, not automatically called holidays. No official calendar is fabricated. There are 157 distinct contracts and an expansion of far-dated listings in May. The primary nearest-12 roster uses relative tenors. PX_LAST is used consistently and is not asserted to be official settlement. Some bid/ask spreads are extreme, so mid is not substituted for PX_LAST. Missing volume is retained; optional liquidity checks may be skipped.

## Sources checked 2026-10-05

- ICE product specifications: https://www.ice.com/products/219/Brent-Crude-Futures
- ICE expiry details: https://www.ice.com/products/219/Brent-Crude-Futures/expiry
- SciPy bounded minimisation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
- User-provided project specification: Nelson Siegel for Brent Futures Curve Fitting.docx

## Reproducibility

See data_provenance.json, observation_calendar.csv, run_config.json, run_metadata.json, environment.json and input_hashes.json. The input CSV is never forward-filled. There is no demo mode in this project. Generated outputs include contract audit, long panel, coverage, roster changes, fits, residuals, train/validation/test scores, tuning, withheld predictions, robustness, seed sensitivity and figures.
'''
    (out/'report.md').write_text(text,encoding='utf-8')
    def table(frame):
        return frame.to_html(index=False,float_format=lambda x:f'{x:.5g}',border=0,classes='data')
    images=''.join(f'<figure><img alt="{name}" src="data:image/png;base64,{base64.b64encode((out/"figures"/name).read_bytes()).decode()}"><figcaption>{name}</figcaption></figure>' for name in pictures)
    # Plain prose is escaped. Tables are generated from data; report works offline.
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Brent NS research report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172b4d;line-height:1.6}}h1{{font-size:32px}}.flag{{background:#fff2cf;padding:16px;font-weight:700}}pre{{white-space:pre-wrap;font-family:inherit}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border:1px solid #d9e1e8;text-align:right}}th{{background:#edf2f7}}.scroll{{overflow:auto}}figure{{margin:32px 0}}figcaption{{color:#65768a;font-size:12px}}@media print{{body{{margin:0}}figure{{break-inside:avoid}}}}</style>
<h1>Brent Nelson–Siegel Curve Research</h1><p class="flag">{html.escape(truth)}</p>
<p>Validation-selected model: <b>{preferred}</b>. Test RMSE: <b>{chosen.rmse:.4f} USD/barrel</b>. Withheld-tenor RMSE: <b>{holdout_rmse:.4f}</b>.</p>
<h2>Chronological split</h2>{table(dates)}<h2>Untouched test period</h2><div class="scroll">{table(test)}</div>
<h2>Validation tuning</h2><div class="scroll">{table(tuning)}</div>{images}
<h2>Robustness</h2><div class="scroll">{table(robustness)}</div><h2>Methods and interpretation</h2><pre>{html.escape(text)}</pre></html>'''
    (out/'report.html').write_text(document,encoding='utf-8')
