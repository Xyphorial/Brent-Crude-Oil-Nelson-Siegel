"""完整运行：真实 CSV → 数据整理 → 曲线拟合 → 验证 → 报告。"""
import argparse, hashlib, importlib.metadata, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
from brent_ns.data import read_quotes, make_panel
from brent_ns.research import run_study, split_labels, estimate_scales, tune, calibrate, summary, cross_section_holdout
from brent_ns.report import write_report
ROOT=Path(__file__).resolve().parent

def resolve(path):
    p=Path(path).expanduser()
    return p if p.is_absolute() else ROOT/p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/brent_futures.csv')
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--output',default='output/my_run')
    parser.add_argument('--skip-robustness',action='store_true')
    args=parser.parse_args()
    config=json.loads(resolve(args.config).read_text(encoding='utf-8-sig'))
    out=resolve(args.output)
    if (out/'run_metadata.json').exists():
        raise FileExistsError('请使用新的 --output 目录，避免覆盖此前的结果。')
    out.mkdir(parents=True,exist_ok=True)
    # 1. 读取用户提供的真实价格与到期日，没有模拟数据。
    quotes,contracts,calendar,manifest=read_quotes(resolve(args.input),config)
    # 2. 每天计算期限，剔除到期合约，选择最近 12 个有效合约。
    panel,roster,coverage,slices=make_panel(quotes,contracts,calendar,config)
    for name,frame in [('panel',panel),('roster_changes',roster),('coverage',coverage),
                       ('contract_dictionary',contracts),('observation_calendar',calendar)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    (out/'data_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Uploaded quotes: {manifest["price_observations"]}; complete daily curves: {len(slices)}',flush=True)
    # 3. 拟合并按时间调参；最后一段作为测试期。
    results,residuals,scores,tuning,metadata=run_study(slices,config)
    for name,frame in [('daily_fits',results),('residuals',residuals),('scores',scores),('tuning',tuning)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    results.loc[~results.success].to_csv(out/'failures.csv',index=False)
    preferred=metadata['preferred_model']; selection=results.loc[results.method.eq(preferred)].iloc[0]
    scales=np.array(metadata['scales']); val_end=metadata['validation_end_slice']
    # 4. 留出部分合约，检验当天未参与拟合的价格。
    holdout=cross_section_holdout(slices,config,scales,selection.strength,selection.alpha,preferred,val_end)
    holdout.to_csv(out/'heldout_tenors.csv',index=False)
    # 5. 不同设置下重新在训练/验证数据调参，测试只用来评价。
    variants=[('raw_prices',dict(target='raw')),('all_in_band',dict(design='band')),
              ('midmonth_anchor',dict(anchor_day=15)),('two_observed_session_buffer',dict(expiry_buffer=2)),
              ('liquidity_weights',dict(weighting='liquidity')),('alternative_solver',dict(solver='SLSQP')),
              ('small_l1_epsilon',dict(l1_epsilon=.001))]
    robustness=[]; skipped=[]
    if not args.skip_robustness:
        for name,change in variants:
            print(f'Robustness: {name}',flush=True)
            if name=='liquidity_weights' and panel.loc[panel.included,'volume'].isna().any():
                skipped.append(dict(variant=name,reason='Missing volume on selected contracts')); continue
            cfg={**config,**change}; _,_,_,alt=make_panel(quotes,contracts,calendar,cfg)
            tr,ve=split_labels(len(alt),cfg); s,unit,_=estimate_scales(alt[:tr],cfg)
            method='lasso_smooth' if name=='small_l1_epsilon' else 'ridge'; alpha=1. if method=='lasso_smooth' else 0.
            strength,_=tune(alt,cfg,s,unit,tr,ve,method,alpha)
            fitted,_=calibrate(alt,cfg,method,strength,alpha,s); baseline,_=calibrate(alt,cfg,'warm',scales=s)
            for label,frame in [(method,fitted),('warm',baseline)]:
                robustness.append(dict(variant=name,method=label,strength=strength if label!='warm' else 0.,**summary(frame.iloc[ve:])))
    columns=['variant','method','strength','rmse','mae','mean_scaled_step','p95_scaled_step','response_rmse','forecast_rmse','failures','bound_hits','mean_nfev','n_days']
    robustness=pd.DataFrame(robustness,columns=columns); robustness.to_csv(out/'robustness.csv',index=False)
    seeds=[]
    for seed in [7,42,2026]:
        for method in dict.fromkeys(['random','warm',preferred]):
            strength=float(selection.strength) if method==preferred else 0.
            alpha=float(selection.alpha) if method==preferred else 0.
            frame,_=calibrate(slices,config,method,strength,alpha,scales,seed=seed)
            seeds.append(dict(seed=seed,method=method,**summary(frame.iloc[val_end:])))
    pd.DataFrame(seeds).to_csv(out/'seed_sensitivity.csv',index=False)
    metadata.update(data_kind='real',robustness_run=not args.skip_robustness,skipped_robustness=skipped,
                    complete_slices=len(slices),holdout_rmse=float(np.sqrt(np.mean(holdout.error**2))),
                    calendar_verified=False,expiry_independently_verified=False)
    (out/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    environment=dict(python=sys.version,platform=platform.platform(),packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','matplotlib']})
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    (out/'input_hashes.json').write_text(json.dumps({resolve(args.input).name:manifest['raw_sha256']},indent=2))
    # 6. 输出可直接浏览的图表与报告。
    write_report(out,config,manifest,results,scores,tuning,metadata,coverage,holdout,robustness)
    print(f'Done. Open {out/"report.html"}',flush=True)

if __name__=='__main__':
    main()
