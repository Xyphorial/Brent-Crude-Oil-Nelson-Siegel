"""Chronological calibration, train-only scales and constrained validation tuning."""
import numpy as np
import pandas as pd
from .model import curve, fit, initial_guess


def target_values(prices,target):
    return np.log(prices) if target=='log' else prices


def price_values(values,target):
    return np.exp(values) if target=='log' else values


def calibrate(slices, config, method='warm', strength=0., alpha=0., scales=None, seed=None):
    rng=np.random.default_rng(config['seed'] if seed is None else seed)
    if scales is None:
        scales=np.array([.1,.1,.1,1.]) if config['target']=='log' else np.array([10.,10.,10.,1.])
    previous=None; records=[]; residuals=[]; previous_grid=None; previous_market=None
    fixed=None; elapsed=0
    for index,sl in enumerate(slices):
        y=target_values(sl['prices'],config['target']); tau=sl['tau']
        if fixed is None:
            fixed=initial_guess(tau,y,rng,random=False)
        forecast_rmse=np.nan
        if previous is not None:
            forecast=price_values(curve(tau,previous),config['target'])
            forecast_rmse=float(np.sqrt(np.mean((forecast-sl['prices'])**2)))
        if method=='fixed':
            starts=[fixed]
        elif method in ('random','multistart'):
            starts=[initial_guess(tau,y,rng) for _ in range(config['multistarts'] if method=='multistart' else 1)]
        else:
            starts=[previous.copy() if previous is not None else initial_guess(tau,y,rng)]
        active_strength=strength*min(1.,index/max(1,config['burn_in']))
        reference=previous if method in ('ridge','lasso_smooth','elastic') else None
        candidates=[fit(tau,y,sl['weights'],p,config['target'],scales,reference,
                        active_strength,alpha,config['l1_epsilon'],config['solver']) for p in starts]
        # Failures are recorded, not silently passed off as successful solutions.
        valid=[c for c in candidates if c[1]['success'] and np.all(np.isfinite(c[0]))]
        pool=valid or candidates
        phi,diag=min(pool,key=lambda c:c[1]['objective'])
        diag['total_nfev']=sum(c[1]['nfev'] for c in candidates)
        diag['total_elapsed_seconds']=sum(c[1]['elapsed_seconds'] for c in candidates)
        fitted=price_values(curve(tau,phi),config['target']); error=fitted-sl['prices']
        grid=np.linspace(max(3.,tau.min()),min(12.,tau.max()),30)
        # Fixed 3–12 month grid only when observed range covers it; no extrapolation.
        full_grid=tau.min()<=3 and tau.max()>=12
        fitted_grid=price_values(curve(grid,phi),config['target'])
        market_grid=np.interp(grid,tau,sl['prices'])
        response=np.nan; curve_move=np.nan
        if full_grid and previous_grid is not None:
            curve_move=float(np.sqrt(np.mean((fitted_grid-previous_grid)**2)))
            response=float(np.sqrt(np.mean(((fitted_grid-previous_grid)-(market_grid-previous_market))**2)))
        step=float(np.linalg.norm((phi-previous)/scales)) if previous is not None else np.nan
        # Do not propagate optimiser failures to the next date.
        if diag['success']:
            previous=phi.copy()
            previous_grid=fitted_grid if full_grid else None
            previous_market=market_grid if full_grid else None
        else:
            previous_grid=None; previous_market=None
        records.append(dict(date=sl['date'],method=method,strength=strength,
                            active_strength=active_strength,alpha=alpha,
                            beta0=phi[0],beta1=phi[1],beta2=phi[2],log_lambda=phi[3],
                            decay_months=np.exp(phi[3]),n_contracts=len(tau),
                            rmse=float(np.sqrt(np.mean(error**2))),mae=float(np.mean(np.abs(error))),
                            max_abs_error=float(np.max(np.abs(error))),scaled_step=step,
                            forecast_rmse=forecast_rmse,response_rmse=response,curve_move=curve_move,
                            roster_change=sl['roster_change'],**diag))
        for ticker,t,p,f,e in zip(sl['tickers'],tau,sl['prices'],fitted,error):
            residuals.append(dict(date=sl['date'],method=method,ticker=ticker,tenor=t,
                                  observed=p,fitted=f,residual=e))
    return pd.DataFrame(records),pd.DataFrame(residuals)


def split_labels(count,config):
    train=int(count*config['train_fraction'])
    valid=int(count*config['validation_fraction'])
    if train<config['burn_in']+5 or valid<5 or count-train-valid<5:
        raise ValueError('Insufficient complete slices: need burn-in + training, >=5 validation and >=5 test days.')
    return train,train+valid


def estimate_scales(training,config):
    preliminary,_=calibrate(training,config,method='multistart')
    good=preliminary.loc[preliminary.success,['beta0','beta1','beta2','log_lambda']].to_numpy()
    if len(good)<5:
        raise RuntimeError('Too few successful training fits to estimate scales.')
    median=np.median(good,axis=0)
    robust=1.4826*np.median(np.abs(good-median),axis=0)
    floor=np.array([.01,.02,.02,.15]) if config['target']=='log' else np.array([1.,2.,2.,.15])
    scales=np.maximum(robust,floor)
    # Baseline loss sets strength units using TRAINING quotes only.
    loss_unit=max(float(preliminary.loc[preliminary.success,'data_loss'].median()),
                  1e-8 if config['target']=='log' else 1e-4)
    return scales,loss_unit,preliminary


def summary(frame):
    return dict(rmse=float(np.sqrt(np.mean(frame.rmse**2))),mae=float(frame.mae.mean()),
                mean_scaled_step=float(frame.scaled_step.mean()),
                p95_scaled_step=float(frame.scaled_step.quantile(.95)),
                response_rmse=float(np.sqrt(np.nanmean(frame.response_rmse**2))),
                forecast_rmse=float(np.sqrt(np.nanmean(frame.forecast_rmse**2))),
                failures=int((~frame.success).sum()),bound_hits=int(frame.bound_hit.sum()),
                mean_nfev=float(frame.total_nfev.mean()),n_days=len(frame))


def tune(slices,config,scales,loss_unit,train_end,validation_end,method,alpha):
    # Candidate loops receive only train+validation slices, never the test period.
    prefix=slices[:validation_end]
    baseline,_=calibrate(prefix,config,'warm',scales=scales)
    base=summary(baseline.iloc[train_end:validation_end])
    rows=[]
    for mult in config['penalty_multipliers']:
        strength=loss_unit*mult
        result,_=calibrate(prefix,config,method,strength,alpha,scales)
        score=summary(result.iloc[train_end:validation_end])
        fit_ok=score['rmse']<=base['rmse']*(1+config['fit_tolerance'])
        response_ok=score['response_rmse']<=base['response_rmse']*(1+config['response_tolerance'])
        feasible=fit_ok and response_ok and score['failures']==0
        rows.append(dict(method=method,multiplier=mult,strength=strength,
                         feasible=feasible,fit_ok=fit_ok,response_ok=response_ok,**score))
    table=pd.DataFrame(rows)
    feasible=table.loc[table.feasible]
    if len(feasible):
        chosen=feasible.sort_values(['mean_scaled_step','rmse']).iloc[0]
    else:
        chosen=table.loc[table.multiplier.eq(0)].iloc[0]
    table['selected']=table.strength.eq(chosen.strength)
    return float(chosen.strength),table


def run_study(slices,config):
    train_end,validation_end=split_labels(len(slices),config)
    scales,unit,preliminary=estimate_scales(slices[:train_end],config)
    specs=[('fixed',0.,0.),('random',0.,0.),('multistart',0.,0.),('warm',0.,0.)]
    tuning=[]
    for method,alpha in [('ridge',0.),('lasso_smooth',1.),('elastic',.5)]:
        strength,table=tune(slices,config,scales,unit,train_end,validation_end,method,alpha)
        tuning.append(table); specs.append((method,strength,alpha))
    results=[]; errors=[]; scores=[]
    for method,strength,alpha in specs:
        print(f'Calibrating {method} (strength={strength:.3g})',flush=True)
        frame,residual=calibrate(slices,config,method,strength,alpha,scales)
        frame['split']=['train' if i<train_end else 'validation' if i<validation_end else 'test' for i in range(len(frame))]
        # Operational criterion: five successive successful scaled increments <= .25.
        stable=(frame.scaled_step.le(.25)&frame.success).rolling(5,min_periods=5).sum().eq(5)
        reached=np.flatnonzero(stable.to_numpy())
        first_stable=int(reached[0]+1) if len(reached) else None
        for split,group in frame.groupby('split',sort=False):
            scores.append(dict(method=method,split=split,strength=strength,
                               stability_first_slice=first_stable,**summary(group)))
        results.append(frame); errors.append(residual)
    all_results=pd.concat(results,ignore_index=True)
    test=pd.DataFrame(scores).query("split=='test'")
    # Final model choice based only on validation, not the test leaderboard.
    val=pd.DataFrame(scores).query("split=='validation' and method in ['warm','ridge','lasso_smooth','elastic']")
    baseline=val.loc[val.method.eq('warm')].iloc[0]
    feasible=val.loc[(val.rmse<=baseline.rmse*(1+config['fit_tolerance'])) &
                     (val.response_rmse<=baseline.response_rmse*(1+config['response_tolerance'])) & val.failures.eq(0)]
    preferred=feasible.sort_values(['mean_scaled_step','rmse']).iloc[0].method if len(feasible) else 'warm'
    metadata=dict(scales=scales.tolist(),training_loss_unit=unit,train_slices=train_end,
                  validation_end_slice=validation_end,test_slices=len(slices)-validation_end,
                  preferred_model=preferred,selection='validation constraints then minimum mean scaled step',
                  lasso='smooth L1 increment approximation; epsilon='+str(config['l1_epsilon']),
                  stability_definition='5 consecutive successful fits with scaled step <= 0.25; diagnostic only')
    return all_results,pd.concat(errors,ignore_index=True),pd.DataFrame(scores),pd.concat(tuning),metadata


def cross_section_holdout(slices,config,scales,strength,alpha,method,validation_end):
    # Every fourth tenor is genuinely withheld from each day's calibration.
    reduced=[]; held=[]
    for sl in slices:
        mask=np.arange(len(sl['tau']))%4==2
        reduced.append({**sl,**{k:sl[k][~mask] for k in ('tau','prices','tickers','weights')}})
        held.append({k:sl[k][mask] for k in ('tau','prices','tickers','weights')})
    fitted,_=calibrate(reduced,config,method,strength,alpha,scales)
    records=[]
    for index,(row,h) in enumerate(zip(fitted.itertuples(),held)):
        if index<validation_end:
            continue
        phi=np.array([row.beta0,row.beta1,row.beta2,row.log_lambda])
        prediction=price_values(curve(h['tau'],phi),config['target'])
        records.extend(dict(date=row.date,ticker=t,observed=p,predicted=f,error=f-p,success=row.success)
                       for t,p,f in zip(h['tickers'],h['prices'],prediction))
    return pd.DataFrame(records)
