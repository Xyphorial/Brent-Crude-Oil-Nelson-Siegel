"""Read the user's actual Bloomberg-style CSV. Never create or fill prices."""
import hashlib
from pathlib import Path
import re
import numpy as np
import pandas as pd

FIELDS={
    'query_date':'date',
    'ID':'ticker',
    'FUT_MONTH_YR':'contract_month_label',
    'fut_last_trade_dt()':'last_trade_date',
    'px_last(dates=#dt)':'price',
    'px_volume(dates=#dt)':'volume',
    'px_bid(dates=#dt)':'bid',
    'px_ask(dates=#dt)':'ask',
    'px_mid(dates=#dt)':'mid'
}
MONTHS={name:i+1 for i,name in enumerate('JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC'.split())}


def parse_contract_month(label):
    """MAR 26 becomes 2026-03-01; do not infer expiry from month codes."""
    match=re.fullmatch(r'([A-Z]{3}) (\d{2})',str(label).strip().upper())
    if match is None or match[1] not in MONTHS:
        raise ValueError(f'Invalid FUT_MONTH_YR: {label}')
    return pd.Timestamp(year=2000+int(match[2]),month=MONTHS[match[1]],day=1)


def read_quotes(path,config):
    path=Path(path)
    raw=pd.read_csv(path)
    missing=set(FIELDS)-set(raw.columns)
    if missing:
        raise ValueError(f'Missing CSV columns: {sorted(missing)}')
    quotes=raw.rename(columns=FIELDS).copy()
    quotes['date']=pd.to_datetime(quotes.date,errors='raise')
    quotes['last_trade_date']=pd.to_datetime(quotes.last_trade_date,errors='raise')
    quotes['contract_month']=quotes.contract_month_label.map(parse_contract_month)
    if quotes[['date','ticker','contract_month','last_trade_date']].isna().any().any():
        raise ValueError('Missing date, ticker or contract metadata.')
    if quotes.duplicated(['date','ticker']).any():
        raise ValueError('Duplicate date/ticker observations.')
    conflicts=quotes.groupby('ticker')[['contract_month','last_trade_date']].nunique()
    if conflicts.gt(1).any().any():
        raise ValueError('Inconsistent contract-month or last-trade metadata.')
    if (quotes.last_trade_date>=quotes.contract_month).any():
        raise ValueError('Brent last trade must precede contract month.')
    for col in ['price','volume','bid','ask','mid']:
        quotes[col]=pd.to_numeric(quotes[col],errors='raise')
    observed=quotes.price.notna()
    if ((quotes.price.le(0)|~np.isfinite(quotes.price))&observed).any():
        raise ValueError('Observed PX_LAST must be positive and finite.')
    quotes=quotes.loc[quotes.date.between(config['start'],config['end'])].copy()
    if quotes.empty:
        raise ValueError('No rows in sample period.')
    contracts=quotes[['ticker','contract_month','last_trade_date']].drop_duplicates().sort_values('contract_month')
    contracts['expiry_source']='Uploaded CSV: fut_last_trade_dt(); vendor metadata, not independently exchange-verified'
    contracts['quote_unit']='USD/barrel (Brent convention)'
    contracts['expiry_independently_verified']=False
    observed_dates=quotes.loc[quotes.price.notna(),'date'].unique()
    dates=pd.date_range(config['start'],config['end'])
    calendar=pd.DataFrame({'date':dates})
    calendar['has_observed_price']=calendar.date.isin(observed_dates)
    calendar['status']=np.where(calendar.has_observed_price,'quotes_present','no_prices_in_uploaded_file')
    # Status deliberately does not claim to identify official holidays.
    manifest={
        'data_kind':'real',
        'source':'User-uploaded brent_futures.csv; Bloomberg-style field names',
        'price_field':'px_last(dates=#dt)',
        'price_definition':'Vendor last/close field; not asserted to be official settlement',
        'retrieval_timestamp':'Not provided in uploaded CSV',
        'received_date':'2026-10-05',
        'overrides':'Not provided',
        'calendar_verified':False,
        'expiry_source':'fut_last_trade_dt() from uploaded CSV; independent historical ICE cross-check outstanding',
        'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'rows_in_sample':len(quotes),
        'contracts_in_sample':quotes.ticker.nunique(),
        'price_observations':int(quotes.price.notna().sum()),
        'dates_with_prices':len(observed_dates),
        'calendar_days_without_prices':int((~calendar.has_observed_price).sum())
    }
    return quotes,contracts,calendar,manifest


def make_panel(quotes,contracts,calendar,config):
    """Create daily active nearest-K curves from prices actually in the file."""
    days=calendar.loc[calendar.has_observed_price,'date']
    # Include absent day/ticker keys to expose missing data and late listings.
    grid=pd.MultiIndex.from_product([days,contracts.ticker],names=['date','ticker']).to_frame(index=False)
    q=quotes[['date','ticker','price','volume','bid','ask','mid']]
    panel=grid.merge(q,on=['date','ticker'],how='left',validate='one_to_one')
    panel=panel.merge(contracts,on='ticker',validate='many_to_one')
    anchor=panel.contract_month+pd.to_timedelta(config['anchor_day']-1,unit='D')
    panel['economic_tenor_months']=(anchor-panel.date).dt.days/(365.25/12)
    panel['days_to_last_trade']=(panel.last_trade_date-panel.date).dt.days
    panel['month_bucket']=(panel.contract_month.dt.year-panel.date.dt.year)*12+panel.contract_month.dt.month-panel.date.dt.month
    panel['eligible_until']=panel.last_trade_date
    if config['expiry_buffer']:
        # Optional buffer counts observed quote sessions, NOT verified ICE sessions.
        # Beyond-sample expiries far from sample end cannot affect current roster.
        observed=np.sort(days.to_numpy(dtype='datetime64[D]'))
        cutoff={}
        for row in contracts.itertuples():
            if row.last_trade_date>days.max():
                if (row.last_trade_date-days.max()).days<=7*config['expiry_buffer']:
                    raise ValueError('Need post-sample observed sessions for near-term expiry buffer.')
                cutoff[row.ticker]=row.last_trade_date
            else:
                pos=np.searchsorted(observed,np.datetime64(row.last_trade_date.date()),side='right')-1-config['expiry_buffer']
                if pos<0:
                    raise ValueError('Insufficient observed history for expiry buffer.')
                cutoff[row.ticker]=pd.Timestamp(observed[pos])
        panel['eligible_until']=panel.ticker.map(cutoff)
    panel['tradable']=panel.date.le(panel.eligible_until)&panel.economic_tenor_months.gt(0)
    panel['source_price']=panel.price
    panel['post_eligibility_quote']=~panel.tradable&panel.price.notna()
    panel['price']=panel.price.where(panel.tradable)
    panel['valid_price']=panel.price.notna()
    panel['eligible']=panel.tradable&panel.valid_price
    panel['included']=False
    panel['unchanged_price_flag']=panel.groupby('ticker').price.diff().eq(0)&panel.price.notna()
    panel['bid_ask_spread']=panel.ask-panel.bid
    panel['quote_issue_flag']=panel.bid_ask_spread.lt(0)|panel.bid_ask_spread.gt(5)
    panel['exclusion_reason']=np.where(~panel.tradable,'expired_or_buffer',np.where(~panel.valid_price,'no_price_or_not_yet_listed','outside_selected_roster'))
    coverage=[]; roster=[]; slices=[]; previous=set()
    for day,group in panel.groupby('date',sort=True):
        eligible=group.loc[group.eligible].sort_values('economic_tenor_months')
        if config['design']=='fixed_k':
            selected=eligible.head(config['k']); complete=len(selected)==config['k']
        else:
            lo,hi=config['tenor_band']
            selected=eligible.loc[eligible.economic_tenor_months.between(lo,hi)]; complete=len(selected)>=6
        current=set(selected.ticker) if complete else set()
        change=bool(previous and current!=previous)
        coverage.append(dict(date=day,n_eligible=len(eligible),n_selected=len(selected),complete=complete,
                             roster_change=change,missing_active=int((group.tradable&~group.valid_price).sum())))
        roster.append(dict(date=day,entered=';'.join(sorted(current-previous)),exited=';'.join(sorted(previous-current)),roster_change=change))
        if complete:
            panel.loc[selected.index,'included']=True
            panel.loc[selected.index,'exclusion_reason']='included'
            weights=np.ones(len(selected))
            if config['weighting']=='liquidity':
                volume=selected.volume.to_numpy()
                if np.any(~np.isfinite(volume)) or np.any(volume<0):
                    raise ValueError('Selected contracts have missing/invalid volume; use equal weights.')
                weights=np.sqrt(np.maximum(volume,1))
                weights=np.minimum(weights,3*np.median(weights))
            slices.append(dict(date=day,tau=selected.economic_tenor_months.to_numpy(),
                               prices=selected.price.to_numpy(),tickers=selected.ticker.to_numpy(),
                               weights=weights,roster_change=change))
        previous=current
    return panel,pd.DataFrame(roster),pd.DataFrame(coverage),slices
