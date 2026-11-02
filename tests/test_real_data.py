import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative
from brent_ns.data import read_quotes,make_panel,parse_contract_month
from brent_ns.model import curve,objective,fit
ROOT=Path(__file__).resolve().parents[1]

class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'data/brent_futures.csv').exists():
            raise unittest.SkipTest('Original licensed Brent quotes not included in public repository')
        cls.config=json.loads((ROOT/'config.json').read_text())
        cls.quotes,cls.contracts,cls.calendar,cls.manifest=read_quotes(ROOT/'data/brent_futures.csv',cls.config)
        cls.panel,cls.roster,cls.coverage,cls.slices=make_panel(cls.quotes,cls.contracts,cls.calendar,cls.config)
    def test_actual_file_counts(self):
        self.assertEqual(len(self.quotes),14263)
        self.assertEqual(self.quotes.price.notna().sum(),9946)
        self.assertEqual(len(self.slices),106)
        self.assertEqual(self.quotes.ticker.nunique(),157)
    def test_first_curve_matches_raw_file(self):
        sl=self.slices[0]
        self.assertEqual(str(sl['date'].date()),'2026-01-02')
        self.assertEqual(sl['tickers'][0],'COH26 Comdty')
        self.assertEqual(sl['prices'][0],60.75)
        self.assertEqual(len(sl['prices']),12)
    def test_no_expired_or_filled_prices(self):
        active=self.panel.loc[self.panel.included]
        self.assertTrue((active.date<=active.last_trade_date).all())
        source=self.quotes.set_index(['date','ticker']).price
        for row in active.itertuples():
            self.assertEqual(row.price,source.loc[(row.date,row.ticker)])
        self.assertTrue(self.roster.roster_change.any())
        self.assertTrue(self.coverage.complete.all())
    def test_empty_dates_are_not_called_holidays(self):
        self.assertEqual((~self.calendar.has_observed_price).sum(),46)
        self.assertFalse(self.manifest['calendar_verified'])
        self.assertEqual(set(self.calendar.status),{'quotes_present','no_prices_in_uploaded_file'})
    def test_month_parser(self):
        self.assertEqual(parse_contract_month('MAR 26'),pd.Timestamp('2026-03-01'))
        with self.assertRaises(ValueError): parse_contract_month('WRONG')
    def test_ns_units_and_derivatives(self):
        phi=np.array([4.3,.1,-.08,np.log(6.)]); tau=np.array([0.,2.,6.,12.])
        year_phi=phi.copy(); year_phi[3]-=np.log(12)
        np.testing.assert_allclose(curve(tau,phi),curve(tau/12,year_phi))
        _,jac=curve(tau,phi,True)
        np.testing.assert_allclose(jac,approx_derivative(lambda v:curve(tau,v),phi),rtol=1e-5,atol=1e-8)
    def test_penalty_gradients(self):
        phi=np.array([4.3,.1,-.08,np.log(6.)]); prev=phi+.01; tau=np.arange(2.,14.); y=curve(tau,prev); scales=np.array([.1,.1,.1,1.])
        for alpha in [0.,.5,1.]:
            f=lambda v:objective(v,tau,y,np.ones(12),prev,scales,.03,alpha,.01)
            np.testing.assert_allclose(f(phi)[1],approx_derivative(lambda v:np.array([f(v)[0]]),phi).ravel(),rtol=1e-5,atol=1e-7)
    def test_noiseless_curve(self):
        tau=np.arange(2.,14.); truth=np.array([4.3,.12,-.08,np.log(6.)]); y=curve(tau,truth)
        phi,diag=fit(tau,y,np.ones(12),np.array([4.25,.1,-.05,np.log(5.)]),'log',np.array([.1,.1,.1,1.]))
        self.assertTrue(diag['success']); self.assertLess(np.sqrt(np.mean((curve(tau,phi)-y)**2)),1e-5)

if __name__=='__main__': unittest.main()
