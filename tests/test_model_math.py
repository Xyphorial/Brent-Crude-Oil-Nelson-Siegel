import unittest
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from brent_ns.model import curve, objective, fit

class NelsonSiegelMathTests(unittest.TestCase):
    def test_jacobian(self):
        p = np.array([4.3, .1, -.08, np.log(6.)])
        t = np.array([0., 2., 6., 12.])
        _, jac = curve(t, p, True)
        np.testing.assert_allclose(jac, approx_derivative(lambda x: curve(t, x), p), rtol=1e-5, atol=1e-8)

    def test_analytic_penalty_gradients(self):
        p = np.array([4.3, .1, -.08, np.log(6.)]); prev = p + .01
        t = np.arange(2.,14.); y = curve(t,prev); scales = np.array([.1,.1,.1,1.])
        for a in [0., .5, 1.]:
            f=lambda v: objective(v,t,y,np.ones(12),prev,scales,.03,a,.01)
            np.testing.assert_allclose(f(p)[1],approx_derivative(lambda v: np.array([f(v)[0]]),p).ravel(),rtol=1e-5,atol=1e-7)

    def test_noiseless_fit(self):
        t=np.arange(2.,14.); true=np.array([4.3,.12,-.08,np.log(6.)]); y=curve(t,true)
        p,diag=fit(t,y,np.ones(12),np.array([4.25,.1,-.05,np.log(5.)]),'log',np.array([.1,.1,.1,1.]))
        self.assertTrue(diag['success'])
        self.assertLess(np.sqrt(np.mean((curve(t,p)-y)**2)),1e-5)
