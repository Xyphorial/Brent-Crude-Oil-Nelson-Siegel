"""Nelson–Siegel in months, analytic derivatives and bounded optimisation.
phi = [beta0, beta1, beta2, log(decay_months)].
The penalty applies to scaled INCREMENTS, never the absolute price level.
"""
import time
import numpy as np
from scipy.optimize import minimize


def curve(tau, phi, jacobian=False):
    tau = np.asarray(tau, dtype=float)
    if np.any(tau < 0):
        raise ValueError('Economic tenors must be nonnegative.')
    b0, b1, b2, log_decay = phi
    x = tau / np.exp(log_decay)
    # expm1 and Taylor limits avoid catastrophic cancellation at the origin.
    a = np.ones_like(x)
    da = np.zeros_like(x)
    tiny = x < 1e-5
    a[tiny] = 1 - x[tiny]/2 + x[tiny]**2/6 - x[tiny]**3/24
    da[tiny] = x[tiny]/2 - x[tiny]**2/3 + x[tiny]**3/8
    a[~tiny] = -np.expm1(-x[~tiny]) / x[~tiny]
    da[~tiny] = a[~tiny] - np.exp(-x[~tiny])
    e = np.exp(-x)
    c = a-e
    value = b0 + b1*a + b2*c
    if jacobian:
        derivative = b1*da + b2*(da-x*e)
        return value, np.column_stack([np.ones_like(x), a, c, derivative])
    return value


def initial_guess(tau, y, rng, random=True):
    level = float(y[-1])
    amplitude = max(float(np.ptp(y)), .02 if level < 10 else 1.)
    phi = np.array([level, y[0]-level, 0., np.log(6.)])
    if random:
        phi[:3] += rng.normal(0, amplitude*.3, 3)
        phi[3] = rng.uniform(np.log(1.), np.log(24.))
    return phi


def bounds_for(target):
    # Bounds are deliberately explicit and reported; raw long-end price is positive.
    if target == 'log':
        return np.array([0., -4., -4., np.log(.25)]), np.array([8., 4., 4., np.log(60.)])
    return np.array([.01, -500., -500., np.log(.25)]), np.array([500., 500., 500., np.log(60.)])


def objective(phi, tau, y, weights, previous, scales, strength, alpha, epsilon):
    fitted, jac = curve(tau, phi, True)
    errors = fitted-y
    weights = np.asarray(weights)/np.sum(weights)
    loss = float(np.sum(weights*errors**2))
    gradient = 2*jac.T @ (weights*errors)
    penalty = 0.
    if previous is not None and strength > 0:
        delta = (phi-previous)/scales
        root = np.sqrt(delta**2 + epsilon**2)
        # Smooth L1 approximation, NOT an exact sparse Lasso solver.
        penalty = strength*((1-alpha)*np.sum(delta**2) + alpha*np.sum(root-epsilon))
        gradient += strength*(2*(1-alpha)*delta + alpha*delta/root)/scales
    return loss+penalty, gradient, loss, float(penalty)


def fit(tau, y, weights, start, target, scales=None, previous=None,
        strength=0., alpha=0., epsilon=.01, solver='L-BFGS-B'):
    scales = np.ones(4) if scales is None else np.asarray(scales)
    low, high = bounds_for(target)
    start = np.clip(start, low+1e-10, high-1e-10)
    # Optimisation coordinates are scaled as well as penalty coordinates.
    origin = start.copy()
    # Multiply the ENTIRE objective by one positive constant: minimiser unchanged,
    # but absolute stopping tolerances no longer dwarf small log-price losses.
    numerical_scale = 1. / max(float(np.var(y)), 1e-6 if target == 'log' else .01)
    def fun(u):
        phi = origin + u*scales
        value, grad, _, _ = objective(phi, tau, y, weights, previous, scales, strength, alpha, epsilon)
        return value*numerical_scale, grad*scales*numerical_scale
    tick = time.perf_counter()
    options = {'maxiter': 700, 'ftol': 1e-12}
    if solver == 'L-BFGS-B':
        options.update(gtol=1e-8, maxls=40)
    result = minimize(fun, np.zeros(4), jac=True, method=solver,
                      bounds=list(zip((low-origin)/scales, (high-origin)/scales)), options=options)
    phi = origin + result.x*scales
    value, grad, loss, penalty = objective(phi, tau, y, weights, previous, scales, strength, alpha, epsilon)
    # Projected gradient is appropriate at a box-constrained solution.
    gu = grad*scales
    at_low = np.isclose(phi, low, atol=1e-7)
    at_high = np.isclose(phi, high, atol=1e-7)
    projected = gu.copy()
    projected[at_low & (gu > 0)] = 0
    projected[at_high & (gu < 0)] = 0
    return phi, dict(success=bool(result.success), message=str(result.message),
                     iterations=int(result.nit), nfev=int(result.nfev),
                     objective=float(value), data_loss=loss, penalty=penalty,
                     gradient_norm=float(np.linalg.norm(gu)),
                     projected_gradient_norm=float(np.linalg.norm(projected)),
                     bound_hit=bool(np.any(at_low | at_high)),
                     elapsed_seconds=time.perf_counter()-tick)
