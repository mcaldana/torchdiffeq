import time
import collections
import torch
from .interp import _interp_evaluate, _interp_fit
from .misc import (_compute_error_ratio,
                   _select_initial_step,
                   _optimal_step_size)
from .misc import Perturb
from .solvers import AdaptiveStepsizeODESolver

def tensor_to_cpp(tensor, i=0):
    torch.set_printoptions(precision=17, threshold=10_000)
    s = tensor.__repr__()
    s = s.replace('[', '{')
    s = s.replace(']', '}')
    s = s.replace('\n\n', '\n')
    s = s.replace('tensor', 'torch::tensor')
    s = s.replace(""",
       dtype=torch.float64""", '')
    s = s.replace(""", dtype=torch.float64""", '')
    s = s.replace(')', ',cTOptions)')
    s = s.replace('grad_fn=<CopySlices>,', '')
    s = s.replace('grad_fn=<MeanBackward0>,', '')
    s = s.replace('requires_grad=True,', '')
    s = s.replace('Parameter containing:\n', '')
    s = "const auto t{} = ".format(i) + s + ";\n"
    print(s)
    return

def tl(*tensors):
    for i, t in enumerate(tensors):
        tensor_to_cpp(t, i)


_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha, beta, c_sol, c_error')


_RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')
# Saved state of the Runge Kutta solver.
#
# Attributes:
#     y1: Tensor giving the function value at the end of the last time step.
#     f1: Tensor giving derivative at the end of the last time step.
#     t0: scalar float64 Tensor giving start of the last time step.
#     t1: scalar float64 Tensor giving end of the last time step.
#     dt: scalar float64 Tensor giving the size for the next time step.
#     interp_coeff: list of Tensors giving coefficients for polynomial
#         interpolation between `t0` and `t1`.


def _runge_kutta_step(func, y0, f0, t0, dt, t1, tableau):
    """Take an arbitrary Runge-Kutta step and estimate error.
    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        t1: float64 scalar Tensor giving the end time; equal to t0 + dt. This is used (rather than t0 + dt) to ensure
            floating point accuracy when needed.
        tableau: _ButcherTableau describing how to take the Runge-Kutta step.
    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """

    t0 = t0.to(y0.dtype)
    dt = dt.to(y0.dtype)
    t1 = t1.to(y0.dtype)

    # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
    # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
    k = torch.empty(*f0.shape, len(tableau.alpha) + 1, dtype=y0.dtype, device=y0.device)
    k[..., 0] = f0
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        if alpha_i == 1.:
            # Always step to perturbing just before the end time, in case of discontinuities.
            ti = t1
            perturb = Perturb.PREV
        else:
            ti = t0 + alpha_i * dt
            perturb = Perturb.NONE
        yi = y0 + k[..., :i + 1].matmul(beta_i * dt).view_as(f0)
        f = func(ti, yi, perturb=perturb)
        k[..., i + 1] = f

    if not (tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        yi = y0 + k.matmul(dt * tableau.c_sol).view_as(f0)

    y1 = yi
    f1 = k[..., -1]
    y1_error = k.matmul(dt * tableau.c_error)
    return y1, f1, y1_error, k


# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3


def rk4_alt_step_func(func, t0, dt, t1, y0, f0=None, perturb=False):
    """Smaller error with slightly more compute."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
    k4 = func(t1, y0 + dt * (k1 - k2 + k3), perturb=Perturb.PREV if perturb else Perturb.NONE)
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeODESolver):
    order: int
    tableau: _ButcherTableau
    mid: torch.Tensor

    def __init__(self, func, y0, rtol, atol,
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 max_num_steps=2 ** 31 - 1,
                 dtype=torch.float64,
                 **kwargs):
        super(RKAdaptiveStepsizeODESolver, self).__init__(dtype=dtype, y0=y0, **kwargs)

        # We use mixed precision. y has its original dtype (probably float32), whilst all 'time'-like objects use
        # `dtype` (defaulting to float64).
        dtype = torch.promote_types(dtype, y0.dtype)
        device = y0.device

        self.func = func
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        self.dtype = dtype

        # Copy from class to instance to set device
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=device, dtype=y0.dtype),
                                       beta=[b.to(device=device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=device, dtype=y0.dtype))
        self.mid = self.mid.to(device=device, dtype=y0.dtype)

        self.profiler = {
            "_before_integrate": 0,
            "_select_initial_step": 0,
            "_advance": 0,
            "_adaptive_step": 0,
            "_interp_evaluate": 0,
            "_runge_kutta_step": 0,
            "_compute_error_ratio": 0,
            "_interp_fit": 0,
            "_optimal_step_size": 0,
            "_interp_fit_ext": 0,
        }

    def _before_integrate(self, t):
        t1 = time.time()
        f0 = self.func(t[0], self.y0)

        t3 = time.time()
        first_step = _select_initial_step(self.func, t[0], self.y0, self.order - 1, self.rtol, self.atol,
                                              self.norm, f0=f0)

        
        t4 = time.time()
        self.profiler['_select_initial_step'] += t4 - t3

        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, [self.y0] * 5)
        t2 = time.time()
        self.profiler['_before_integrate'] += t2 - t1

    def _advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        t1 = time.time()
        n_steps = 0
        # while next_t > self.rk_state.t1:
        #     assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
        #     # print('==========step', n_steps)
        #     # tl(*self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, self.rk_state.dt, self.rk_state.y1, self.rk_state.f1)      

        #     self.rk_state = self._adaptive_step(self.rk_state)
        #     n_steps += 1
        self.rk_state = _RungeKuttaState(
            self.rk_state.y1, 
            self.rk_state.f1, 
            self.rk_state.t0, 
            next_t + self.rk_state.dt, 
            self.rk_state.dt, 
            self.rk_state.interp_coeff
        )

        t5 = time.time()
        eval = _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)
        t6 = time.time()
        self.profiler['_interp_evaluate'] += t6 - t5
        
        t2 = time.time()
        self.profiler['_advance'] += t2 - t1
        return eval

    def _adaptive_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        tt1 = time.time()


        y0, f0, _, t0, dt, interp_coeff = rk_state
        t1 = t0 + dt

        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        assert torch.isfinite(y0).all(), 'non-finite values in state `y`: {}'.format(y0)

        t3 = time.time()
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, t1, tableau=self.tableau)
        t4 = time.time()
        self.profiler['_runge_kutta_step'] += t4 - t3

        t5 = time.time()
        error_ratio = _compute_error_ratio(y1_error, self.rtol, self.atol, y0, y1, self.norm)
        t6 = time.time()
        self.profiler['_compute_error_ratio'] += t6 - t5
        accept_step = error_ratio <= 1

        if accept_step:
            t_next = t1
            y_next = y1
            interp_coeff = self._interp_fit(y0, y_next, k, dt)
            f_next = f1
        else:
            t_next = t0
            y_next = y0
            f_next = f0

        t7 = time.time()
        dt_next = _optimal_step_size(dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
        t8 = time.time()
        self.profiler['_optimal_step_size'] += t8 - t7

        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)

        t2 = time.time()
        self.profiler['_adaptive_step'] += t2 - tt1
        return rk_state

    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        t1 = time.time()
        dt = dt.type_as(y0)
        y_mid = y0 + k.matmul(dt * self.mid).view_as(y0)
        f0 = k[..., 0]
        f1 = k[..., -1]

        t3 = time.time()
        fit = _interp_fit(y0, y1, y_mid, f0, f1, dt)
        t4 = time.time()
        self.profiler['_interp_fit_ext'] += t4 - t3

        t2 = time.time()
        self.profiler['_interp_fit'] += t2 - t1
        return fit

    def profile(self):
        print(sorted( ((v,k) for k,v in self.profiler.items()), reverse=True))
