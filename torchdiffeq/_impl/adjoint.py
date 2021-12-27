import warnings

from matplotlib.pyplot import autoscale
import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .misc import _check_inputs
from .misc import _mixed_norm, _flat_to_shape


class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad

        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
            y = ans
            ctx.save_for_backward(t, y, *adjoint_params)

        return ans

    @staticmethod
    def backward(ctx, *grad_y):
        with torch.no_grad():
            func = ctx.func
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad

            # Backprop as if integrating up to event time.
            # Does NOT backpropagate through the event time.
            t, y, *adjoint_params = ctx.saved_tensors
            grad_y = grad_y[0]

            adjoint_params = tuple(adjoint_params)

            ##################################
            #      Set up initial state      #
            ##################################

            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            tmp = [torch.zeros((), dtype=y.dtype, device=y.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
            tmp.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params

            aug_flat = torch.cat([f_.reshape(-1) for f_ in tmp])
            shapes = [f_.shape for f_ in tmp]
            numel = [f_.numel() for f_ in tmp]
            cumel = [0]+[sum(numel[:i+1]) for i in range(len(numel))]

            ##################################
            #    Set up backward ODE func    #
            ##################################

            # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
            def augmented_dynamics(t, y_aug):
                # Dynamics of the original system augmented with
                # the adjoint wrt y, and an integrator wrt t and args.
                def get(i):
                    return y_aug[cumel[i]:cumel[i+1]].view((*(), *shapes[i]))
                y = get(1)
                adj_y = get(2)
                # ignore gradients wrt time and parameters

                with torch.enable_grad():
                    t_ = t.detach()
                    t = t_.requires_grad_(True)
                    y = y.detach().requires_grad_(True)

                    # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
                    # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
                    # wrt t here means we won't compute that if we don't need it.
                    func_eval = func(t if t_requires_grad else t_, y)

                    # Workaround for PyTorch bug #39784
                    _t = torch.as_strided(t, (), ())  # noqa
                    _y = torch.as_strided(y, (), ())  # noqa
                    _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)  # noqa

                    vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                        func_eval, (t, y) + adjoint_params, -adj_y,
                        allow_unused=True, retain_graph=True
                    )

                # autograd.grad returns None if no gradient, set to zero.
                vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
                vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                              for param, vjp_param in zip(adjoint_params, vjp_params)]

                return torch.cat([f_.reshape(-1) for f_ in (vjp_t, func_eval, vjp_y, *vjp_params)])

            ##################################
            #       Solve adjoint ODE        #
            ##################################

            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
            else:
                time_vjps = None
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    # Compute the effect of moving the current time measurement point.
                    # We don't compute this unless we need to, to save some computation.
                    func_eval = func(t[i], y[i])
                    dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                    aug_flat[cumel[0]:cumel[0+1]] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t

                # Run the augmented system backwards in time.

                aug_flat = odeint(
                    augmented_dynamics, aug_flat,
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
                # print('y', y[i - 1].shape, y[i - 1])
                # print('grad', grad_y[i - 1].shape, grad_y[i - 1])
                # print('aug', aug_flat[1].shape)
                aug_flat = aug_flat[1]  # extract just the t[i - 1] value
                aug_flat[cumel[1]:cumel[1+1]] = y[i - 1].reshape(-1)  # update to use our forward-pass estimate of the state
                aug_flat[cumel[2]:cumel[2+1]] += grad_y[i - 1].reshape(-1)  # update any gradients wrt state at this time point

            if t_requires_grad:
                time_vjps[0] = aug_flat[cumel[0]:cumel[0+1]].view((*(), *shapes[0]))

            adj_y = aug_flat[cumel[2]:cumel[2+1]].view((*(), *shapes[2]))
            adj_params = _flat_to_shape(aug_flat[cumel[3]:], (), shapes[3:])

        return (None, None, adj_y, time_vjps, None, None, None, None, None, None, None, None, None, None, *adj_params)


def odeint_adjoint(func, y0, t, rtol=1e-7, atol=1e-9, method=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    assert method

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    adjoint_rtol = rtol
    adjoint_atol = atol
    adjoint_method = method

    adjoint_options = {}
    assert not getattr(func, '_is_replica', False)
    adjoint_params = tuple(list(func.parameters()))


    # Filter params that don't require gradients.
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)
    
    # Convert to flattened state.
    shapes, func, y0, t, rtol, atol, method, options, event_fn, _ = _check_inputs(func, y0, t, rtol, atol, method, None, None, SOLVERS)

    # Handle the adjoint norm function.
    state_norm = options["norm"] #rms
    handle_adjoint_norm_(adjoint_options, shapes, state_norm)

    assert shapes is None
    return OdeintAdjointMethod.apply(shapes, func, y0, t, rtol, atol, method, options, event_fn, adjoint_rtol, adjoint_atol,
                                    adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)


def handle_adjoint_norm_(adjoint_options, shapes, state_norm):
    """In-place modifies the adjoint options to choose or wrap the norm function."""

    # This is the default adjoint norm on the backward pass: a mixed norm over the tuple of inputs.
    def default_adjoint_norm(tensor_tuple):
        t, y, adj_y, *adj_params = tensor_tuple
        # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
        return max(t.abs(), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

    assert "norm" not in adjoint_options
    # `adjoint_options` was not explicitly specified by the user. Use the default norm.
    adjoint_options["norm"] = default_adjoint_norm

