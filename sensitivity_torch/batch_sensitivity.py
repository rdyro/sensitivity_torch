import math, pdb, os, sys, time
from functools import reduce
from operator import mul
from typing import Mapping, Callable, Union, Sequence
from copy import copy

import matplotlib.pyplot as plt, torch
from torch import Tensor

from .utils import t, topts, ss, fn_with_sol_cache
from .differentiation import JACOBIAN, HESSIAN, HESSIAN_DIAG, fwd_grad, grad
from .differentiation import BATCH_JACOBIAN, BATCH_HESSIAN, BATCH_HESSIAN_DIAG
from .specialized_matrix_inverse import solve_gmres, solve_cg

prod = lambda zs: reduce(mul, zs, 1)


def _generate_default_Dzk_solve_fn(optimizations: Mapping, k_fn):
    """Generates the default Dzk (embedding Hessian) solution function A x = y

    Args:
        optimizations: dictionary with optional problem-specific optimizations
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function

    Returns:
        The `Dzk_solve_fn(z, *params, rhs=rhs, T=False)`, solving Dzk x = rhs.
    """

    def Dzk_solve_fn(z, *params, rhs=None, T=False):
        blen, zlen = z.shape[0], prod(z.shape[1:])
        if optimizations.get("Dzk", None) is None:
            optimizations["Dzk"] = BATCH_JACOBIAN(lambda z: k_fn(z, *params), z)
        Dzk = optimizations["Dzk"]
        if T:
            if optimizations.get("FT", None) is None:
                optimizations["FT"] = torch.linalg.lu_factor(
                    t(Dzk.reshape((blen, zlen, zlen)))
                )
            return torch.lu_solve(rhs, *optimizations["FT"])
        else:
            if optimizations.get("F", None) is None:
                optimizations["F"] = torch.linalg.lu_factor(
                    Dzk.reshape((blen, zlen, zlen))
                )
            return torch.lu_solve(rhs, *optimizations["F"])

    optimizations["Dzk_solve_fn"] = Dzk_solve_fn
    return


def _ensure_list(a):
    """Optionally convert a single element to a list. Leave a list unchanged."""
    return a if isinstance(a, (list, tuple)) else [a]


def implicit_jacobian(
    k_fn: Callable,
    z: Tensor,
    *params: Tensor,
    Dg: Tensor = None,
    jvp_vec: Union[Tensor, Sequence[Tensor]] = None,
    matrix_free_inverse: bool = False,
    full_output: bool = False,
    optimizations: Mapping = None,
):
    """Computes the implicit Jacobian or VJP or JVP depending on Dg, jvp_vec.

    Args:
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function
        z: the optimal embedding variable value array
        *params: the parameters p of the bilevel program
        Dg: left sensitivity vector (wrt z), for a VJP
        jvp_vec: right sensitivity vector(s) (wrt p) for a JVP
        matrix_free_inverse: whether to use approximate matrix inversion
        full_output: whether to append accumulated optimizations to the output
        optimizations: optional optimizations
    Returns:
        Jacobian/VJP/JVP as specified by arguments
    """
    optimizations = {} if optimizations is None else copy(optimizations)
    blen, zlen = z.shape[0], prod(z.shape[1:])
    plen = [prod(param.shape[1:]) for param in params]
    jvp_vec = _ensure_list(jvp_vec) if jvp_vec is not None else jvp_vec

    # construct a default Dzk_solve_fn ##########################
    if optimizations.get("Dzk_solve_fn", None) is None:
        _generate_default_Dzk_solve_fn(optimizations, k_fn)
    #############################################################

    if Dg is not None:
        if matrix_free_inverse:
            raise NotImplementedError
        else:
            Dzk_solve_fn = optimizations["Dzk_solve_fn"]
            v = -Dzk_solve_fn(
                z, *params, rhs=Dg.reshape((blen, zlen, 1)), T=True
            )
        v = v.detach()
        fn = lambda *params: torch.sum(
            v.reshape((blen, zlen)) * k_fn(z, *params).reshape((blen, zlen))
        )
        Dp = JACOBIAN(fn, params)
        Dp_shaped = [Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)]
        ret = Dp_shaped[0] if len(params) == 1 else Dp_shaped
    else:
        if jvp_vec is not None:
            Dp = _ensure_list(
                torch.autograd.functional.jvp(
                    lambda *params: k_fn(z, *params),
                    tuple(params),
                    tuple(jvp_vec),
                )[1]
            )
            Dp = [Dp.reshape((blen, zlen, 1)) for (Dp, plen) in zip(Dp, plen)]
            Dpk = Dp
        else:
            Dpk = _ensure_list(
                BATCH_JACOBIAN(lambda *params: k_fn(z, *params), params)
            )
            Dpk = [
                Dpk.reshape((blen, zlen, plen))
                for (Dpk, plen) in zip(Dpk, plen)
            ]

        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        Dpz = [-Dzk_solve_fn(z, *params, rhs=Dpk, T=False) for Dpk in Dpk]

        if jvp_vec is not None:
            Dpz_shaped = [Dpz.reshape(z.shape) for Dpz in Dpz]
        else:
            Dpz_shaped = [
                Dpz.reshape((blen,) + z.shape[1:] + param.shape[1:])
                for (Dpz, param) in zip(Dpz, params)
            ]
        ret = Dpz_shaped if len(params) != 1 else Dpz_shaped[0]
    return (ret, optimizations) if full_output else ret


def implicit_hessian(
    k_fn: Callable,
    z: Tensor,
    *params: Tensor,
    Dg: Tensor = None,
    Hg: Tensor = None,
    jvp_vec: Union[Tensor, Sequence[Tensor]] = None,
    optimizations: Mapping = None,
):
    """Computes the implicit Hessian or chain rule depending on Dg, Hg, jvp_vec.

    Args:
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function
        z: the optimal embedding variable value array
        *params: the parameters p of the bilevel program
        Dg: gradient sensitivity vector (wrt z), for chain rule
        Hg: Hessian sensitivity vector (wrt z), for chain rule
        jvp_vec: right sensitivity vector(s) (wrt p) for Hessian-vector-product
        optimizations: optional optimizations
    Returns:
        Hessian/chain rule Hessian as specified by arguments
    """
    optimizations = {} if optimizations is None else copy(optimizations)
    blen, zlen = z.shape[0], prod(z.shape[1:])
    plen = [prod(param.shape[1:]) for param in params]
    jvp_vec = _ensure_list(jvp_vec) if jvp_vec is not None else jvp_vec
    if jvp_vec is not None:
        assert Dg is not None

    # construct a default Dzk_solve_fn ##########################
    if optimizations.get("Dzk_solve_fn", None) is None:
        _generate_default_Dzk_solve_fn(optimizations, k_fn)
    #############################################################

    # compute 2nd implicit gradients
    if Dg is not None:
        assert Dg.numel() == blen * zlen
        assert Hg is None or Hg.numel() == blen * zlen ** 2

        Dg_ = Dg.reshape((blen, zlen, 1))
        Hg_ = Hg.reshape((blen, zlen, zlen)) if Hg is not None else Hg

        # compute the left hand vector in the VJP
        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        v = -Dzk_solve_fn(z, *params, rhs=Dg_.reshape((blen, zlen, 1)), T=True)
        v = v.detach()
        fn = lambda z, *params: torch.sum(
            v.reshape((blen, zlen)) * k_fn(z, *params).reshape((blen, zlen))
        )

        if jvp_vec is not None:
            Dpz_jvp = _ensure_list(
                implicit_jacobian(
                    k_fn,
                    z,
                    *params,
                    jvp_vec=jvp_vec,
                    optimizations=optimizations,
                )
            )
            Dpz_jvp = [
                Dpz_jvp.reshape((blen, -1)).detach() for Dpz_jvp in Dpz_jvp
            ]

            # compute the 2nd order derivatives consisting of 4 terms
            # term 1 ##############################
            Dpp1 = [
                torch.autograd.functional.jvp(
                    lambda param: JACOBIAN(
                        lambda param: fn(
                            z, *params[:i], param, *params[i + 1 :]
                        ),
                        param,
                        create_graph=True,
                    ),
                    param,
                    jvp_vec[i],
                )[1].reshape((blen, -1))
                for (i, param) in enumerate(params)
            ]

            # term 2 ##############################
            Dpp2 = [
                torch.autograd.functional.jvp(
                    lambda z: BATCH_JACOBIAN(
                        lambda param: fn(
                            z, *params[:i], param, *params[i + 1 :]
                        ),
                        params[i],
                        create_graph=True,
                    ),
                    z,
                    Dpz_jvp.reshape(z.shape),
                )[1].reshape((blen, -1))
                for (i, (plen, Dpz_jvp)) in enumerate(zip(plen, Dpz_jvp))
            ]

            # term 3 ##############################
            g_ = _ensure_list(
                torch.autograd.functional.jvp(
                    lambda *params: JACOBIAN(
                        lambda z: fn(z, *params), z, create_graph=True
                    ),
                    params,
                    tuple(jvp_vec),
                )[1]
            )
            Dpp3 = [
                _ensure_list(
                    implicit_jacobian(
                        k_fn,
                        z,
                        *params,
                        Dg=g_,
                        optimizations=optimizations,
                    )
                )[i].reshape((blen, -1))
                for (i, g_) in enumerate(g_)
            ]

            # term 4 ##############################
            g_ = [
                torch.autograd.functional.jvp(
                    lambda z: JACOBIAN(
                        lambda z: fn(z, *params), z, create_graph=True
                    ),
                    z,
                    Dpz_jvp.reshape(z.shape),
                )[1]
                for Dpz_jvp in Dpz_jvp
            ]
            if Hg is not None:
                g_ = [
                    g_.reshape((blen, zlen, 1))
                    + Hg_ @ Dpz_jvp.reshape((blen, zlen, 1))
                    for (g_, Dpz_jvp) in zip(g_, Dpz_jvp)
                ]

            Dpp4 = [
                _ensure_list(
                    implicit_jacobian(
                        k_fn,
                        z,
                        *params,
                        Dg=g_,
                        optimizations=optimizations,
                    )
                )[i].reshape((blen, -1))
                for ((i, g_), plen) in zip(enumerate(g_), plen)
            ]
            Dp = [
                Dg_.reshape((blen, 1, zlen)) @ Dpz_jvp.reshape((blen, zlen, 1))
                for Dpz_jvp in Dpz_jvp
            ]
            Dpp = [sum(Dpp) for Dpp in zip(Dpp1, Dpp2, Dpp3, Dpp4)]

            # return the results
            Dp_shaped = [Dp.reshape((blen,)) for Dp in Dp]
            Dpp_shaped = [
                Dpp.reshape(param.shape) for (Dpp, param) in zip(Dpp, params)
            ]
        else:
            # compute the full first order 1st gradients
            Dpz = _ensure_list(
                implicit_jacobian(
                    k_fn,
                    z,
                    *params,
                    optimizations=optimizations,
                )
            )
            Dpz = [
                Dpz.reshape((blen, zlen, plen)).detach()
                for (Dpz, plen) in zip(Dpz, plen)
            ]

            # compute the 2nd order derivatives consisting of 4 terms
            Dpp1 = BATCH_HESSIAN_DIAG(lambda *params: fn(z, *params), params)
            Dpp1 = [
                Dpp1.reshape((blen, plen, plen))
                for (Dpp1, plen) in zip(Dpp1, plen)
            ]

            temp = BATCH_JACOBIAN(
                lambda *params: JACOBIAN(
                    lambda z: fn(z, *params), z, create_graph=True
                ),
                params,
            )
            Dpp2 = [
                (t(temp.reshape((blen, zlen, plen))) @ Dpz).reshape(
                    (blen, plen, plen)
                )
                for (temp, Dpz, plen) in zip(temp, Dpz, plen)
            ]
            Dpp3 = [t(Dpp2) for Dpp2 in Dpp2]
            Dzz = BATCH_HESSIAN(lambda z: fn(z, *params), z).reshape(
                (blen, zlen, zlen)
            )
            if Hg is not None:
                Dpp4 = [t(Dpz) @ (Hg_ + Dzz) @ Dpz for Dpz in Dpz]
            else:
                Dpp4 = [t(Dpz) @ Dzz @ Dpz for Dpz in Dpz]
            Dp = [Dg_.reshape((blen, 1, zlen)) @ Dpz for Dpz in Dpz]
            Dpp = [sum(Dpp) for Dpp in zip(Dpp1, Dpp2, Dpp3, Dpp4)]

            # return the results
            Dp_shaped = [
                Dp.reshape(param.shape) for (Dp, param) in zip(Dp, params)
            ]
            Dpp_shaped = [
                Dpp.reshape((blen,) + 2 * param.shape[1:])
                for (Dpp, param) in zip(Dpp, params)
            ]
        return (
            (Dp_shaped[0], Dpp_shaped[0])
            if len(params) == 1
            else (Dp_shaped, Dpp_shaped)
        )
    else:
        Dpz, optimizations = implicit_jacobian(
            k_fn,
            z,
            *params,
            full_output=True,
            optimizations=optimizations,
        )
        Dpz = _ensure_list(Dpz)
        Dpz = [
            Dpz.reshape((blen, zlen, plen)) for (Dpz, plen) in zip(Dpz, plen)
        ]

        # compute derivatives
        if optimizations.get("Dzzk", None) is None:
            Hk = BATCH_HESSIAN_DIAG(k_fn, (z, *params))
            Dzzk, Dppk = Hk[0], Hk[1:]
            optimizations["Dzzk"] = Dzzk
        else:
            Dppk = BATCH_HESSIAN_DIAG(lambda *params: k_fn(z, *params), params)
        Dppk = [
            Dppk.reshape((blen, zlen, plen, plen))
            for (Dppk, plen) in zip(Dppk, plen)
        ]
        Dzpk = BATCH_JACOBIAN(
            lambda *params: BATCH_JACOBIAN(
                lambda z: k_fn(z, *params), z, create_graph=True
            ),
            params,
        )
        Dzzk = Dzzk.reshape((blen, zlen, zlen, zlen))
        Dzpk = [
            Dzpk.reshape((blen, zlen, zlen, plen))
            for (Dzpk, plen) in zip(Dzpk, plen)
        ]
        Dpzk = [t(Dzpk) for Dzpk in Dzpk]

        # solve the IFT equation
        lhs = [
            Dppk
            + Dpzk @ Dpz[:, None, ...]
            + t(Dpz)[:, None, ...] @ Dzpk
            + (t(Dpz)[:, None, ...] @ Dzzk) @ Dpz[:, None, ...]
            for (Dpz, Dzpk, Dpzk, Dppk) in zip(Dpz, Dzpk, Dpzk, Dppk)
        ]
        Dzk_solve_fn = optimizations["Dzk_solve_fn"]
        Dppz = [
            -Dzk_solve_fn(
                z, *params, rhs=lhs.reshape((blen, zlen, plen * plen)), T=False
            ).reshape((blen, zlen, plen, plen))
            for (lhs, plen) in zip(lhs, plen)
        ]

        # return computed values
        Dpz_shaped = [
            Dpz.reshape((blen,) + z.shape[1:] + param.shape[1:])
            for (Dpz, param) in zip(Dpz, params)
        ]
        Dppz_shaped = [
            Dppz.reshape((blen,) + z.shape[1:] + 2 * param.shape[1:])
            for (Dppz, param) in zip(Dppz, params)
        ]
        return (
            (Dpz_shaped[0], Dppz_shaped[0])
            if len(params) == 1
            else (Dpz_shaped, Dppz_shaped)
        )


def _detach_args(*args):
    return tuple(
        arg.detach() if isinstance(arg, torch.Tensor) else arg for arg in args
    )


def generate_optimization_fns(
    loss_fn: Callable,
    opt_fn: Callable,
    k_fn: Callable,
    normalize_grad: bool = False,
    optimizations: Mapping = None,
):
    """Directly generates upper/outer bilevel program derivative functions.

    Args:
        loss_fn: loss_fn(z, *params), upper/outer level loss
        opt_fn: opt_fn(*params) = z, lower/inner argmin function
        k_fn: k_fn(z, *params) = 0, lower/inner implicit function
        normalize_grad: whether to normalize the gradient by its norm
        jit: whether to apply just-in-time (jit) compilation to the functions
    Returns:
        ``f_fn(*params), g_fn(*params), h_fn(*params)``
        parameters-only upper/outer level loss, gradient and Hessian.
    """
    sol_cache = dict()
    opt_fn_ = lambda *args, **kwargs: opt_fn(*args, **kwargs).detach()
    optimizations = {} if optimizations is None else copy(optimizations)

    @fn_with_sol_cache(opt_fn_, sol_cache)
    def f_fn(z, *params):
        z = z.detach() if isinstance(z, torch.Tensor) else z
        params = _detach_args(*params)
        return loss_fn(z, *params)

    @fn_with_sol_cache(opt_fn_, sol_cache)
    def g_fn(z, *params):
        z = z.detach() if isinstance(z, torch.Tensor) else z
        params = _detach_args(*params)
        g = JACOBIAN(loss_fn, (z, *params))
        Dp = implicit_jacobian(
            k_fn, z.detach(), *params, Dg=g[0], optimizations=optimizations
        )
        Dp = Dp if len(params) != 1 else [Dp]
        # opts = dict(device=z.device, dtype=z.dtype)
        # Dp = [
        #    torch.zeros(param.shape, **opts) for param in params
        # ]
        ret = [Dp + g for (Dp, g) in zip(Dp, g[1:])]
        if normalize_grad:
            ret = [(z / (torch.norm(z) + 1e-7)).detach() for z in ret]
        ret = [ret.detach() for ret in ret]
        return ret[0] if len(ret) == 1 else ret

    @fn_with_sol_cache(opt_fn_, sol_cache)
    def h_fn(z, *params):
        z = z.detach() if isinstance(z, torch.Tensor) else z
        params = _detach_args(*params)
        g = JACOBIAN(loss_fn, (z, *params))

        if optimizations.get("Hz_fn", None) is None:
            optimizations["Hz_fn"] = lambda z, *params: BATCH_HESSIAN_DIAG(
                lambda z: loss_fn(z, *params), (z,)
            )[0]
        Hz_fn = optimizations["Hz_fn"]
        Hz = Hz_fn(z, *params)
        H = [Hz] + BATCH_HESSIAN_DIAG(
            lambda *params: loss_fn(z, *params), params
        )

        Dp, Dpp = implicit_hessian(
            k_fn,
            z,
            *params,
            Dg=g[0],
            Hg=H[0],
            optimizations=optimizations,
        )
        Dpp = Dpp if len(params) != 1 else [Dpp]
        ret = [Dpp + H for (Dpp, H) in zip(Dpp, H[1:])]
        ret = [ret.detach() for ret in ret]
        return ret[0] if len(ret) == 1 else ret

    return f_fn, g_fn, h_fn
