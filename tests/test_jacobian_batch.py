import unittest
from functools import reduce
from operator import mul

try:
    import import_header
except ModuleNotFoundError:
    import tests.import_header

####################################################################################################

import torch
from sensitivity_torch.batch_sensitivity import implicit_jacobian
from sensitivity_torch.sensitivity import (
    implicit_jacobian as implicit_jacobian_,
)
import objs

torch.set_default_dtype(torch.float64)

OPT = objs.CE()
X = torch.randn((100, 3))
Y = torch.randn((100, 5))
lam = 1e-3

blen = 7
p = torch.randn((blen, 3, 6))
W = OPT.solve(X @ p, Y, lam)
prod = lambda x: reduce(mul, x, 1)


# we test here 1st order implicit gradients
class DpzTest(unittest.TestCase):
    def test_shapes(self):
        k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)

        def Dzk_solve_fn(W, p, rhs=None, T=False):
            return OPT.Dzk_solve(W, X @ p, Y, lam, rhs=rhs, T=T)

        optimizations = {"Dzk_solve_fn": Dzk_solve_fn}
        Dpz = implicit_jacobian(k_fn, W, p)
        Dpz2 = implicit_jacobian(k_fn, W, p, optimizations=optimizations)
        Dpz3 = torch.stack([implicit_jacobian_(k_fn, W_, p_) for (W_, p_) in zip(W, p)])

        self.assertEqual(Dpz.shape, (blen,) + W.shape[1:] + p.shape[1:])
        self.assertEqual(Dpz2.shape, (blen,) + W.shape[1:] + p.shape[1:])
        self.assertEqual(Dpz3.shape, (blen,) + W.shape[1:] + p.shape[1:])

        err_Dpz2 = torch.norm(Dpz - Dpz2)
        err_Dpz3 = torch.norm(Dpz - Dpz3)

        eps = max(torch.finfo(Dpz.dtype).resolution, 1e-9)

        self.assertTrue(err_Dpz2 < eps)
        self.assertTrue(err_Dpz3 < eps)

    def test_batch_jacobian_with_Dg(self):
        k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
        Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(W, X @ p, Y, lam, rhs=rhs, T=T)
        Dg = torch.randn(W.shape)
        optimizations = {"Dzk_solve_fn": Dzk_solve_fn}
        Dpz1 = implicit_jacobian(k_fn, W, p, optimizations=optimizations)
        Dpz1 = (
            Dg.reshape((blen, 1, prod(W.shape[1:])))
            @ Dpz1.reshape((blen, prod(W.shape[1:]), prod(p.shape[1:])))
        ).reshape(p.shape)

        Dpz2 = implicit_jacobian(k_fn, W, p, Dg=Dg)
        self.assertEqual(Dpz2.shape, p.shape)

        eps = max(torch.finfo(Dpz1.dtype).resolution, 1e-9)

        Dpz3 = torch.stack(
            [implicit_jacobian_(k_fn, W_, p_, Dg=Dg_) for (W_, p_, Dg_) in zip(W, p, Dg)]
        )
        err_Dpz2 = torch.norm(Dpz1 - Dpz2)
        err_Dpz3 = torch.norm(Dpz1 - Dpz3)

        self.assertTrue(err_Dpz2 < eps)
        self.assertTrue(err_Dpz3 < eps)

    def test_batch_jacobian_with_jvp(self):
        prod = lambda x: reduce(mul, x, 1)
        k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
        Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(W, X @ p, Y, lam, rhs=rhs, T=T)
        jvp_vec = torch.randn(p.shape)
        optimizations = {"Dzk_solve_fn": Dzk_solve_fn}
        Dpz1 = implicit_jacobian(k_fn, W, p, optimizations=optimizations)
        Dpz1 = (
            Dpz1.reshape((blen, prod(W.shape[1:]), prod(p.shape[1:])))
            @ jvp_vec.reshape((blen, prod(p.shape[1:]), 1))
        ).reshape(W.shape)

        Dpz2 = implicit_jacobian(k_fn, W, p, jvp_vec=jvp_vec)
        self.assertEqual(Dpz2.shape, W.shape)

        eps = max(torch.finfo(Dpz1.dtype).resolution, 1e-9)

        Dpz3 = torch.stack(
            [
                implicit_jacobian_(k_fn, W_, p_, jvp_vec=jvp_vec_)
                for (W_, p_, jvp_vec_) in zip(W, p, jvp_vec)
            ]
        )
        err_Dpz2 = torch.norm(Dpz1 - Dpz2)
        err_Dpz3 = torch.norm(Dpz1 - Dpz3)

        self.assertTrue(err_Dpz2 < eps)
        self.assertTrue(err_Dpz3 < eps)

    # (Dg XOR jvp_vec) currently only supported
    # def test_batch_jacobian_with_jvp_and_Dg(self):
    #    prod = lambda x: reduce(mul, x, 1)
    #    k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
    #    Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(
    #        W, X @ p, Y, lam, rhs=rhs, T=T
    #    )
    #    jvp_vec = torch.randn(p.shape)
    #    Dg = torch.randn(W.shape)
    #    optimizations = {"Dzk_solve_fn": Dzk_solve_fn}
    #    Dpz1 = implicit_jacobian(k_fn, W, p, optimizations=optimizations)
    #    Dpz1 = (
    #        Dg.reshape((blen, 1, prod(W.shape[1:])))
    #        @ Dpz1.reshape((blen, prod(W.shape[1:]), prod(p.shape[1:])))
    #        @ jvp_vec.reshape((blen, prod(p.shape[1:]), 1))
    #    ).reshape(blen)

    #    Dpz2 = implicit_jacobian(k_fn, W, p, jvp_vec=jvp_vec, Dg=Dg)
    #    self.assertEqual(Dpz2.shape, (blen,))

    #    eps = max(torch.finfo(Dpz1.dtype).resolution, 1e-9)

    #    Dpz3 = torch.stack(
    #        [
    #            implicit_jacobian_(k_fn, W_, p_, jvp_vec=jvp_vec_, Dg=Dg_)
    #            for (W_, p_, jvp_vec_, Dg_) in zip(W, p, jvp_vec, Dg)
    #        ]
    #    )
    #    err_Dpz2 = torch.norm(Dpz1 - Dpz2)
    #    err_Dpz3 = torch.norm(Dpz1 - Dpz3)

    #    self.assertTrue(err_Dpz2 < eps)
    #    self.assertTrue(err_Dpz3 < eps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
