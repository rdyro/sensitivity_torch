import unittest, time
from functools import reduce
from operator import mul

try:
    import import_header
except ModuleNotFoundError:
    import tests.import_header

####################################################################################################

import torch
from sensitivity_torch.batch_sensitivity import implicit_hessian
from sensitivity_torch.sensitivity import implicit_hessian as implicit_hessian_
import objs

OPT = objs.CE()
X = torch.randn((100, 3))
Y = torch.randn((100, 5))
lam = 1e-3
blen = 2
p = torch.randn((blen, 3, 6))
W = OPT.solve(X @ p, Y, lam)

prod = lambda x: reduce(mul, x, 1)

VERBOSE = True


# we test here 2nd order implicit gradients
class DpzTest(unittest.TestCase):
    def test_shape_and_val(self):
        if VERBOSE:
            print()
        k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
        Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(W, X @ p, Y, lam, rhs=rhs, T=T)
        optimizations = dict(Dzk_solve_fn=Dzk_solve_fn)
        Dpz, Dppz = implicit_hessian(k_fn, W, p, optimizations=optimizations)
        self.assertEqual(Dpz.shape, (blen,) + W.shape[1:] + p.shape[1:])
        self.assertEqual(Dppz.shape, (blen,) + W.shape[1:] + p.shape[1:] + p.shape[1:])

        Dpz2, Dppz2 = implicit_hessian(k_fn, W, p)
        self.assertEqual(Dpz2.shape, (blen,) + W.shape[1:] + p.shape[1:])
        self.assertEqual(Dppz2.shape, (blen,) + W.shape[1:] + p.shape[1:] + p.shape[1:])

        Dpz3, Dppz3 = [
            torch.stack(y)
            for y in zip(*[implicit_hessian_(k_fn, W_, p_) for (W_, p_) in zip(W, p)])
        ]
        self.assertEqual(Dppz3.shape, (blen,) + W.shape[1:] + p.shape[1:] + p.shape[1:])

        eps = max(torch.finfo(Dpz.dtype).resolution, 1e-9)

        err_Dpz2 = torch.norm(Dpz - Dpz2)
        err_Dppz2 = torch.norm(Dppz - Dppz2)
        err_Dpz3 = torch.norm(Dpz - Dpz3)
        err_Dppz3 = torch.norm(Dppz - Dppz3)

        if VERBOSE:
            print("err_Dpz: %9.4e" % err_Dpz2)
            print("err_Dppz: %9.4e" % err_Dppz2)
            print("err_Dpz: %9.4e" % err_Dpz3)
            print("err_Dppz: %9.4e" % err_Dppz3)

        self.assertTrue(err_Dpz2 < eps)
        self.assertTrue(err_Dppz2 < eps)
        self.assertTrue(err_Dpz3 < eps)
        self.assertTrue(err_Dppz3 < eps)

    def test_shape_jvp_without_Dzk_solve(self, Dzk_solve_fn=None):
        if VERBOSE:
            print()

        k_fn = lambda W, p: OPT.grad(W, X @ p, Y, lam)
        jvp_vec = torch.randn(p.shape)
        v = torch.randn(W.shape)

        t_ = time.time()
        optimizations = dict(Dzk_solve_fn=Dzk_solve_fn)
        Dpz1, Dppz1 = implicit_hessian(k_fn, W, p, Dg=v, optimizations=optimizations)
        if VERBOSE:
            print("Elapsed %9.4e" % (time.time() - t_))

        t_ = time.time()
        Dpz2, Dppz2 = implicit_hessian(
            k_fn, W, p, Dg=v, jvp_vec=jvp_vec, optimizations=optimizations
        )
        if VERBOSE:
            print("Elapsed %9.4e" % (time.time() - t_))

        self.assertEqual(Dpz2.shape, (blen,))
        self.assertEqual(Dppz2.shape, (blen,) + p.shape[1:])

        Dpz3, Dppz3 = [
            torch.stack(y)
            for y in zip(
                *[
                    implicit_hessian_(k_fn, W_, p_, Dg=v_, jvp_vec=jvp_vec_)
                    for (W_, p_, v_, jvp_vec_) in zip(W, p, v, jvp_vec)
                ]
            )
        ]
        self.assertEqual(Dppz3.shape, (blen,) + p.shape[1:])

        Dpz1 = torch.sum(
            Dpz1.reshape((blen, prod(p.shape[1:]))) * jvp_vec.reshape((blen, prod(p.shape[1:]))),
            -1,
        ).reshape((blen,))
        Dppz1 = torch.sum(
            Dppz1.reshape((blen, prod(p.shape[1:]), prod(p.shape[1:])))
            * jvp_vec.reshape((blen, 1, prod(p.shape[1:]))),
            -1,
        ).reshape(p.shape)

        eps = max(torch.finfo(Dpz1.dtype).resolution, 1e-9)

        err_Dpz2 = torch.norm(Dpz1 - Dpz2)
        err_Dppz2 = torch.norm(Dppz1 - Dppz2)
        err_Dpz3 = torch.norm(Dpz1 - Dpz3)
        err_Dppz3 = torch.norm(Dppz1 - Dppz3)

        if VERBOSE:
            print("err_Dpz: %9.4e" % err_Dpz2)
            print("err_Dppz: %9.4e" % err_Dppz2)
            print("err_Dpz: %9.4e" % err_Dpz3)
            print("err_Dppz: %9.4e" % err_Dppz3)
        self.assertTrue(err_Dpz2 < eps)
        self.assertTrue(err_Dppz2 < eps)
        self.assertTrue(err_Dpz3 < eps)
        self.assertTrue(err_Dppz3 < eps)

    def test_shape_jvp_with_Dzk_solve(self):
        Dzk_solve_fn = lambda W, p, rhs=None, T=False: OPT.Dzk_solve(W, X @ p, Y, lam, rhs=rhs, T=T)
        self.test_shape_jvp_without_Dzk_solve(Dzk_solve_fn=Dzk_solve_fn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
