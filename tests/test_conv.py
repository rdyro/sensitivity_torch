import unittest

import torch

try:
    import import_header
except ModuleNotFoundError:
    import tests.import_header

####################################################################################################

torch.set_default_dtype(torch.float64)

from sensitivity_torch.differentiation import JACOBIAN
from objs import LS, CE, OPT_conv

N = 100
X = torch.randn((N, 3, 6, 5)).reshape((N, -1))
Y = torch.randn((N, 10))

OPT = OPT_conv(LS(), 3, 5, 3, 2)
param = OPT.generate_parameter()
params = (param,)


class DpzTest(unittest.TestCase):
    def test_conv(self):
        W = OPT.solve(X, Y, *params)
        g = OPT.grad(W, X, Y, *params)
        self.assertTrue(torch.norm(g) < 1e-5)

        # gradient quality
        g_ = JACOBIAN(lambda W: OPT.fval(W, X, Y, *params), W)
        err = torch.norm(g_ - g)
        self.assertTrue(err < 1e-9)

        # hessian quality
        H = OPT.hess(W, X, Y, *params)
        err = torch.norm(JACOBIAN(lambda W: OPT.grad(W, X, Y, *params), W) - H)
        self.assertTrue(err < 1e-9)

        # Dzk_solve
        H = OPT.hess(W, X, Y, *params).reshape((W.numel(), W.numel()))
        rhs = torch.randn((W.numel(), 3))
        err = torch.norm(
            torch.linalg.solve(H, rhs) - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=False)
        )
        self.assertTrue(err < 1e-9)
        err = torch.norm(
            torch.linalg.solve(torch.transpose(H, -1, -2), rhs)
            - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=True)
        )
        self.assertTrue(err < 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
