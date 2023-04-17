import unittest

import torch

try:
    import import_header
except ModuleNotFoundError:
    import tests.import_header

####################################################################################################

torch.set_default_dtype(torch.float64)

from sensitivity_torch.differentiation import JACOBIAN

import objs

X = torch.randn((100, 3))
Y = torch.randn((100, 5))


def generate_test(OPT, *params, name=""):
    def fn(self):
        W = OPT.solve(X, Y, *params)
        g = OPT.grad(W, X, Y, *params)
        self.assertTrue(torch.norm(g) < 1e-5)

        # gradient quality
        g_ = JACOBIAN(lambda W: OPT.fval(W, X, Y, *params), W)
        err = torch.norm(g_ - g)
        self.assertTrue(err < 1e-9)

        # hessian quality
        H = OPT.hess(W, X, Y, *params)
        H_ = JACOBIAN(lambda W: OPT.grad(W, X, Y, *params), W)
        err = torch.norm(H_ - H)
        self.assertTrue(err < 1e-9)

        # Dzk_solve
        H = OPT.hess(W, X, Y, *params).reshape((W.numel(), W.numel()))
        rhs = torch.randn((W.numel(), 3))
        err = torch.norm(
            torch.linalg.solve(H, rhs) - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=False)
        )
        self.assertTrue(err < 1e-9)

        err = torch.norm(
            torch.linalg.solve(H.transpose(-1, -2), rhs)
            - OPT.Dzk_solve(W, X, Y, *params, rhs=rhs, T=True)
        )
        self.assertTrue(err < 1e-9)

    # fn.__name__ = OPT.__class__.__name__
    fn.__name__ = name
    return fn


class DpzTest(unittest.TestCase):
    pass


names = [
    "LS",
    "CE",
    "LS_with_centers",
    "LS_with_diag",
    "CE_with_diag",
]
OPTs = [
    objs.LS(),
    objs.CE(verbose=True, max_it=30),
    objs.OPT_with_centers(objs.LS(), 2),
    # objs.OPT_with_diag(objs.LS()),
    objs.LS_with_diag(),
    objs.OPT_with_diag(objs.CE(verbose=True, max_it=30)),
]
param_list = [
    (1e-1,),
    (1e-1,),
    (torch.tensor([1e-1, 1e-1]),),
    (1e-1 * torch.ones(X.shape[-1] * Y.shape[-1] + 1),),
    # (1e-1 * torch.ones(X.shape[-1]),),
    (1e-1 * torch.ones(X.shape[-1] * (Y.shape[-1] - 1)),),
]

for OPT, params, name in zip(OPTs, param_list, names):
    fn = generate_test(OPT, *params, name=name)
    setattr(DpzTest, "test_%s" % fn.__name__, fn)

if __name__ == "__main__":
    unittest.main(verbosity=2)
