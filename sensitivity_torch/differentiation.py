import functools
import operator
import torch
from tqdm import tqdm


def JACOBIAN(fn, inputs, **config):
    config.setdefault("vectorize", True)
    return torch.autograd.functional.jacobian(fn, inputs, **config)


JACOBIAN.__doc__ = """Equivalent to torch.autograd.functional.jacobian"""


def HESSIAN(fn, inputs, **config):
    config.setdefault("vectorize", True)
    return torch.autograd.functional.hessian(fn, inputs, **config)


HESSIAN.__doc__ = """Equivalent to torch.autograd.functional.hessian"""


def HESSIAN_CUSTOM(fn, args, **config):
    single_input = not isinstance(args, (list, tuple))
    args = (args,) if single_input else tuple(args)

    f = fn(*args)
    n = f.numel()
    if n == 1:
        return HESSIAN(fn, args, **config)
    else:
        Hs = [HESSIAN(lambda *args: fn(*args).reshape(-1)[i], args, **config) for i in range(n)]
        Hs = [
            [
                torch.stack([Hs[k][i][j] for k in range(n)], 0).reshape(f.shape + Hs[0][i][j].shape)
                for i in range(len(args))
            ]
            for j in range(len(args))
        ]
        return Hs


def HESSIAN_DIAG(fn, args, **config):
    """Generates a function which computes per-argument partial Hessians."""
    single_input = not isinstance(args, (list, tuple))
    args = (args,) if single_input else tuple(args)
    try:
        ret = [
            HESSIAN(lambda arg: fn(*args[:i], arg, *args[i + 1 :]), arg, **config)
            for (i, arg) in enumerate(args)
        ]
    except RuntimeError:  # function has more than 1 output, need to use JACOBIAN
        ret = [
            JACOBIAN(
                lambda arg: JACOBIAN(
                    lambda arg: fn(*args[:i], arg, *args[i + 1 :]),
                    arg,
                    **dict(config, create_graph=True),
                ),
                arg,
                **config,
            )
            for (i, arg) in enumerate(args)
        ]
    return ret[0] if single_input else ret


def BATCH_JACOBIAN(fn, args, **config):
    """Computes the Hessian, assuming the first in/out dimension is the batch."""
    single_input = not isinstance(args, (list, tuple))
    args = (args,) if single_input else tuple(args)
    Js = JACOBIAN(lambda *args: torch.sum(fn(*args), 0), args, **config)
    out_shapes = [J.shape[: -len(arg.shape)] for (J, arg) in zip(Js, args)]
    Js = [
        J.reshape((prod(out_shape),) + arg.shape)
        .swapaxes(0, 1)
        .reshape((arg.shape[0],) + out_shape + arg.shape[1:])
        for (J, out_shape, arg) in zip(Js, out_shapes, args)
    ]
    return Js[0] if single_input else Js


def BATCH_HESSIAN(fn, args, **config):
    """Computes the Hessian, assuming the first in/out dimension is the batch."""
    single_input = not isinstance(args, (list, tuple))
    args = (args,) if single_input else tuple(args)
    assert single_input
    return BATCH_JACOBIAN(
        lambda *args: BATCH_JACOBIAN(fn, args, **dict(config, create_graph=True))[0],
        args,
        **config,
    )[0]


def BATCH_HESSIAN_DIAG(fn, args, **config):
    """Evaluates per-argument partial batch (first dimension) Hessians."""
    single_input = not isinstance(args, (list, tuple))
    args = (args,) if single_input else tuple(args)
    try:
        ret = [
            BATCH_HESSIAN(lambda arg: fn(*args[:i], arg, *args[i + 1 :]), arg, **config)
            for (i, arg) in enumerate(args)
        ]
    except RuntimeError:  # function has more than 1 output, need to use JACOBIAN
        assert False
        ret = [
            BATCH_JACOBIAN(
                lambda arg: BATCH_JACOBIAN(
                    lambda arg: fn(*args[:i], arg, *args[i + 1 :]),
                    arg,
                    **dict(config, create_graph=True),
                ),
                arg,
                **config,
            )
            for (i, arg) in enumerate(args)
        ]
    return ret[0] if single_input else ret


####################################################################################################
def prod(x):
    return functools.reduce(operator.mul, x, 1)


def write_list_at(xs, idxs, els):
    assert len(idxs) == len(els)
    k, xs = 0, [x for x in xs]
    for idx in idxs:
        xs[idx] = els[k]
        k += 1
    return xs


def fwd_grad(ys, xs, grad_inputs=None, **kwargs):
    # we only support a single input, otherwise undesirable accumulation occurs
    if isinstance(xs, list) or isinstance(xs, tuple):
        assert len(xs) == 1

    # select only the outputs which have a gradient path/graph
    ys_ = [ys] if not (isinstance(ys, list) or isinstance(ys, tuple)) else ys
    idxs = [i for (i, y) in enumerate(ys_) if y.grad_fn is not None]
    ys_select = [ys_[i] for i in idxs]
    if len(ys_) == 0:
        return [torch.zeros_like(y) for y in ys_]

    # perform the first step of forward mode emulation in reverse mode AD
    vs_ = [torch.ones_like(y, requires_grad=True) for y in ys_select]
    gs_ = torch.autograd.grad(ys_select, xs, grad_outputs=vs_, create_graph=True, allow_unused=True)

    # perform the second step of reverse mode emulation in reverse mode AD
    if grad_inputs is not None:
        # apply the JVP if necessary
        gs_ = torch.autograd.grad(gs_, vs_, grad_outputs=grad_inputs, allow_unused=True)
    else:
        gs_ = torch.autograd.grad(gs_, vs_, allow_unused=True)

    # fill in the unused outputs with zeros (rather than None)
    gs_ = [torch.zeros_like(y) if g is None else g for (g, y) in zip(gs_, ys_select)]

    # fill in the outputs which did not have gradient paths
    ret = write_list_at([torch.zeros_like(y) for y in ys_], idxs, gs_)

    # return a single output if the output was a single element
    return ret if isinstance(ys, list) or isinstance(ys, tuple) else ret[0]


def grad(y, xs, **kwargs):
    gs = torch.autograd.grad(y, xs, **kwargs)
    return gs[0] if not (isinstance(xs, list) or isinstance(xs, tuple)) else gs


####################################################################################################


def reshape_linear(fs):
    if not (isinstance(fs, tuple) or isinstance(fs, list)):
        return [fs], {"size": 1, "nb": 0}
    else:
        ret = [reshape_linear(f) for f in fs]
        vals, trees = [el[0] for el in ret], [el[1] for el in ret]
        tree = {i: trees[i] for i in range(len(trees))}
        tree["size"] = sum([tree["size"] for tree in trees])
        tree["nb"] = len(trees)
        return sum(vals, []), tree


def reshape_struct(fs_flat, tree):
    if tree["nb"] == 0:
        return fs_flat[0]
    k = 0
    fs = [None for i in range(tree["nb"])]
    for i in range(tree["nb"]):
        if tree[i]["nb"] == 0:
            fs[i] = fs_flat[k]
        elif tree[i]["nb"] > 0:
            fs[i] = reshape_struct(fs_flat[k : k + tree[i]["size"]], tree[i])
        else:
            fs[i] = fs_flat[k : k + tree[i]["size"]]
        k += tree[i]["size"]
    return tuple(fs)


def torch_grad(
    fn,
    argnums=0,
    bdims=0,
    create_graph=False,
    retain_graph=False,
    verbose=False,
):
    def g_fn(*args, **kwargs):
        args = list(args)
        argnums_ = list(argnums) if hasattr(argnums, "__iter__") else [argnums]
        for i in range(len(args)):
            if i in argnums_:
                args[i] = torch.as_tensor(args[i])
                args[i].requires_grad = True
        gargs = [arg for (i, arg) in enumerate(args) if i in argnums_]
        fs = fn(*args, **kwargs)
        fs, tree = reshape_linear(fs)
        G = [None for _ in range(len(fs))]
        for j, f in enumerate(fs):
            f_org_shape = f.shape
            f = f.reshape(f.shape[:bdims] + (-1,))
            Js = [
                torch.zeros((f.shape[-1],) + garg.shape, dtype=f.dtype, device=f.device)
                for garg in gargs
            ]
            rng = tqdm(range(f.shape[-1])) if verbose else range(f.shape[-1])
            for i in rng:
                f_ = torch.sum(f[..., i])
                if f_.grad_fn is not None:
                    gs = torch.autograd.grad(
                        f_,
                        gargs,
                        create_graph=create_graph,
                        retain_graph=(
                            create_graph or j < len(fs) - 1 or i < f.shape[-1] - 1 or retain_graph
                        ),
                        allow_unused=True,
                    )
                else:
                    gs = [None for garg in gargs]
                gs = [
                    g
                    if g is not None
                    else torch.zeros(
                        gargs[k].shape,
                        dtype=gargs[k].dtype,
                        device=gargs[k].device,
                    )
                    for (k, g) in enumerate(gs)
                ]
                for k in range(len(Js)):
                    Js[k][i, ...] = gs[k].reshape((-1,) + gs[k].shape)
            for k, J in enumerate(Js):
                lshp = f_org_shape[bdims:]
                bshp = gargs[k].shape[:bdims]
                rshp = gargs[k].shape[bdims:]
                Js[k] = (
                    J.reshape((prod(lshp), prod(bshp), prod(rshp)))
                    .transpose(-2, -3)
                    .reshape(bshp + lshp + rshp)
                )
            if len(Js) > 1 or hasattr(argnums, "__iter__"):
                G[j] = tuple(Js)
            else:
                G[j] = Js[0]

        ret = reshape_struct(G, tree)
        return ret

    return g_fn


def torch_hessian(fn, argnums=0, bdims=0, create_graph=False):
    g_fn = torch_grad(fn, argnums=argnums, bdims=bdims, create_graph=True)
    g2_fn = torch_grad(g_fn, argnums=argnums, bdims=bdims, create_graph=create_graph)
    return g2_fn
