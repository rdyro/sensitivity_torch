

# `batch_sensitivity`

|                                                                                       name                                                                                       |                                  summary                                  |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
|     [generate_optimization_fns(loss_fn, opt_fn, k_fn, normalize_grad, optimizations)](/sensitivity_torch/api/sensitivity_torch/batch_sensitivity/generate_optimization_fns)      |   Directly generates upper/outer bilevel program derivative functions.    |
|                 [implicit_hessian(k_fn, z, params, Dg, Hg, jvp_vec, optimizations)](/sensitivity_torch/api/sensitivity_torch/batch_sensitivity/implicit_hessian)                 | Computes the implicit Hessian or chain rule depending on Dg, Hg, jvp_vec. |
| [implicit_jacobian(k_fn, z, params, Dg, jvp_vec, matrix_free_inverse, full_output, optimizations)](/sensitivity_torch/api/sensitivity_torch/batch_sensitivity/implicit_jacobian) |  Computes the implicit Jacobian or VJP or JVP depending on Dg, jvp_vec.   |


# `differentiation`

|                                                        name                                                         |                                 summary                                 |
|---------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
|      [BATCH_HESSIAN(fn, args, config)](/sensitivity_torch/api/sensitivity_torch/differentiation/BATCH_HESSIAN)      | Computes the Hessian, assuming the first in/out dimension is the batch. |
| [BATCH_HESSIAN_DIAG(fn, args, config)](/sensitivity_torch/api/sensitivity_torch/differentiation/BATCH_HESSIAN_DIAG) |    Evaluates per-argument partial batch (first dimension) Hessians.     |
|     [BATCH_JACOBIAN(fn, args, config)](/sensitivity_torch/api/sensitivity_torch/differentiation/BATCH_JACOBIAN)     | Computes the Hessian, assuming the first in/out dimension is the batch. |
|           [HESSIAN(fn, inputs, config)](/sensitivity_torch/api/sensitivity_torch/differentiation/HESSIAN)           |             Equivalent to torch.autograd.functional.hessian             |
|       [HESSIAN_DIAG(fn, args, config)](/sensitivity_torch/api/sensitivity_torch/differentiation/HESSIAN_DIAG)       |   Generates a function which computes per-argument partial Hessians.    |
|          [JACOBIAN(fn, inputs, config)](/sensitivity_torch/api/sensitivity_torch/differentiation/JACOBIAN)          |            Equivalent to torch.autograd.functional.jacobian             |


# `extras.optimization`

|                                                                                                                           name                                                                                                                            |                                                            summary                                                            |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| [minimize_agd(f_fn, g_fn, args, verbose, verbose_prefix, max_it, ai, af, batched, full_output, callback_fn, use_writer, use_tqdm, optimizer, optimizer_state, optimizer_opts)](/sensitivity_torch/api/sensitivity_torch/extras/optimization/minimize_agd) |      Minimize a loss function ``f_fn`` with Accelerated Gradient Descent (AGD) with respect to ``*args``. Uses PyTorch.       |
|                       [minimize_lbfgs(f_fn, g_fn, args, verbose, verbose_prefix, lr, max_it, batched, full_output, callback_fn, use_writer, use_tqdm)](/sensitivity_torch/api/sensitivity_torch/extras/optimization/minimize_lbfgs)                       |                 Minimize a loss function ``f_fn`` with L-BFGS with respect to ``*args``. Taken from PyTorch.                  |
|         [minimize_sqp(f_fn, g_fn, h_fn, args, reg0, verbose, verbose_prefix, max_it, ls_pts_nb, force_step, batched, full_output, callback_fn, use_writer, use_tqdm)](/sensitivity_torch/api/sensitivity_torch/extras/optimization/minimize_sqp)          | Minimize a loss function ``f_fn`` with Unconstrained Sequential Quadratic Programming (SQP) with respect to a single ``arg``. |


# `sensitivity`

|                                                                                    name                                                                                    |                                  summary                                  |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
|     [generate_optimization_fns(loss_fn, opt_fn, k_fn, normalize_grad, optimizations)](/sensitivity_torch/api/sensitivity_torch/sensitivity/generate_optimization_fns)      |   Directly generates upper/outer bilevel program derivative functions.    |
|                 [implicit_hessian(k_fn, z, params, Dg, Hg, jvp_vec, optimizations)](/sensitivity_torch/api/sensitivity_torch/sensitivity/implicit_hessian)                 | Computes the implicit Hessian or chain rule depending on Dg, Hg, jvp_vec. |
| [implicit_jacobian(k_fn, z, params, Dg, jvp_vec, matrix_free_inverse, full_output, optimizations)](/sensitivity_torch/api/sensitivity_torch/sensitivity/implicit_jacobian) |  Computes the implicit Jacobian or VJP or JVP depending on Dg, jvp_vec.   |
