sensitivity\_torch package
==========================

Public API
==========

.. toctree::
   :maxdepth: 1
   :caption: Public API

.. currentmodule:: sensitivity_torch

Sensitivity Analysis (:code:`sensitivity`)
------------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  sensitivity.implicit_jacobian
  sensitivity.implicit_hessian
  sensitivity.generate_optimization_fns

Differentiation (:code:`differentiation`)
-----------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  differentiation.JACOBIAN
  differentiation.HESSIAN
  differentiation.HESSIAN_DIAG

Extras: Optimization (:code:`extras.optimization`)
--------------------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  extras.optimization.minimize_agd
  extras.optimization.minimize_lbfgs
  extras.optimization.minimize_sqp

Extras: Neural Network Tools (:code:`extras.nn_tools`)
----------------------------------------------------------

.. autosummary::
  :nosignatures:
  :toctree: _autosummary

  extras.nn_tools.nn_all_params
  extras.nn_tools.nn_forward_gen

..  Subpackages
    -----------

    .. toctree::
       :maxdepth: 4

       sensitivity_torch.extras

    Submodules
    ----------

    .. toctree::
       :maxdepth: 4

       sensitivity_torch.differentiation
       sensitivity_torch.sensitivity
       sensitivity_torch.specialized_matrix_inverse
       sensitivity_torch.utils

    Module contents
    ---------------

    .. automodule:: sensitivity_torch
       :members:
       :undoc-members:
       :show-inheritance:

