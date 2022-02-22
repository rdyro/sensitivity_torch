Tour of ``sensitivity_torch``
===========================

A simple, exhaustive, example of model tuning using sensitivity analysis:::

  import torch, numpy as np
  from sklearn.linear_model import LogisticRegression
  from sklearn.datasets import make_moons

  from sensitivity_torch.sensitivity import (
      implicit_jacobian,
      generate_optimization_fns,
  )
  from sensitivity_torch.extras.optimization import (
      minimize_agd,
      minimize_lbfgs,
      minimize_sqp,
  )


  def feature_map(X, P, q):
      Phi = torch.cos(X @ P + q)
      return Phi


  def optimize_fn(P, q, X, Y):
      """Non-differentiable optimization function using sklearn."""
      Phi = feature_map(X, P, q)
      model = LogisticRegression(max_iter=300, C=1e-2)
      model.fit(Phi, Y)
      z = np.concatenate([model.coef_.reshape(-1), model.intercept_])
      return torch.tensor(z)


  def loss_fn(z, P, q, X, Y):
      """The measure of model performance."""
      Phi = feature_map(X, P, q)
      W, b = z[:-1], z[-1]
      prob = torch.sigmoid(Phi @ W + b)
      loss = torch.sum(-(Y * torch.log(prob) + (1 - Y) * torch.log(1 - prob)))
      regularization = 0.5 * torch.sum(W ** 2) / 1e-2
      return (loss + regularization) / X.shape[-2]


  def k_fn(z, P, q, X, Y):
      """Optimalty conditions of the model – the fixed-point of optimization."""
      return torch.autograd.functional.jacobian(
          lambda z: loss_fn(z, P, q, X, Y), z, create_graph=True
      )


  if __name__ == "__main__":
      ######################## SETUP ####################################
      # generate data and split into train and test sets
      X, Y = make_moons(n_samples=1000, noise=1e-1)
      X, Y = torch.tensor(X), torch.tensor(Y)
      n_train = 200
      Xtr, Ytr = X[:n_train, :], Y[:n_train]
      Xts, Yts = X[n_train:, :], Y[n_train:]

      # generate the Fourier feature map parameters Phi = cos(X @ P + q)
      n_features = 200
      P = torch.tensor(np.random.randn(2, n_features))
      q = torch.tensor(np.zeros(n_features))

      # check that the fixed-point is numerically close to zero
      z = optimize_fn(P, q, X, Y)
      k = k_fn(z, P, q, X, Y)
      assert torch.max(torch.abs(k)) < 1e-5

      ######################## SENSITIVITY ##############################
      # generate sensitivity Jacobians and check their shape
      JP, Jq = implicit_jacobian(lambda z, P, q: k_fn(z, P, q, X, Y), z, P, q)
      assert JP.shape == z.shape + P.shape
      assert Jq.shape == z.shape + q.shape

      ######################## OPTIMIZATION #############################
      # generate necessary functions for optimization
      opt_fn_ = lambda P: optimize_fn(P, q, Xtr, Ytr)  # model optimization
      k_fn_ = lambda z, P: k_fn(z, P, q, Xtr, Ytr)  # fixed-point
      loss_fn_ = lambda z, P: loss_fn(z, P, q, Xts, Yts)  # loss to improve
      f_fn, g_fn, h_fn = generate_optimization_fns(loss_fn_, opt_fn_, k_fn_)

      # choose any optimization routine
      # Ps = minimize_agd(f_fn, g_fn, P, verbose=True, ai=1e-1, af=1e-1, max_it=100)
      # Ps = minimize_lbfgs(f_fn, g_fn, P, verbose=True, lr=1e-1, max_it=10)
      Ps = minimize_sqp(f_fn, g_fn, h_fn, P, verbose=True, max_it=10)

      def predict(z, P, q, X):
          Phi = feature_map(X, P, q)
          W, b = z[:-1], z[-1]
          prob = torch.sigmoid(Phi @ W + b)
          return torch.round(prob)

      # evaluate the results
      acc0 = torch.mean(1.0 * (predict(opt_fn_(P), P, q, Xts) == Yts))
      accf = torch.mean(1.0 * (predict(opt_fn_(Ps), Ps, q, Xts) == Yts))
      print("Accuracy before: %4.2f%%" % (1e2 * acc0))
      print("Accuracy after:  %4.2f%%" % (1e2 * accf))
      print("Loss before:     %9.4e" % loss_fn(opt_fn_(P), P, q, Xts, Yts))
      print("Loss after:      %9.4e" % loss_fn(opt_fn_(Ps), Ps, q, Xts, Yts))
