# sensitivity_torch

``sensitivity_torch`` is a package designed to allow taking first- and
**second-order** derivatives through optimization or any other fixed-point
process.

This package builds on top of [PyTorch](https://pytorch.org/). We also
maintain an implementation in [JAX](https://github.com/google/jax)
[here](https://rdyro.github.io/sensitivity_jax/).

## Documentation

[Documentation can be found here.](https://rdyro.github.io/sensitivity_torch/)

## Installation

Install using pip
```bash
$ pip install git+https://github.com/rdyro/sensitivity_torch.git
```
or from source
```bash
$ git clone git@github.com:rdyro/sensitivity_torch.git
$ cd sensitivity_torch
$ python3 setup.py install --user
```

### Testing

Run all unit tests using
```bash
$ python3 setup.py test
```
