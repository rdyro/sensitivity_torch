import os
from setuptools import setup

# borrowed from https://pythonhosted.org/an_example_pypi_project/setuptools.html


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except FileNotFoundError:
        return ""


setup(
    name="sensitivity_torch",
    version="0.3.0",
    author="Robert Dyro",
    description=(
        "Optimization Sensitivity Analysis for Bilevel Programming for Torch"
    ),
    license="MIT",
    packages=["sensitivity_torch", "sensitivity_torch.extras"],
    #install_requires=["torch", "numpy", "tqdm"],
    long_description=read("README.md"),
)
