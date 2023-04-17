from pathlib import Path
from setuptools import setup


setup(
    name="sensitivity_torch",
    version="0.3.1",
    author="Robert Dyro",
    description=("Optimization Sensitivity Analysis for Bilevel Programming for PyTorch"),
    license="MIT",
    packages=["sensitivity_torch", "sensitivity_torch.extras"],
    install_requires=["torch", "tensorboard", "numpy", "scipy", "tqdm", "matplotlib"],
    long_description=(Path(__file__).parent / "README.md").read_text(),
)
