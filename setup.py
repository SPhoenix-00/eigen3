"""Setup script for Eigen3 - JAX-based ERL Stock Trading System"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eigen3",
    version="0.1.0",
    author="Eigen Team",
    description="JAX-based Evolutionary Reinforcement Learning for Stock Trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*", "scripts"]),
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.20",
        "jaxlib>=0.4.20",
        "flax>=0.8.0",
        "optax>=0.1.9",
        "chex>=0.1.85",
        "jaxtyping>=0.2.25",
        "gymnasium>=0.29.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "wandb>=0.16.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
