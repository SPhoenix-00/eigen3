"""Example unit test to verify test setup"""

import pytest
import jax
import jax.numpy as jnp


def test_jax_installation():
    """Test that JAX is properly installed"""
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    assert y == 6.0


def test_jax_random():
    """Test JAX random number generation"""
    key = jax.random.PRNGKey(42)
    random_array = jax.random.normal(key, (10,))
    assert random_array.shape == (10,)


def test_jax_grad():
    """Test JAX automatic differentiation"""

    def f(x):
        return jnp.sum(x**2)

    x = jnp.array([1.0, 2.0, 3.0])
    grad_f = jax.grad(f)
    gradient = grad_f(x)

    # df/dx = 2x
    expected = jnp.array([2.0, 4.0, 6.0])
    assert jnp.allclose(gradient, expected)


@pytest.mark.slow
def test_jax_jit():
    """Test JAX JIT compilation"""

    @jax.jit
    def fast_function(x):
        return jnp.sum(x**2)

    x = jnp.ones((1000, 1000))
    result = fast_function(x)
    assert result.shape == ()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
