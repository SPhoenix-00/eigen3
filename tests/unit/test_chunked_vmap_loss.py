"""Population-axis chunking for loss vmap (VRAM cap)."""

import jax.numpy as jnp

from eigen3.workflows.trading_workflow import run_chunked_vmap_loss


def test_run_chunked_vmap_loss_matches_full():
    pop_size = 6
    stacked = {"w": jnp.arange(pop_size * 2, dtype=jnp.float32).reshape(pop_size, 2)}
    batch = None  # unused by fake loss
    loss_keys = jnp.zeros((pop_size, 2), dtype=jnp.uint32)

    def vmap_loss(sub_p, _batch, _k):
        n = sub_p["w"].shape[0]
        s = jnp.sum(sub_p["w"], axis=1)
        return {
            "actor_loss": s,
            "critic_loss": s * 0.5,
            "mean_q": s * 0.25,
        }

    full = run_chunked_vmap_loss(vmap_loss, stacked, batch, loss_keys, pop_size, None)
    chunked = run_chunked_vmap_loss(vmap_loss, stacked, batch, loss_keys, pop_size, 2)
    assert jnp.allclose(full["actor_loss"], chunked["actor_loss"])
    assert jnp.allclose(full["critic_loss"], chunked["critic_loss"])
    assert jnp.allclose(full["mean_q"], chunked["mean_q"])


def test_run_chunked_vmap_loss_uneven_last_chunk():
    pop_size = 5
    stacked = {"w": jnp.arange(pop_size * 3, dtype=jnp.float32).reshape(pop_size, 3)}
    loss_keys = jnp.zeros((pop_size, 2), dtype=jnp.uint32)

    def vmap_loss(sub_p, _batch, _k):
        s = jnp.sum(sub_p["w"], axis=1)
        return {"actor_loss": s, "critic_loss": s, "mean_q": s}

    full = run_chunked_vmap_loss(vmap_loss, stacked, None, loss_keys, pop_size, None)
    chunked = run_chunked_vmap_loss(vmap_loss, stacked, None, loss_keys, pop_size, 2)
    assert jnp.allclose(full["actor_loss"], chunked["actor_loss"])
