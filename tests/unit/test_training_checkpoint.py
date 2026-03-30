"""Unit tests for training-state serialisation helpers."""

from unittest.mock import MagicMock

import jax.numpy as jnp

import eigen3.utils.training_checkpoint as tc
from eigen3.agents.trading_agent import TradingNetworkParams
from eigen3.workflows.trading_workflow import ReplayBufferState


def test_replay_buffer_roundtrip():
    buf = ReplayBufferState(
        obs=jnp.zeros((10, 2, 3, 1), dtype=jnp.float32),
        actions=jnp.zeros((10, 4, 3), dtype=jnp.float32),
        rewards=jnp.zeros(10, dtype=jnp.float32),
        next_obs=jnp.zeros((10, 2, 3, 1), dtype=jnp.float32),
        dones=jnp.zeros(10, dtype=jnp.float32),
        size=jnp.array(3, dtype=jnp.int32),
        insert_idx=jnp.array(7, dtype=jnp.int32),
    )
    d = tc._replay_to_checkpoint(buf)
    buf2 = tc._replay_from_checkpoint(d)
    assert int(buf2.size) == 3
    assert int(buf2.insert_idx) == 7
    assert buf2.obs.shape == buf.obs.shape


def test_fresh_replay_buffer_for_workflow():
    wf = MagicMock()
    wf.env.obs_space.shape = (5, 2, 1)
    wf.env.action_space.shape = (3, 2)
    wf.config.replay_buffer_size = 128
    buf = tc.fresh_replay_buffer_for_workflow(wf)
    assert int(buf.size) == 0
    assert buf.obs.shape == (128, 5, 2, 1)


def test_stacked_params_roundtrip():
    p = TradingNetworkParams(
        actor_params={"w": jnp.ones((2, 3))},
        critic_params={"w": jnp.zeros((1,))},
        target_actor_params={"w": jnp.ones((2, 3))},
        target_critic_params={"w": jnp.zeros((1,))},
    )
    d = tc._stacked_params_to_checkpoint(p)
    p2 = tc.stacked_params_from_checkpoint(d)
    assert jnp.allclose(p2.actor_params["w"], p.actor_params["w"])
