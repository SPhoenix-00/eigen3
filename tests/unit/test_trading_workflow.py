"""Unit tests for TradingERLWorkflow (GPU-vectorized)"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from unittest.mock import MagicMock
from eigen3.workflows import TradingERLWorkflow, TradingWorkflowConfig, create_trading_workflow
from eigen3.workflows.trading_workflow import (
    stack_params,
    unstack_params,
    create_replay_buffer,
    buffer_insert_batch,
    buffer_sample,
)
from eigen3.environment import TradingEnv
from eigen3.agents import TradingAgent
from eigen3.models import Actor, DoubleCritic
from evorl.agent import AgentState
from evorl.sample_batch import SampleBatch


def create_test_data(num_days=1000, num_columns=669):
    """Create test data for trading environment"""
    key = random.PRNGKey(42)

    # Base prices around 100
    prices = 100.0 + random.normal(key, (num_days, num_columns)) * 10

    # Create observation data (5 features)
    data_array = jnp.ones((num_days, num_columns, 5))
    data_array = data_array.at[:, :, 0].set(prices)

    # Create full data (9 features)
    data_array_full = jnp.ones((num_days, num_columns, 9))
    data_array_full = data_array_full.at[:, :, 1].set(prices)
    data_array_full = data_array_full.at[:, :, 2].set(prices * 1.02)
    data_array_full = data_array_full.at[:, :, 0].set(prices * 0.99)
    data_array_full = data_array_full.at[:, :, 3].set(prices * 0.98)

    norm_stats = {
        'mean': jnp.zeros(5),
        'std': jnp.ones(5)
    }

    return data_array, data_array_full, norm_stats


def create_test_workflow(pop_size=4):
    """Create a test workflow with small population"""
    data_array, data_array_full, norm_stats = create_test_data()

    env = TradingEnv(data_array, data_array_full, norm_stats)
    nc = int(data_array.shape[1])
    agent = TradingAgent(
        actor_network=Actor(num_columns=nc),
        critic_network=DoubleCritic(num_columns=nc),
        exploration_noise=0.1,
    )
    evaluator = MagicMock()

    config = TradingWorkflowConfig(
        population_size=pop_size,
        elite_size=1,
        tournament_size=2,
        mutation_rate=0.1,
        mutation_std=0.01,
        crossover_rate=0.5,
        gradient_steps_per_gen=5,
        batch_size=16,
        replay_buffer_size=1000,
        val_episodes=2,
        steps_per_agent=10,
    )

    workflow = TradingERLWorkflow(
        env=env,
        agent=agent,
        evaluator=evaluator,
        config=config,
        seed=42,
    )

    return workflow


class TestWorkflowInitialization:
    """Test workflow initialization"""

    def test_create_workflow(self):
        """Test creating workflow"""
        workflow = create_test_workflow()

        assert workflow.config.population_size == 4
        assert workflow.generation == 0
        assert workflow.total_env_steps == 0
        assert workflow._stacked_params is None

    def test_create_workflow_convenience(self):
        """Test convenience creation function"""
        data_array, data_array_full, norm_stats = create_test_data()

        env = TradingEnv(data_array, data_array_full, norm_stats)
        nc = int(data_array.shape[1])
        agent = TradingAgent(
            actor_network=Actor(num_columns=nc),
            critic_network=DoubleCritic(num_columns=nc),
        )
        evaluator = MagicMock()

        workflow = create_trading_workflow(env, agent, evaluator, seed=42)

        assert isinstance(workflow, TradingERLWorkflow)
        assert workflow.config.population_size == 10  # Default


class TestPopulationManagement:
    """Test population initialization and management"""

    def test_initialize_population(self):
        """Test population initialization produces stacked params"""
        workflow = create_test_workflow(pop_size=3)

        key = random.PRNGKey(0)
        workflow._initialize_population(key)

        stacked = workflow._stacked_params
        assert stacked is not None

        # Leading dim of each leaf should be pop_size
        leaves = jax.tree.leaves(stacked)
        for leaf in leaves:
            assert leaf.shape[0] == 3

    def test_population_initialized_on_first_generation(self):
        """Test that population is initialized on first generation"""
        workflow = create_test_workflow(pop_size=2)

        assert workflow._stacked_params is None

        metrics = workflow.run_generation()

        assert workflow._stacked_params is not None
        # Verify leading dim = pop_size on a sample leaf
        some_leaf = jax.tree.leaves(workflow._stacked_params)[0]
        assert some_leaf.shape[0] == 2


class TestStackUnstack:
    """Test stack_params / unstack_params utilities"""

    def test_roundtrip(self):
        """stack then unstack should recover original params"""
        workflow = create_test_workflow(pop_size=3)
        key = random.PRNGKey(0)
        keys = random.split(key, 3)
        population = [
            workflow.agent.init(workflow.env.obs_space, workflow.env.action_space, k).params
            for k in keys
        ]

        stacked = stack_params(population)
        recovered = unstack_params(stacked, 3)

        for i in range(3):
            orig_leaves = jax.tree.leaves(population[i])
            rec_leaves = jax.tree.leaves(recovered[i])
            for o, r in zip(orig_leaves, rec_leaves):
                assert jnp.array_equal(o, r)


class TestReplayBuffer:
    """Test JAX replay buffer operations"""

    def test_create_buffer(self):
        buf = create_replay_buffer(100, (5,), (3,))
        assert buf.obs.shape == (100, 5)
        assert buf.actions.shape == (100, 3)
        assert int(buf.size) == 0

    def test_insert_and_sample(self):
        buf = create_replay_buffer(100, (5,), (3,))
        key = random.PRNGKey(0)

        obs = random.normal(key, (10, 5))
        acts = random.normal(key, (10, 3))
        rews = jnp.ones(10)
        nobs = random.normal(key, (10, 5))
        dones = jnp.zeros(10)

        buf = buffer_insert_batch(buf, obs, acts, rews, nobs, dones)
        assert int(buf.size) == 10
        assert int(buf.insert_idx) == 10

        batch = buffer_sample(buf, key, 4)
        assert batch.obs.shape == (4, 5)
        assert batch.actions.shape == (4, 3)

    def test_ring_wrap(self):
        buf = create_replay_buffer(10, (2,), (1,))
        key = random.PRNGKey(0)

        for i in range(3):
            obs = jnp.ones((5, 2)) * i
            buf = buffer_insert_batch(
                buf, obs, jnp.zeros((5, 1)),
                jnp.zeros(5), obs, jnp.zeros(5),
            )

        # 15 inserts into capacity-10 buffer → size capped at 10
        assert int(buf.size) == 10
        assert int(buf.insert_idx) == 5  # 15 % 10


class TestGeneticOperators:
    """Test genetic algorithm operators"""

    def test_tournament_selection(self):
        """Test tournament selection returns valid index"""
        workflow = create_test_workflow(pop_size=5)
        fitness = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])

        key = random.PRNGKey(1)
        idx = workflow._tournament_select(fitness, key)
        assert 0 <= idx < 5

    def test_crossover(self):
        """Test crossover operator"""
        workflow = create_test_workflow()

        key = random.PRNGKey(0)
        k1, k2, cross_key = random.split(key, 3)

        p1 = workflow.agent.init(workflow.env.obs_space, workflow.env.action_space, k1).params
        p2 = workflow.agent.init(workflow.env.obs_space, workflow.env.action_space, k2).params

        child = workflow._crossover_single(p1, p2, cross_key)

        assert hasattr(child, 'actor_params')
        assert hasattr(child, 'critic_params')
        assert child.actor_params is not None

    def test_mutation(self):
        """Test mutation operator"""
        workflow = create_test_workflow()

        key = random.PRNGKey(0)
        params = workflow.agent.init(workflow.env.obs_space, workflow.env.action_space, key).params
        original_actor_params = params.actor_params

        key = random.PRNGKey(1)
        mutated = workflow._mutate_single(params, key)

        assert hasattr(mutated, 'actor_params')
        assert hasattr(mutated, 'critic_params')

        def any_changed(original, mutated):
            diffs = jax.tree.map(lambda o, m: jnp.any(o != m), original, mutated)
            return any(jax.tree.leaves(diffs))

        assert any_changed(original_actor_params, mutated.actor_params)

    def test_mutation_rate(self):
        """Test mutation respects mutation rate"""
        workflow = create_test_workflow()
        workflow.config.mutation_rate = 1.0

        key = random.PRNGKey(0)
        params = workflow.agent.init(workflow.env.obs_space, workflow.env.action_space, key).params

        key = random.PRNGKey(1)
        mutated = workflow._mutate_single(params, key)

        diffs = jax.tree.map(
            lambda o, m: jnp.any(o != m),
            params.actor_params,
            mutated.actor_params,
        )
        assert any(jax.tree.leaves(diffs))


class TestGenerationExecution:
    """Test running generations"""

    def test_run_single_generation(self):
        """Test running a single generation"""
        workflow = create_test_workflow(pop_size=2)

        metrics = workflow.run_generation()

        assert 'generation' in metrics
        assert 'mean_fitness' in metrics
        assert 'max_fitness' in metrics
        assert 'min_fitness' in metrics
        assert 'std_fitness' in metrics
        assert 'total_env_steps' in metrics

        assert metrics['generation'] == 1
        assert workflow.generation == 1
        assert workflow._stacked_params is not None
        # Daily-alpha mode: keep legacy key for compatibility plus explicit alpha alias.
        assert "top5_bh_excess_usd" in metrics
        assert "top5_alpha_sum_usd" in metrics
        # Terminal BNH reward rewrite controls were removed in daily-alpha mode.
        assert "bnh_terminal_clamp_active" not in metrics

    def test_run_multiple_generations(self):
        """Test running multiple generations"""
        workflow = create_test_workflow(pop_size=2)

        all_metrics = workflow.train(num_generations=3)

        assert len(all_metrics) == 3
        assert all_metrics[0]['generation'] == 1
        assert all_metrics[1]['generation'] == 2
        assert all_metrics[2]['generation'] == 3
        assert all_metrics[2]['total_env_steps'] > all_metrics[0]['total_env_steps']

    def test_elite_preservation(self):
        """Test that population size is preserved after breeding"""
        workflow = create_test_workflow(pop_size=4)
        workflow.config.elite_size = 2

        workflow.run_generation()

        some_leaf = jax.tree.leaves(workflow._stacked_params)[0]
        assert some_leaf.shape[0] == 4

    def test_replay_buffer_management(self):
        """Test replay buffer growth"""
        workflow = create_test_workflow(pop_size=2)
        workflow.config.replay_buffer_size = 100

        workflow.run_generation()

        assert workflow._replay_buffer is not None
        assert int(workflow._replay_buffer.size) > 0
        assert int(workflow._replay_buffer.size) <= workflow.config.replay_buffer_size

class TestBestAgentRetrieval:
    """Test retrieving the best agent"""

    def test_get_best_agent(self):
        """Test getting best agent from population"""
        workflow = create_test_workflow(pop_size=3)

        with pytest.raises(ValueError, match="Population not initialized"):
            workflow.get_best_agent()

        workflow.run_generation()

        best_agent = workflow.get_best_agent()

        assert hasattr(best_agent, 'actor_params')
        assert hasattr(best_agent, 'critic_params')


class TestWorkflowDeterminism:
    """Test workflow determinism with seeds"""

    def test_deterministic_with_same_seed(self):
        """Test that same seed produces same results"""
        workflow1 = create_test_workflow(pop_size=2)
        workflow2 = create_test_workflow(pop_size=2)

        assert workflow1.seed == workflow2.seed

        metrics1 = workflow1.run_generation()
        metrics2 = workflow2.run_generation()

        assert abs(metrics1['mean_fitness'] - metrics2['mean_fitness']) < 1e-5

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)
        nc = int(data_array.shape[1])
        agent = TradingAgent(
            actor_network=Actor(num_columns=nc),
            critic_network=DoubleCritic(num_columns=nc),
        )
        evaluator = MagicMock()
        config = TradingWorkflowConfig(
            population_size=2,
            steps_per_agent=10,
            gradient_steps_per_gen=5,
            batch_size=16,
            replay_buffer_size=1000,
            val_episodes=2,
        )

        workflow1 = TradingERLWorkflow(env, agent, evaluator, config, seed=0)
        workflow2 = TradingERLWorkflow(env, agent, evaluator, config, seed=999)

        metrics1 = workflow1.run_generation()
        metrics2 = workflow2.run_generation()

        assert abs(metrics1['mean_fitness'] - metrics2['mean_fitness']) > 1e-10 or \
               abs(metrics1['max_fitness'] - metrics2['max_fitness']) > 1e-10


@pytest.mark.slow
class TestFullWorkflow:
    """Test complete workflow execution"""

    def test_full_training_run(self):
        """Test running complete training workflow"""
        workflow = create_test_workflow(pop_size=3)

        all_metrics = workflow.train(num_generations=2)

        assert len(all_metrics) == 2

        best_agent = workflow.get_best_agent()
        assert best_agent is not None

        key = random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, *workflow.env.obs_space.shape))
        actions, _ = workflow.agent.evaluate_actions(
            agent_state=AgentState(params=best_agent),
            sample_batch=SampleBatch(obs=dummy_obs),
            key=key,
        )
        assert actions.shape == (1, 108, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
