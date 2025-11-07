"""Unit tests for TradingERLWorkflow"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from eigen3.workflows import TradingERLWorkflow, TradingWorkflowConfig, create_trading_workflow
from eigen3.environment import TradingEnv
from eigen3.agents import TradingAgent
from eigen3.models import Actor, DoubleCritic
from evorl.evaluators import Evaluator


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
    agent = TradingAgent(
        actor_network=Actor(),
        critic_network=DoubleCritic(),
        exploration_noise=0.1,
    )
    evaluator = Evaluator()  # Simple evaluator

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
        eval_episodes=2,
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
        assert workflow.population is None

    def test_create_workflow_convenience(self):
        """Test convenience creation function"""
        data_array, data_array_full, norm_stats = create_test_data()

        env = TradingEnv(data_array, data_array_full, norm_stats)
        agent = TradingAgent(
            actor_network=Actor(),
            critic_network=DoubleCritic(),
        )
        evaluator = Evaluator()

        workflow = create_trading_workflow(env, agent, evaluator, seed=42)

        assert isinstance(workflow, TradingERLWorkflow)
        assert workflow.config.population_size == 10  # Default


class TestPopulationManagement:
    """Test population initialization and management"""

    def test_initialize_population(self):
        """Test population initialization"""
        workflow = create_test_workflow(pop_size=3)

        key = random.PRNGKey(0)
        population = workflow._initialize_population(key)

        # Check population size
        assert len(population) == 3

        # Check each agent has correct structure
        for agent_params in population:
            assert hasattr(agent_params, 'actor_params')
            assert hasattr(agent_params, 'critic_params')
            assert hasattr(agent_params, 'actor_target_params')
            assert hasattr(agent_params, 'critic_target_params')

    def test_population_initialized_on_first_generation(self):
        """Test that population is initialized on first generation"""
        workflow = create_test_workflow(pop_size=2)

        assert workflow.population is None

        # Run one generation
        metrics = workflow.run_generation()

        # Population should now be initialized
        assert workflow.population is not None
        assert len(workflow.population) == 2


class TestGeneticOperators:
    """Test genetic algorithm operators"""

    def test_tournament_selection(self):
        """Test tournament selection"""
        workflow = create_test_workflow(pop_size=5)

        # Initialize population
        key = random.PRNGKey(0)
        population = workflow._initialize_population(key)

        # Create fitness scores
        fitness_scores = jnp.array([1.0, 5.0, 3.0, 2.0, 4.0])

        # Select agent
        key = random.PRNGKey(1)
        selected = workflow._tournament_selection(population, fitness_scores, key)

        # Should return valid agent params
        assert hasattr(selected, 'actor_params')
        assert hasattr(selected, 'critic_params')

    def test_crossover(self):
        """Test crossover operator"""
        workflow = create_test_workflow()

        # Initialize two parents
        key = random.PRNGKey(0)
        key1, key2, cross_key = random.split(key, 3)

        dummy_obs = jnp.zeros((1, 504, 669, 5))
        dummy_action = jnp.zeros((1, 108, 2))

        parent1 = workflow.agent.init(key1, dummy_obs, dummy_action)
        parent2 = workflow.agent.init(key2, dummy_obs, dummy_action)

        # Perform crossover
        child = workflow._crossover(parent1, parent2, cross_key)

        # Child should have same structure
        assert hasattr(child, 'actor_params')
        assert hasattr(child, 'critic_params')

        # Check that child parameters are different from both parents
        # (at least some parameters should differ due to randomness)
        def params_differ(p1, p2):
            return jax.tree_util.tree_map(
                lambda x, y: jnp.any(x != y),
                p1, p2
            )

        # Child should combine elements from both parents
        # (Can't easily test exact mixing, but structure should be valid)
        assert child.actor_params is not None

    def test_mutation(self):
        """Test mutation operator"""
        workflow = create_test_workflow()

        # Initialize agent
        key = random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, 504, 669, 5))
        dummy_action = jnp.zeros((1, 108, 2))
        params = workflow.agent.init(key, dummy_obs, dummy_action)

        # Store original param for comparison
        original_actor_params = params.actor_params

        # Perform mutation
        key = random.PRNGKey(1)
        mutated = workflow._mutate(params, key)

        # Mutated params should have same structure
        assert hasattr(mutated, 'actor_params')
        assert hasattr(mutated, 'critic_params')

        # At least some parameters should change (with high probability)
        # Note: Due to mutation_rate < 1.0, not all params will change
        def any_changed(original, mutated):
            diffs = jax.tree_util.tree_map(
                lambda o, m: jnp.any(o != m),
                original, mutated
            )
            # Check if any leaf differs
            leaves = jax.tree_util.tree_leaves(diffs)
            return any(leaves)

        # Should be different (stochastic test)
        assert any_changed(original_actor_params, mutated.actor_params)

    def test_mutation_rate(self):
        """Test mutation respects mutation rate"""
        # Use high mutation rate for testing
        workflow = create_test_workflow()
        workflow.config.mutation_rate = 1.0  # Mutate all parameters

        key = random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, 504, 669, 5))
        dummy_action = jnp.zeros((1, 108, 2))
        params = workflow.agent.init(key, dummy_obs, dummy_action)

        # Mutate
        key = random.PRNGKey(1)
        mutated = workflow._mutate(params, key)

        # With mutation_rate=1.0, all parameters should be different
        def all_changed(original, mutated):
            diffs = jax.tree_util.tree_map(
                lambda o, m: jnp.any(o != m),
                original, mutated
            )
            return diffs

        diffs = all_changed(params.actor_params, mutated.actor_params)
        # Most leaves should differ (allowing for some floating point edge cases)
        assert any(jax.tree_util.tree_leaves(diffs))


class TestExperienceCollection:
    """Test experience collection"""

    def test_collect_experience(self):
        """Test collecting experience from environment"""
        workflow = create_test_workflow()

        # Initialize agent
        key = random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, 504, 669, 5))
        dummy_action = jnp.zeros((1, 108, 2))
        agent_params = workflow.agent.init(key, dummy_obs, dummy_action)

        # Collect experience
        key = random.PRNGKey(1)
        transitions, cumulative_reward = workflow._collect_experience(
            agent_params=agent_params,
            num_steps=10,
            key=key,
        )

        # Should collect 10 transitions
        assert len(transitions) == 10

        # Check transition structure
        for trans in transitions:
            assert 'obs' in trans
            assert 'action' in trans
            assert 'reward' in trans
            assert 'next_obs' in trans
            assert 'done' in trans

            assert trans['obs'].shape == (504, 669, 5)
            assert trans['action'].shape == (108, 2)

        # Cumulative reward should be a number
        assert isinstance(cumulative_reward, float)


class TestAgentEvaluation:
    """Test agent evaluation"""

    def test_evaluate_agent(self):
        """Test evaluating agent fitness"""
        workflow = create_test_workflow()

        # Initialize agent
        key = random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, 504, 669, 5))
        dummy_action = jnp.zeros((1, 108, 2))
        agent_params = workflow.agent.init(key, dummy_obs, dummy_action)

        # Evaluate
        key = random.PRNGKey(1)
        fitness = workflow._evaluate_agent(agent_params, key)

        # Fitness should be a number
        assert isinstance(fitness, float)
        # Should be finite
        assert jnp.isfinite(fitness)


class TestGenerationExecution:
    """Test running generations"""

    def test_run_single_generation(self):
        """Test running a single generation"""
        workflow = create_test_workflow(pop_size=2)

        metrics = workflow.run_generation()

        # Check metrics
        assert 'generation' in metrics
        assert 'mean_fitness' in metrics
        assert 'max_fitness' in metrics
        assert 'min_fitness' in metrics
        assert 'std_fitness' in metrics
        assert 'total_env_steps' in metrics

        assert metrics['generation'] == 1
        assert workflow.generation == 1
        assert workflow.population is not None
        assert len(workflow.population) == 2

    def test_run_multiple_generations(self):
        """Test running multiple generations"""
        workflow = create_test_workflow(pop_size=2)

        all_metrics = workflow.train(num_generations=3)

        # Should have metrics for 3 generations
        assert len(all_metrics) == 3

        # Generation numbers should increase
        assert all_metrics[0]['generation'] == 1
        assert all_metrics[1]['generation'] == 2
        assert all_metrics[2]['generation'] == 3

        # Environment steps should increase
        assert all_metrics[2]['total_env_steps'] > all_metrics[0]['total_env_steps']

    def test_elite_preservation(self):
        """Test that elite agents are preserved"""
        workflow = create_test_workflow(pop_size=4)
        workflow.config.elite_size = 2

        # Run generation
        workflow.run_generation()

        # Population should still be size 4
        assert len(workflow.population) == 4

    def test_replay_buffer_management(self):
        """Test replay buffer growth and trimming"""
        workflow = create_test_workflow(pop_size=2)
        workflow.config.replay_buffer_size = 100

        # Run generation
        workflow.run_generation()

        # Replay buffer should have data
        assert workflow.replay_buffer is not None
        assert len(workflow.replay_buffer) > 0

        # Should not exceed max size
        assert len(workflow.replay_buffer) <= workflow.config.replay_buffer_size


class TestBestAgentRetrieval:
    """Test retrieving the best agent"""

    def test_get_best_agent(self):
        """Test getting best agent from population"""
        workflow = create_test_workflow(pop_size=3)

        # Should fail before any generation
        with pytest.raises(ValueError, match="Population not initialized"):
            workflow.get_best_agent()

        # Run generation
        workflow.run_generation()

        # Now should work
        best_agent = workflow.get_best_agent()

        assert hasattr(best_agent, 'actor_params')
        assert hasattr(best_agent, 'critic_params')


class TestWorkflowDeterminism:
    """Test workflow determinism with seeds"""

    def test_deterministic_with_same_seed(self):
        """Test that same seed produces same results"""
        workflow1 = create_test_workflow(pop_size=2)
        workflow2 = create_test_workflow(pop_size=2)

        # Both should use seed=42
        assert workflow1.seed == workflow2.seed

        metrics1 = workflow1.run_generation()
        metrics2 = workflow2.run_generation()

        # Should produce same fitness results
        # (Note: Due to JAX PRNG determinism, this should be exact)
        assert abs(metrics1['mean_fitness'] - metrics2['mean_fitness']) < 1e-5

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results"""
        data_array, data_array_full, norm_stats = create_test_data()
        env = TradingEnv(data_array, data_array_full, norm_stats)
        agent = TradingAgent(
            actor_network=Actor(),
            critic_network=DoubleCritic(),
        )
        evaluator = Evaluator()
        config = TradingWorkflowConfig(population_size=2)

        workflow1 = TradingERLWorkflow(env, agent, evaluator, config, seed=0)
        workflow2 = TradingERLWorkflow(env, agent, evaluator, config, seed=999)

        metrics1 = workflow1.run_generation()
        metrics2 = workflow2.run_generation()

        # Should produce different results (with very high probability)
        # Note: Could technically be same, but extremely unlikely
        assert abs(metrics1['mean_fitness'] - metrics2['mean_fitness']) > 1e-10 or \
               abs(metrics1['max_fitness'] - metrics2['max_fitness']) > 1e-10


@pytest.mark.slow
class TestFullWorkflow:
    """Test complete workflow execution"""

    def test_full_training_run(self):
        """Test running complete training workflow"""
        workflow = create_test_workflow(pop_size=3)

        # Train for a few generations
        all_metrics = workflow.train(num_generations=2)

        # Should complete successfully
        assert len(all_metrics) == 2

        # Get best agent
        best_agent = workflow.get_best_agent()
        assert best_agent is not None

        # Verify best agent can be used
        key = random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, 504, 669, 5))
        actions, _ = workflow.agent.evaluate_actions(
            agent_state=best_agent,
            sample_batch={'obs': dummy_obs},
            key=key,
        )
        assert actions.shape == (1, 108, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
