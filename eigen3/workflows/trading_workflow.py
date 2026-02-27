"""Custom ERL workflow for stock trading with DDPG agents

This module implements a custom evolutionary reinforcement learning workflow
specifically designed for the stock trading environment. It combines:
- DDPG-based agents for continuous action spaces
- Evolutionary algorithms for population-based training
- Custom genetic operators for neural network parameters
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import chex
import jax
import jax.numpy as jnp
import jax.random as random
from flax.core.frozen_dict import FrozenDict

from evorl.workflows import Workflow
from evorl.types import PyTreeData
from evorl.utils.sample_batch import SampleBatch
from evorl.envs import Env
from evorl.agents import Agent
from evorl.evaluators import Evaluator

from evorl.agent import AgentState
from eigen3.agents import TradingNetworkParams


@dataclass
class TradingWorkflowConfig:
    """Configuration for the trading ERL workflow

    Args:
        population_size: Number of agents in the population
        elite_size: Number of top agents to preserve each generation
        tournament_size: Number of agents to sample for tournament selection
        mutation_rate: Probability of mutation for each parameter
        mutation_std: Standard deviation of Gaussian mutation
        crossover_rate: Probability of crossover between parents
        gradient_steps_per_gen: Number of gradient updates per generation
        batch_size: Batch size for gradient updates
        replay_buffer_size: Maximum size of replay buffer
        warmup_steps: Number of random exploration steps before training
        eval_episodes: Number of episodes to evaluate each agent
        target_update_period: How often to update target networks (in gradient steps)
    """
    population_size: int = 10
    elite_size: int = 2
    tournament_size: int = 3
    mutation_rate: float = 0.1
    mutation_std: float = 0.02
    crossover_rate: float = 0.5
    gradient_steps_per_gen: int = 100
    batch_size: int = 32
    replay_buffer_size: int = 100000
    warmup_steps: int = 1000
    eval_episodes: int = 5
    target_update_period: int = 10


class TradingERLWorkflow(Workflow):
    """Custom ERL workflow for stock trading

    This workflow implements a hybrid evolutionary-gradient approach:
    1. Evolutionary phase: Use genetic operators to explore parameter space
    2. Gradient phase: Refine each agent using DDPG updates
    3. Evaluation phase: Assess fitness of each agent
    4. Selection phase: Keep elite agents and breed new population

    The workflow is designed to leverage both the exploration capabilities
    of evolutionary algorithms and the sample efficiency of gradient-based RL.
    """

    def __init__(
        self,
        env: Env,
        agent: Agent,
        evaluator: Evaluator,
        config: TradingWorkflowConfig,
        seed: int = 0,
    ):
        """Initialize the trading ERL workflow

        Args:
            env: Trading environment implementing Env interface
            agent: DDPG trading agent implementing Agent interface
            evaluator: Evaluator for assessing agent fitness
            config: Workflow configuration
            seed: Random seed for reproducibility
        """
        super().__init__(env=env, agent=agent, evaluator=evaluator)

        self.config = config
        self.seed = seed
        self.key = random.PRNGKey(seed)

        # Initialize population
        self.population = None
        self.replay_buffer = None
        self.generation = 0
        self.total_env_steps = 0

    def _initialize_population(self, key: chex.PRNGKey) -> List[TradingNetworkParams]:
        """Initialize a population of agents with random parameters

        Args:
            key: JAX random key

        Returns:
            List of initialized network parameters
        """
        population = []

        for i in range(self.config.population_size):
            key, subkey = random.split(key)

            # Initialize agent via the EvoRL Agent interface
            agent_state = self.agent.init(
                obs_space=self.env.obs_space,
                action_space=self.env.action_space,
                key=subkey,
            )
            agent_params = agent_state.params

            population.append(agent_params)

        return population

    def _tournament_selection(
        self,
        population: List[TradingNetworkParams],
        fitness_scores: chex.Array,
        key: chex.PRNGKey,
    ) -> TradingNetworkParams:
        """Select an agent using tournament selection

        Args:
            population: List of agent parameters
            fitness_scores: Fitness score for each agent
            key: JAX random key

        Returns:
            Selected agent parameters
        """
        # Sample random indices for tournament
        indices = random.choice(
            key,
            jnp.arange(len(population)),
            shape=(self.config.tournament_size,),
            replace=False,
        )

        # Find best agent in tournament
        tournament_fitness = fitness_scores[indices]
        winner_idx = indices[jnp.argmax(tournament_fitness)]

        return population[int(winner_idx)]

    def _crossover(
        self,
        parent1: TradingNetworkParams,
        parent2: TradingNetworkParams,
        key: chex.PRNGKey,
    ) -> TradingNetworkParams:
        """Perform uniform crossover between two parents

        Args:
            parent1: First parent parameters
            parent2: Second parent parameters
            key: JAX random key

        Returns:
            Child parameters
        """
        def crossover_pytree(p1, p2, key):
            """Crossover operation for PyTree leaves"""
            # Generate random mask for each parameter
            mask = random.uniform(key, p1.shape) < self.config.crossover_rate
            # Uniform crossover: randomly take from either parent
            return jnp.where(mask, p1, p2)

        # Split keys for each parameter
        keys = random.split(key, 4)

        # Perform crossover on actor and critic parameters
        child_actor = jax.tree_map(
            lambda p1, p2: crossover_pytree(p1, p2, keys[0]),
            parent1.actor_params,
            parent2.actor_params,
        )

        child_critic = jax.tree_map(
            lambda p1, p2: crossover_pytree(p1, p2, keys[1]),
            parent1.critic_params,
            parent2.critic_params,
        )

        # Copy target networks from parent1 (will be updated during training)
        child_actor_target = parent1.target_actor_params
        child_critic_target = parent1.target_critic_params

        return TradingNetworkParams(
            actor_params=child_actor,
            critic_params=child_critic,
            actor_target_params=child_actor_target,
            critic_target_params=child_critic_target,
        )

    def _mutate(
        self,
        params: TradingNetworkParams,
        key: chex.PRNGKey,
    ) -> TradingNetworkParams:
        """Apply Gaussian mutation to parameters

        Args:
            params: Agent parameters to mutate
            key: JAX random key

        Returns:
            Mutated parameters
        """
        def mutate_pytree(p, key):
            """Mutation operation for PyTree leaves"""
            # Generate random mask for mutation
            mask = random.uniform(key, p.shape) < self.config.mutation_rate
            # Generate Gaussian noise
            noise = random.normal(key, p.shape) * self.config.mutation_std
            # Apply mutation where mask is True
            return jnp.where(mask, p + noise, p)

        # Split keys for each parameter
        keys = random.split(key, 2)

        # Mutate actor and critic parameters
        mutated_actor = jax.tree_map(
            lambda p: mutate_pytree(p, keys[0]),
            params.actor_params,
        )

        mutated_critic = jax.tree_map(
            lambda p: mutate_pytree(p, keys[1]),
            params.critic_params,
        )

        return TradingNetworkParams(
            actor_params=mutated_actor,
            critic_params=mutated_critic,
            actor_target_params=params.actor_target_params,
            critic_target_params=params.critic_target_params,
        )

    def _collect_experience(
        self,
        agent_params: TradingNetworkParams,
        num_steps: int,
        key: chex.PRNGKey,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Collect experience by running agent in environment

        Args:
            agent_params: Agent parameters to use
            num_steps: Number of steps to collect
            key: JAX random key

        Returns:
            Tuple of (list of transitions, cumulative reward)
        """
        transitions = []
        cumulative_reward = 0.0

        # Reset environment
        key, reset_key = random.split(key)
        env_state = self.env.reset(reset_key)

        for step in range(num_steps):
            # Get action from agent
            key, action_key = random.split(key)
            obs = env_state.obs

            # Compute action (with exploration noise during warmup/training)
            actions, policy_info = self.agent.compute_actions(
                agent_state=AgentState(params=agent_params),
                sample_batch=SampleBatch(obs=obs[None, ...]),
                key=action_key,
            )
            action = actions[0]  # Remove batch dim

            # Step environment
            next_env_state = self.env.step(env_state, action)

            # Store transition
            transition = {
                'obs': obs,
                'action': action,
                'reward': next_env_state.reward,
                'next_obs': next_env_state.obs,
                'done': next_env_state.done,
            }
            transitions.append(transition)

            cumulative_reward += float(next_env_state.reward)

            # Check if episode is done
            if next_env_state.done:
                key, reset_key = random.split(key)
                next_env_state = self.env.reset(reset_key)

            env_state = next_env_state

        return transitions, cumulative_reward

    def _gradient_update(
        self,
        agent_params: TradingNetworkParams,
        replay_buffer: List[Dict[str, Any]],
        key: chex.PRNGKey,
    ) -> Tuple[TradingNetworkParams, Dict[str, float]]:
        """Perform gradient updates on agent using replay buffer

        Args:
            agent_params: Current agent parameters
            replay_buffer: List of transitions
            key: JAX random key

        Returns:
            Tuple of (updated parameters, metrics)
        """
        if len(replay_buffer) < self.config.batch_size:
            # Not enough data yet
            return agent_params, {}

        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for step in range(self.config.gradient_steps_per_gen):
            # Sample batch from replay buffer
            key, sample_key = random.split(key)
            indices = random.choice(
                sample_key,
                len(replay_buffer),
                shape=(self.config.batch_size,),
                replace=False,
            )

            # Create batch
            batch = {
                'obs': jnp.stack([replay_buffer[i]['obs'] for i in indices]),
                'action': jnp.stack([replay_buffer[i]['action'] for i in indices]),
                'reward': jnp.stack([replay_buffer[i]['reward'] for i in indices]),
                'next_obs': jnp.stack([replay_buffer[i]['next_obs'] for i in indices]),
                'done': jnp.stack([replay_buffer[i]['done'] for i in indices]),
            }

            # Compute losses and gradients
            key, loss_key = random.split(key)
            sample_batch = SampleBatch(
                obs=batch['obs'],
                actions=batch['action'],
                rewards=batch['reward'],
                next_obs=batch['next_obs'],
                dones=batch['done'],
            )
            losses = self.agent.loss(
                agent_state=AgentState(params=agent_params),
                sample_batch=sample_batch,
                key=loss_key,
            )

            # Extract losses (simplified - in practice would use optax optimizers)
            total_actor_loss += float(losses['actor_loss'])
            total_critic_loss += float(losses['critic_loss'])

            # Update target networks periodically
            if step % self.config.target_update_period == 0:
                from eigen3.agents import soft_target_update
                agent_params = soft_target_update(
                    agent_params,
                    tau=self.agent.tau,
                )

        metrics = {
            'actor_loss': total_actor_loss / self.config.gradient_steps_per_gen,
            'critic_loss': total_critic_loss / self.config.gradient_steps_per_gen,
        }

        return agent_params, metrics

    def _evaluate_agent(
        self,
        agent_params: TradingNetworkParams,
        key: chex.PRNGKey,
    ) -> float:
        """Evaluate agent fitness over multiple episodes

        Args:
            agent_params: Agent parameters to evaluate
            key: JAX random key

        Returns:
            Average cumulative reward (fitness)
        """
        total_reward = 0.0

        for episode in range(self.config.eval_episodes):
            # Reset environment
            key, reset_key = random.split(key)
            env_state = self.env.reset(reset_key)

            episode_reward = 0.0
            done = False
            max_steps = 1000  # Safety limit
            step = 0

            while not done and step < max_steps:
                # Get deterministic action (no exploration noise)
                key, action_key = random.split(key)
                actions, _ = self.agent.evaluate_actions(
                    agent_state=AgentState(params=agent_params),
                    sample_batch=SampleBatch(obs=env_state.obs[None, ...]),
                    key=action_key,
                )
                action = actions[0]

                # Step environment
                env_state = self.env.step(env_state, action)
                episode_reward += float(env_state.reward)
                done = bool(env_state.done)
                step += 1

            total_reward += episode_reward

        return total_reward / self.config.eval_episodes

    def run_generation(self) -> Dict[str, Any]:
        """Run one generation of the ERL workflow

        Returns:
            Dictionary of metrics and statistics
        """
        self.key, gen_key = random.split(self.key)

        # Initialize population on first generation
        if self.population is None:
            self.key, init_key = random.split(self.key)
            self.population = self._initialize_population(init_key)
            self.replay_buffer = []

        # Phase 1: Collect experience and gradient updates for each agent
        updated_population = []
        all_metrics = []

        for idx, agent_params in enumerate(self.population):
            # Collect experience
            self.key, collect_key = random.split(self.key)
            transitions, ep_reward = self._collect_experience(
                agent_params=agent_params,
                num_steps=100,  # Steps per agent per generation
                key=collect_key,
            )

            # Add to replay buffer
            self.replay_buffer.extend(transitions)

            # Trim replay buffer if too large
            if len(self.replay_buffer) > self.config.replay_buffer_size:
                self.replay_buffer = self.replay_buffer[-self.config.replay_buffer_size:]

            # Gradient updates
            self.key, update_key = random.split(self.key)
            updated_params, metrics = self._gradient_update(
                agent_params=agent_params,
                replay_buffer=self.replay_buffer,
                key=update_key,
            )

            updated_population.append(updated_params)
            all_metrics.append(metrics)

            self.total_env_steps += 100

        self.population = updated_population

        # Phase 2: Evaluate fitness of all agents
        fitness_scores = []
        for agent_params in self.population:
            self.key, eval_key = random.split(self.key)
            fitness = self._evaluate_agent(agent_params, eval_key)
            fitness_scores.append(fitness)

        fitness_scores = jnp.array(fitness_scores)

        # Phase 3: Selection and breeding
        # Keep elite agents
        elite_indices = jnp.argsort(fitness_scores)[-self.config.elite_size:]
        elite_agents = [self.population[int(i)] for i in elite_indices]

        # Breed new agents to fill population
        new_population = elite_agents.copy()

        while len(new_population) < self.config.population_size:
            # Tournament selection for parents
            self.key, key1, key2, key3 = random.split(self.key, 4)

            parent1 = self._tournament_selection(self.population, fitness_scores, key1)
            parent2 = self._tournament_selection(self.population, fitness_scores, key2)

            # Crossover
            child = self._crossover(parent1, parent2, key3)

            # Mutation
            self.key, mutate_key = random.split(self.key)
            child = self._mutate(child, mutate_key)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

        # Return metrics
        return {
            'generation': self.generation,
            'mean_fitness': float(jnp.mean(fitness_scores)),
            'max_fitness': float(jnp.max(fitness_scores)),
            'min_fitness': float(jnp.min(fitness_scores)),
            'std_fitness': float(jnp.std(fitness_scores)),
            'total_env_steps': self.total_env_steps,
            'mean_actor_loss': float(jnp.mean(jnp.array([m.get('actor_loss', 0.0) for m in all_metrics]))),
            'mean_critic_loss': float(jnp.mean(jnp.array([m.get('critic_loss', 0.0) for m in all_metrics]))),
        }

    def train(self, num_generations: int) -> List[Dict[str, Any]]:
        """Train for multiple generations

        Args:
            num_generations: Number of generations to run

        Returns:
            List of metrics for each generation
        """
        all_metrics = []

        for gen in range(num_generations):
            metrics = self.run_generation()
            all_metrics.append(metrics)

            # Log progress
            print(f"Generation {gen + 1}/{num_generations}")
            print(f"  Mean Fitness: {metrics['mean_fitness']:.2f}")
            print(f"  Max Fitness:  {metrics['max_fitness']:.2f}")
            print(f"  Total Steps:  {metrics['total_env_steps']}")
            print()

        return all_metrics

    def get_best_agent(self) -> TradingNetworkParams:
        """Get the best agent from current population

        Returns:
            Parameters of the best agent
        """
        if self.population is None:
            raise ValueError("Population not initialized. Run at least one generation first.")

        # Evaluate all agents
        fitness_scores = []
        for agent_params in self.population:
            self.key, eval_key = random.split(self.key)
            fitness = self._evaluate_agent(agent_params, eval_key)
            fitness_scores.append(fitness)

        fitness_scores = jnp.array(fitness_scores)
        best_idx = int(jnp.argmax(fitness_scores))

        return self.population[best_idx]


def create_trading_workflow(
    env: Env,
    agent: Agent,
    evaluator: Evaluator,
    config: Optional[TradingWorkflowConfig] = None,
    seed: int = 0,
) -> TradingERLWorkflow:
    """Convenience function to create a trading ERL workflow

    Args:
        env: Trading environment
        agent: DDPG trading agent
        evaluator: Evaluator for agent assessment
        config: Workflow configuration (uses defaults if None)
        seed: Random seed

    Returns:
        Initialized TradingERLWorkflow
    """
    if config is None:
        config = TradingWorkflowConfig()

    return TradingERLWorkflow(
        env=env,
        agent=agent,
        evaluator=evaluator,
        config=config,
        seed=seed,
    )
