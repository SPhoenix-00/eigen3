"""Hydra → TradingWorkflowConfig mapping."""

from omegaconf import OmegaConf

from eigen3.entrypoints.training import build_trading_workflow_config


def test_build_trading_workflow_config_elite_and_population():
    cfg = OmegaConf.create(
        {
            "population": {
                "pop_size": 10,
                "elite_frac": 0.4,
                "tournament_size": 3,
                "mutation_rate": 0.2,
                "mutation_std": 0.03,
                "genetic_crossover_rate": 0.6,
                "gradient_steps_per_gen": 8,
                "batch_size": 32,
                "replay_buffer_size": 10000,
                "eval_episodes": 3,
                "steps_per_agent": 50,
            },
            "agent": {"actor_update_interval": 4},
        }
    )
    wfc = build_trading_workflow_config(cfg)
    assert wfc.population_size == 10
    assert wfc.elite_size == 4  # round(10 * 0.4)
    assert wfc.crossover_rate == 0.6
    assert wfc.target_update_period == 4
    assert wfc.steps_per_agent == 50
    assert wfc.batch_size == 32


def test_tournament_size_clamped_to_population():
    cfg = OmegaConf.create(
        {
            "population": {
                "pop_size": 2,
                "elite_frac": 0.5,
                "tournament_size": 5,
                "mutation_rate": 0.2,
                "mutation_std": 0.02,
                "genetic_crossover_rate": 0.5,
                "gradient_steps_per_gen": 1,
                "batch_size": 4,
                "replay_buffer_size": 1000,
                "eval_episodes": 1,
                "steps_per_agent": 5,
            },
            "agent": {"actor_update_interval": 2},
        }
    )
    wfc = build_trading_workflow_config(cfg)
    assert wfc.population_size == 2
    assert wfc.tournament_size == 2


def test_single_gpu_uses_local_batch_and_replay():
    cfg = OmegaConf.create(
        {
            "enable_pmap": False,
            "num_devices": 1,
            "population": {
                "pop_size": 8,
                "elite_frac": 0.25,
                "tournament_size": 2,
                "mutation_rate": 0.1,
                "mutation_std": 0.02,
                "genetic_crossover_rate": 0.5,
                "gradient_steps_per_gen": 2,
                "batch_size": 160,
                "local_batch_size": 32,
                "replay_buffer_size": 1_000_000,
                "local_replay_buffer_size": 5000,
                "eval_episodes": 1,
                "steps_per_agent": 10,
            },
            "agent": {"actor_update_interval": 2},
        }
    )
    wfc = build_trading_workflow_config(cfg)
    assert wfc.batch_size == 32
    assert wfc.replay_buffer_size == 5000


def test_multi_device_uses_global_batch_and_replay():
    cfg = OmegaConf.create(
        {
            "enable_pmap": False,
            "num_devices": 2,
            "population": {
                "pop_size": 8,
                "elite_frac": 0.25,
                "tournament_size": 2,
                "mutation_rate": 0.1,
                "mutation_std": 0.02,
                "genetic_crossover_rate": 0.5,
                "gradient_steps_per_gen": 2,
                "batch_size": 160,
                "local_batch_size": 32,
                "replay_buffer_size": 900_000,
                "local_replay_buffer_size": 5000,
                "eval_episodes": 1,
                "steps_per_agent": 10,
            },
            "agent": {"actor_update_interval": 2},
        }
    )
    wfc = build_trading_workflow_config(cfg)
    assert wfc.batch_size == 160
    assert wfc.replay_buffer_size == 900_000
