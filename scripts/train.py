"""Training script for Eigen3 trading system"""

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
from pathlib import Path
import logging

# These imports will be available after implementation
# from eigen3.workflows.trading_erl_workflow import TradingERLWorkflow
# from eigen3.data.loader import load_trading_data
# from eigen3.utils.logging import setup_logger
# from eigen3.utils.checkpointing import save_checkpoint

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function"""

    # Print configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set random seed
    logger.info(f"Setting random seed: {cfg.seed}")
    key = jax.random.PRNGKey(cfg.seed)

    # Load data
    logger.info(f"Loading data from: {cfg.env.data_path}")
    # data_array, data_array_full, norm_stats = load_trading_data(cfg.env.data_path)

    # Create environment
    logger.info("Creating trading environment...")
    # env = create_trading_env(cfg.env, data_array, data_array_full, norm_stats)

    # Create workflow
    logger.info(f"Creating workflow: {cfg.agent.workflow_cls}")
    # workflow_cls = hydra.utils.get_class(cfg.agent.workflow_cls)
    # workflow = workflow_cls.build_from_config(
    #     cfg.agent,
    #     env=env,
    #     enable_jit=cfg.enable_jit,
    #     enable_pmap=cfg.enable_pmap,
    # )

    # Initialize workflow state
    logger.info("Initializing workflow state...")
    # state = workflow.init(key)

    # Training loop
    logger.info("Starting training...")
    # for generation in range(cfg.agent.total_generations):
    #     # Training step
    #     metrics, state = workflow.step(state)
    #
    #     # Log metrics
    #     if generation % cfg.logging.log_interval == 0:
    #         logger.info(f"Generation {generation}: {metrics}")
    #
    #     # Save checkpoint
    #     if generation % cfg.agent.save_interval == 0:
    #         checkpoint_path = Path(cfg.checkpoint_dir) / f"gen_{generation}.pkl"
    #         save_checkpoint(checkpoint_path, state, cfg)
    #         logger.info(f"Saved checkpoint: {checkpoint_path}")
    #
    #     # Validation
    #     if generation % cfg.agent.eval_interval == 0:
    #         eval_metrics = workflow.evaluate(state)
    #         logger.info(f"Validation metrics: {eval_metrics}")

    # Final save
    # final_checkpoint = Path(cfg.checkpoint_dir) / "final_model.pkl"
    # save_checkpoint(final_checkpoint, state, cfg)
    # logger.info(f"Training complete! Final model saved to: {final_checkpoint}")

    logger.info("Training script template - implementation pending")


if __name__ == "__main__":
    main()
