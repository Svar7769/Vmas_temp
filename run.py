#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import benchmarl.models
from benchmarl.algorithms import *
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment
from benchmarl.hydra_config import (
    load_algorithm_config_from_hydra,
    load_experiment_config_from_hydra,
    load_task_config_from_hydra,
    load_model_config_from_hydra,
)
from tpo.callback import *
from tpo.environments.vmas import render_callback
from tpo.models.tpo import HetControlMlpEmpiricalConfig


def setup(task_name):
    benchmarl.models.model_config_registry.update(
        {
            "hetcontrolmlpempirical": HetControlMlpEmpiricalConfig,
        }
    )
    if task_name == "vmas/sampling":
        # Set the render callback for the navigatio case study
        VmasTask.render_callback = render_callback


# In /home/svarp/Desktop/Projects/Vmas_temp/run.py

def get_experiment(cfg: DictConfig) -> Experiment:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    # --- Start of fix for task_name format ---
    # Convert task_name from config name (e.g., "vmas_sampling_config")
    # to BenchMARL's expected format (e.g., "vmas/sampling").
    if isinstance(task_name, str) and "_" in task_name and "/" not in task_name:
        parts = task_name.split("_")
        if len(parts) >= 2 and parts[0] == "vmas": # Assuming 'vmas' is the environment prefix
            environment_prefix = parts[0]
            inner_task_part = "_".join(parts[1:])
            # Remove '_config' suffix if present
            if inner_task_part.endswith("_config"):
                inner_task_part = inner_task_part[:-len("_config")]
            task_name = f"{environment_prefix}/{inner_task_part}"
    # --- End of fix for task_name format ---

    setup(task_name) # This now uses the potentially modified task_name

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}") # This will now print the corrected format
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    algorithm_config = load_algorithm_config_from_hydra(cfg.algorithm)
    experiment_config = load_experiment_config_from_hydra(cfg.experiment)
    # The second argument to load_task_config_from_hydra now uses the correctly formatted task_name
    task_config = load_task_config_from_hydra(cfg.task, task_name)
    critic_model_config = load_model_config_from_hydra(cfg.critic_model)
    model_config = load_model_config_from_hydra(cfg.model)

    if isinstance(algorithm_config, (MappoConfig, IppoConfig, MasacConfig, IsacConfig)):
        model_config.probabilistic = True
        model_config.scale_mapping = algorithm_config.scale_mapping
        algorithm_config.scale_mapping = (
            "relu"  # The scaling of std_dev will be done in the model
        )
    else:
        model_config.probabilistic = False

    
    experiment_callbacks = [
        SndCallback(),
        NormLoggerCallback(),
        ActionSpaceLoss(
            use_action_loss=cfg.use_action_loss, action_loss_lr=cfg.action_loss_lr
        ),
        DiversityDataCollectorCallback( # Add the new callback
            enable_data_collection=cfg.enable_diversity_data_collection,
            collection_interval_frames=cfg.diversity_data_collection_interval,
            save_path=f"{cfg.experiment.save_folder}/diversity_data.json" # Or a path defined in config
        ),
    ] + (
        [
            TagCurriculum(
                cfg.simple_tag_freeze_policy_after_frames,
                cfg.simple_tag_freeze_policy,
            )
        ]
        if task_name == "vmas/sampling" # This condition will now correctly check "vmas/sampling"
        else []
    )

    experiment = Experiment(
        task=task_config,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=cfg.seed,
        config=experiment_config,
        callbacks=experiment_callbacks, # Use the modified list
    )
    return experiment

@hydra.main(version_base=None, config_path="tpo/conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    experiment = get_experiment(cfg=cfg)
    experiment.run()


if __name__ == "__main__":
    hydra_experiment()