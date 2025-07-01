#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from typing import List

import torch
from tensordict import TensorDictBase, TensorDict

from benchmarl.experiment.callback import Callback
from tpo.models.tpo import HetControlMlpEmpirical
from tpo.snd import compute_behavioral_distance
from tpo.utils import overflowing_logits_norm
import os
import json
import torch
from typing import List, Dict, Any

# ... existing get_het_model and other callbacks ...

class DiversityDataCollectorCallback(Callback):
    """
    Callback to collect data for training the Adaptive Diversity Manager.
    Collects (SystemState, SND_input, ObservedDiversityPattern, AchievedPerformance).
    """

    def __init__(self, enable_data_collection: bool = False, collection_interval_frames: int = 10000, save_path: str = "diversity_data.json"):
        super().__init__()
        self.enable_data_collection = enable_data_collection
        self.collection_interval_frames = collection_interval_frames
        self.save_path = save_path
        self.collected_data: List[Dict[str, Any]] = []
        self._last_collection_frame = 0

    def on_setup(self):
        if self.enable_data_collection:
            print(f"Diversity data collection enabled. Saving to: {self.save_path}")
            self._last_collection_frame = self.experiment.total_frames
        else:
            print("Diversity data collection disabled.")

    def on_batch_collected(self, batch: TensorDictBase):
        if not self.enable_data_collection:
            return

        if self.experiment.total_frames - self._last_collection_frame >= self.collection_interval_frames:
            self._last_collection_frame = self.experiment.total_frames

            # Assuming 'agents' is the relevant group, adapt if different
            group = list(self.experiment.group_map.keys())[0] # Take the first group
            if group not in self.experiment.group_policies:
                return # Skip if policy not available for this group

            policy = self.experiment.group_policies[group]
            model = get_het_model(policy)

            # --- SystemState ---
            # For simplicity, using a flattened observation as SystemState.
            # You might need to refine this based on your environment and specific metrics.
            # Example: mean agent positions, variance of agent velocities, task progress
            observations = batch.get((group, "observation"))
            system_state = observations.mean(dim=-2).flatten() # Average over agents, then flatten batch dims for single state

            # --- SND_input (desired_snd) ---
            snd_input = model.desired_snd.item()

            # --- ObservedDiversityPattern (estimated_snd) ---
            # Ensure estimated_snd is up-to-date from the policy's forward pass
            # You could also explicitly call model.estimate_snd(observations) here if needed
            observed_diversity = model.estimated_snd.item()

            # --- AchievedPerformance (e.g., sum of rewards) ---
            # Sum of rewards for the collected batch, assuming it's episodic or average
            # Adjust key based on your reward structure
            achieved_performance = batch.get(("next", group, "reward")).sum().item()

            self.collected_data.append({
                "frame": self.experiment.total_frames,
                "system_state": system_state.cpu().numpy().tolist(), # Convert to list for JSON
                "snd_input": snd_input,
                "observed_diversity_pattern": observed_diversity,
                "achieved_performance": achieved_performance,
            })
            print(f"Collected data at frame {self.experiment.total_frames}")

    def on_experiment_end(self):
        if self.enable_data_collection:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(self.collected_data, f, indent=4)
            print(f"Saved {len(self.collected_data)} data points to {self.save_path}")
            
def get_het_model(policy):
    model = policy.module[0]
    while not isinstance(model, HetControlMlpEmpirical):
        model = model[0]
    return model


class SndCallback(Callback):
    """
    Callback used to compute SND during evaluations
    """

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        for group in self.experiment.group_map.keys():
            if not len(self.experiment.group_map[group]) > 1:
                # If agent group has 1 agent
                continue
            policy = self.experiment.group_policies[group]
            # Cat observations over time
            obs = torch.cat(
                [rollout.select((group, "observation")) for rollout in rollouts], dim=0
            )  # tensor of shape [*batch_size, n_agents, n_features]
            model = get_het_model(policy)
            agent_actions = []
            # Compute actions that each agent would take in this obs
            for i in range(model.n_agents):
                agent_actions.append(
                    model._forward(obs, agent_index=i, compute_estimate=False).get(
                        model.out_key
                    )
                )
            # Compute SND
            distance = compute_behavioral_distance(agent_actions, just_mean=True)
            self.experiment.logger.log(
                {f"eval/{group}/snd": distance.mean().item()},
                step=self.experiment.n_iters_performed,
            )


class NormLoggerCallback(Callback):
    """
    Callback to log some training metrics
    """

    def on_batch_collected(self, batch: TensorDictBase):
        for group in self.experiment.group_map.keys():
            keys_to_norm = [
                (group, "f"),
                (group, "g"),
                (group, "fdivg"),
                (group, "logits"),
                (group, "observation"),
                (group, "out_loc_norm"),
                (group, "estimated_snd"),
                (group, "scaling_ratio"),
            ]
            to_log = {}

            for key in keys_to_norm:
                value = batch.get(key, None)
                if value is not None:
                    to_log.update(
                        {"/".join(("collection",) + key): torch.mean(value).item()}
                    )
            self.experiment.logger.log(
                to_log,
                step=self.experiment.n_iters_performed,
            )


class TagCurriculum(Callback):
    """
    Tag curriculum used to freeze the green agents' policies during training
    """

    def __init__(self, simple_tag_freeze_policy_after_frames, simple_tag_freeze_policy):
        super().__init__()
        self.n_frames_train = simple_tag_freeze_policy_after_frames
        self.simple_tag_freeze_policy = simple_tag_freeze_policy
        self.activated = not simple_tag_freeze_policy

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            simple_tag_freeze_policy_after_frames=self.n_frames_train,
            simple_tag_freeze_policy=self.simple_tag_freeze_policy,
        )
        # Make agent group homogeneous
        policy = self.experiment.group_policies["agents"]
        model = get_het_model(policy)
        # Set the desired SND of the green agent team to 0
        # This is not important as the green agent team is composed of 1 agent
        model.desired_snd[:] = 0

    def on_batch_collected(self, batch: TensorDictBase):
        if (
            self.experiment.total_frames >= self.n_frames_train
            and not self.activated
            and self.simple_tag_freeze_policy
        ):
            del self.experiment.train_group_map["agents"]
            self.activated = True


class ActionSpaceLoss(Callback):
    """
    Loss to disincentivize actions outside of the space
    """

    def __init__(self, use_action_loss, action_loss_lr):
        super().__init__()
        self.opt_dict = {}
        self.use_action_loss = use_action_loss
        self.action_loss_lr = action_loss_lr

    def on_setup(self):
        # Log params
        self.experiment.logger.log_hparams(
            use_action_loss=self.use_action_loss, action_loss_lr=self.action_loss_lr
        )

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        if not self.use_action_loss:
            return
        policy = self.experiment.group_policies[group]
        model = get_het_model(policy)
        if group not in self.opt_dict:
            self.opt_dict[group] = torch.optim.Adam(
                model.parameters(), lr=self.action_loss_lr
            )
        opt = self.opt_dict[group]
        loss = self.action_space_loss(group, model, batch)
        loss_td = TensorDict({"loss_action_space": loss}, [])

        loss.backward()

        grad_norm = self.experiment._grad_clip(opt)
        loss_td.set(
            f"grad_norm_action_space",
            torch.tensor(grad_norm, device=self.experiment.config.train_device),
        )

        opt.step()
        opt.zero_grad()

        return loss_td

    def action_space_loss(self, group, model, batch):
        logits = model._forward(
            batch.select(*model.in_keys), compute_estimate=True, update_estimate=False
        ).get(
            model.out_key
        )  # Compute logits from batch
        if model.probabilistic:
            logits, _ = torch.chunk(logits, 2, dim=-1)
        out_loc_norm = overflowing_logits_norm(
            logits, self.experiment.action_spec[group, "action"]
        )  # Compute how much they overflow outside the action space bounds

        # Penalise the maximum overflow over the agents
        max_overflowing_logits_norm = out_loc_norm.max(dim=-1)[0]

        loss = max_overflowing_logits_norm.pow(2).mean()
        return loss