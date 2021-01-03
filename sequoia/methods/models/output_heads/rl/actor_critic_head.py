""" An output head for RL based on Advantage Actor Critic.

NOTE: This is the 'online' version of an Advantage Actor Critic, based
on the following blog:

https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import torch
from gym import spaces
from gym.spaces.utils import flatdim
from sequoia.common import Loss
from sequoia.common.layers import Flatten, Lambda
from sequoia.settings import ContinualRLSetting
from sequoia.settings.base.objects import Actions, Observations, Rewards
from sequoia.utils import get_logger
from sequoia.utils.generic_functions import get_slice
from sequoia.utils.utils import prod
from torch import LongTensor, Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from ...forward_pass import ForwardPass
from ..classification_head import ClassificationHead, ClassificationOutput
from .policy_head import Categorical, PolicyHead, PolicyHeadOutput

logger = get_logger(__file__)


@dataclass(frozen=True)
class A2CHeadOutput(PolicyHeadOutput):
    """ Output produced by the A2C output head. """
    # The value estimate coming from the critic.
    value: Tensor

    @classmethod
    def stack(cls, items: List["A2CHeadOutput"]) -> "A2CHeadOutput":
        """TODO: Add a classmethod to 'stack' these objects. """


class ActorCriticHead(PolicyHead):
    
    @dataclass
    class HParams(PolicyHead.HParams):
        """ Hyper-parameters of the Actor-Critic head. """
        gamma: float = 0.95


        actor_loss_coef: float = 0.5
        critic_loss_coef: float = 0.5
        entropy_loss_coef: float = 0.1

    def __init__(self,
                 input_space: spaces.Space,
                 action_space: spaces.Discrete,
                 reward_space: spaces.Box,
                 hparams: "ActorCriticHead.HParams" = None,
                 name: str = "actor_critic"):
        assert isinstance(action_space, spaces.Discrete), "Only support discrete space for now."
        super().__init__(
            input_space=input_space,
            action_space=action_space,
            reward_space=reward_space,
            hparams=hparams,
            name=name,
        )
        if not isinstance(self.hparams, self.HParams):
            self.hparams = self.upgrade_hparams()
            
        action_dims = flatdim(action_space)

        # Critic takes in state-action pairs? or just state?
        self.critic_input_dims = self.input_size
        # self.critic_input_dims = self.input_size + action_dims
        self.critic_output_dims = 1
        self.critic = nn.Sequential(
            # Lambda(concat_obs_and_action),
            Flatten(),
            nn.Linear(self.critic_input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, self.critic_output_dims),
        )
        self.actor_input_dims = self.input_size
        self.actor_output_dims = action_dims
        self.actor = nn.Sequential(
            Flatten(),
            nn.Linear(self.actor_input_dims, 32),
            nn.ReLU(),
            nn.Linear(32, self.actor_output_dims),
        )        
        self._current_state: Optional[Tensor] = None
        self._previous_state: Optional[Tensor] = None
        self._step = 0

    def forward(self,
                observations: ContinualRLSetting.Observations,
                representations: Tensor) -> PolicyHeadOutput:
        # NOTE: Here we could probably use either as the 'state':
        # state = observations.x
        # state = representations
        representations = representations.float()
        if len(representations.shape) != 2:
            representations = representations.reshape([-1, self.actor_input_dims])
        
        # TODO: Actually implement the actor-critic forward pass.
        # predicted_reward = self.critic([state, action])
        value = self.critic(representations)
        # Do we want to detach the representations? or not?
        
        logits = self.actor(representations)
        # The policy is the distribution over actions given the current state.
        action_dist = Categorical(logits=logits)
        
        if action_dist.has_rsample:
            sample = action_dist.rsample()
        else:
            sample = action_dist.sample()

        actions = A2CHeadOutput(
            y_pred=sample,
            logits=logits,
            action_dist=action_dist,
            value=value,
        )
        return actions

    def get_loss(self,
                 forward_pass: ForwardPass,
                 actions: A2CHeadOutput,
                 rewards: ContinualRLSetting.Rewards) -> Loss:
        if not self.representations:
            self.batch_size = forward_pass.batch_size
            self.create_buffers()
        
        action_dist: Categorical = actions.action_dist
        critic_value = actions.value
        rewards = rewards.to(device=actions.device)
        env_reward = torch.as_tensor(rewards.y, device=actions.device)
        observations: ContinualRLSetting.Observations = forward_pass.observations
        assert observations.done is not None, "Need the end-of-episode signal!"
        done = torch.as_tensor(observations.done, device=actions.device)
        
        representations = forward_pass.representations

        total_loss = Loss(self.name)

        critic_loss_tensor = torch.zeros(1).type_as(representations)
        actor_loss_tensor = torch.zeros(1).type_as(representations)
        entropy_loss_tensor = torch.zeros(1).type_as(representations)
        for env_index, env_done in enumerate(done):
            if env_done:
                env_representations, env_actions, env_rewards = self.stack_buffers(env_index)
                # TODO: Make sure this is correct, do the final 'rewards' match
                # what we'd expect them to?
                # todo: might need to detach all representations apart from the last one?
                env_values = self.critic(env_representations.detach())
                env_returns = self.get_returns(env_rewards.y, gamma=self.hparams.gamma)

                critic_loss_tensor += F.mse_loss(env_values, env_returns.type_as(env_values).reshape(env_values.shape))

                self.clear_buffers(env_index)

            # Add stuff to the buffers for this env.
            # Take a slice across the first dimension
            current_representations = representations[env_index]
            current_actions = get_slice(actions, env_index)
            current_rewards = get_slice(rewards, env_index)
            
            self.representations[env_index].append(current_representations)
            self.actions[env_index].append(current_actions)
            self.rewards[env_index].append(current_rewards)

            # Get the last N items from that env (from the buffers)
            # TODO: I don't think it makes sense to calculate a critic loss like
            # this when we just had the `done` signal, right? Should we enforce
            # some kind of restriction on the number of steps to have?
            env_representations, env_actions, env_rewards = self.stack_buffers(env_index)
            env_values = self.critic(env_representations)

            env_returns = self.get_returns(env_rewards.y, gamma=self.hparams.gamma).type_as(env_values)

            env_advantages = env_returns - env_values
            env_log_probs: Tensor = actions.action_log_prob
            actor_loss_tensor += (-env_log_probs * env_advantages.detach()).mean()

            # Just to make sure that the operation is differentiable if needed.
            assert actions.y_pred_log_prob.requires_grad == actor_loss_tensor.requires_grad

            critic_loss_tensor += F.mse_loss(env_values.reshape(env_returns.shape), env_returns)

            entropy_loss_tensor += - current_actions.action_dist.entropy()


        critic_loss = Loss("critic", critic_loss_tensor)
        actor_loss = Loss("actor", actor_loss_tensor)
        entropy_loss = Loss("entropy", entropy_loss_tensor)            

        total_loss = Loss(self.name)
        total_loss += self.hparams.actor_loss_coef * actor_loss
        total_loss += self.hparams.critic_loss_coef * critic_loss
        total_loss += self.hparams.entropy_loss_coef * entropy_loss
        self.detach_all_buffers()
        
        return total_loss
        # TODO: Need to detach something here, right?
        advantage: Tensor = (
            env_reward
            +  (~done) * self.hparams.gamma * self.critic(self._current_state)
            - self.critic(self._previous_state) # detach previous representations?
        )
        
        total_loss = Loss(self.name)

        critic_loss_tensor = (advantage ** 2).mean()
        critic_loss = Loss("critic", loss=critic_loss_tensor)

        total_loss += critic_loss

        actor_loss_tensor = - action_dist.log_prob(actions.action) * advantage.detach()
        actor_loss_tensor = actor_loss_tensor.mean()
        actor_loss = Loss("actor", loss=actor_loss_tensor)
        
        total_loss += actor_loss

        return total_loss


def concat_obs_and_action(observation_action: Tuple[Tensor, Tensor]) -> Tensor:
    observation, action = observation_action
    batch_size = observation.shape[0]
    observation = observation.reshape([batch_size, -1])
    action = action.reshape([batch_size, -1])
    return torch.cat([observation, action], dim=-1)
