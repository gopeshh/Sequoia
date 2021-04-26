""" Method that uses the DQN model from stable-baselines3 and targets the RL
settings in the tree.
"""

from dataclasses import dataclass
from typing import ClassVar, Optional, Type, Union
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.policies import BasePolicy
import gym
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from gym import spaces
from sequoia.common.hparams import categorical
from sequoia.common.transforms import ChannelsFirst
from sequoia.methods import register_method
from sequoia.settings.active import ContinualRLSetting
from sequoia.utils.logging_utils import get_logger
from simple_parsing import mutable_field
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, \
    TrainFrequencyUnit
from stable_baselines3.dqn import DQN
from copy import deepcopy

from .off_policy_method import OffPolicyMethod, OffPolicyModel

logger = get_logger(__file__)


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class selectorAgent(nn.Module):
    def __init__(self, inSize):
        super(selectorAgent, self).__init__()
        self.linear = nn.Linear(inSize, 1)
        self.linear.apply(Xavier)

    def forward(self,x):
        logits = self.linear(x).sum(-1)
        return F.softmax(logits), F.log_softmax(logits)

class DQN_ARModel(DQN, OffPolicyModel):
    """ Customized version of the DQN model from stable-baselines-3. """

    # def __init__(self, policy: Union[str, Type[DQNPolicy]], env: Union[GymEnv, str], policy_base: Type[BasePolicy],
    #              learning_rate: Union[float, Schedule], beta=1, gammaR=0.3, steps=1):
    #     super().__init__(policy, env, policy_base, learning_rate)
    #     self.beta = beta
    #     self.gammaR = gammaR
    #     self.steps = steps

    @dataclass
    class HParams(OffPolicyModel.HParams):
        """ Hyper-parameters of the DQN model from `stable_baselines3`.

        The command-line arguments for these are created with simple-parsing.
        """
        # How many steps of the model to collect transitions for before learning
        # starts.
        learning_starts: int = 50_000

        # Minibatch size for each gradient update
        batch_size: int = 32

        # Update the model every ``train_freq`` steps. Set to `-1` to disable.
        train_freq: int = 4
        # train_freq: int = categorical(1, 10, 100, 1_000, 10_000, default=4)

        # The soft update coefficient ("Polyak update", between 0 and 1) default
        # 1 for hard update
        tau: float = 1.0
        # tau: float = uniform(0., 1., default=1.0)

        # Update the target network every ``target_update_interval`` environment
        # steps.
        target_update_interval: int = categorical(
            1, 10, 100, 1_000, 10_000, default=10_000
        )
        # Fraction of entire training period over which the exploration rate is
        # reduced.
        exploration_fraction: float = 0.1
        # exploration_fraction: float = uniform(0.05, 0.3, default=0.1)
        # Initial value of random action probability.
        exploration_initial_eps: float = 1.0
        # exploration_initial_eps: float = uniform(0.5, 1.0, default=1.0)
        # final value of random action probability.
        exploration_final_eps: float = 0.05
        # exploration_final_eps: float = uniform(0, 0.1, default=0.05)
        # The maximum value for the gradient clipping.
        max_grad_norm: float = 10
        # max_grad_norm: float = uniform(1, 100, default=10)

    def train(self, gradient_steps: int, batch_size: int = 16, beta=1, steps=1, gammaR=0.3, search_size = 25) -> None:
        # super().train(gradient_steps, batch_size=batch_size)
        print("In training loop")
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        ssd_data = self.replay_buffer.sample(search_size, env=self._vec_normalize_env)

        with torch.no_grad():
            nextQVal = self.q_net_target(ssd_data.next_observations)
            nextQVal = nextQVal.max(dim=1)
            nextQVal = nextQVal.values.reshape(-1, 1)
            targetQVal = ssd_data.rewards + (1 - ssd_data.dones) * self.gamma * nextQVal
        currentQVal = self.q_net(ssd_data.observations)
        currentQVal = torch.gather(currentQVal, dim=1, index=ssd_data.actions.long())
        metaLossBefore = F.smooth_l1_loss(currentQVal, targetQVal)
        TDError = targetQVal - currentQVal
        sAgent = selectorAgent(3)
        selectorOpt = torch.optim.Adam(sAgent.parameters(),beta)

        selectedBatch, logproba, entropy = self.getSelectedBatch(search_size, TDError, sAgent)
        self.policy.optimizer.zero_grad()

        with torch.no_grad():
            nextQVal_PolicyAgent = self.q_net_target(selectedBatch.next_observations)
            nextQVal_PolicyAgent = nextQVal_PolicyAgent.max(dim=1)
            nextQVal_PolicyAgent = nextQVal_PolicyAgent.values.reshape(-1, 1)
            targetQVal_PolicyAgent = selectedBatch.rewards + (1 - selectedBatch.dones) * self.gamma * nextQVal_PolicyAgent
        currentQVal_PolicyAgent = self.q_net(selectedBatch.observations)
        currentQVal_PolicyAgent = torch.gather(currentQVal_PolicyAgent, dim=1, index=selectedBatch.actions.long())
        policyAgentLoss = F.smooth_l1_loss(currentQVal_PolicyAgent, targetQVal_PolicyAgent)
        losses.append(policyAgentLoss.item())
        policyAgentLoss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        with torch.no_grad():
            nextQValNew = self.q_net_target(ssd_data.next_observations)
            nextQValNew = nextQValNew.max(dim=1)
            nextQValNew = nextQValNew.values.reshape(-1, 1)
            targetQValNew = ssd_data.rewards + (1 - ssd_data.dones) * self.gamma * nextQValNew
        currentQValNew = self.q_net(ssd_data.observations)
        currentQValNew = torch.gather(currentQValNew, dim=1, index=ssd_data.actions.long())
        metaLossAfter = F.smooth_l1_loss(currentQValNew, targetQValNew)
        value = metaLossBefore - metaLossAfter
        selectorLoss = - logproba*value - self.gamma*entropy
        selectorLoss.backward()
        sAgent.opt.step()

    def getSelectedBatch(self, search_size, TDError, sAgent):
        tempData = self.replay_buffer.sample(search_size, env=self._vec_normalize_env)
        qcurrentFeatures = self.q_net.extract_features(tempData.observations)
        qnextFeatures = self.q_net.extract_features(tempData.next_observations)

        inputMatrix = torch.cat((qcurrentFeatures,TDError,tempData.actions),1)
        numEntries = tempData.actions.size()[0]
        probs,logProbs = sAgent.forward(inputMatrix)
        entropy = -(logProbs * probs).sum(0)
        p = probs.data.numpy()
        selection = np.random.choice(numEntries, 1, p=p)
        logproba = logProbs[selection]
        selectedData = tempData[selection]
        return selectedData, logproba, entropy

@register_method
@dataclass
class DQN_ARMethod(OffPolicyMethod):
    """ Method that uses a DQN model from the stable-baselines3 package. """

    Model: ClassVar[Type[DQN_ARModel]] = DQN_ARModel

    # Hyper-parameters of the DQN model.
    hparams: DQN_ARModel.HParams = mutable_field(DQN_ARModel.HParams)

    # Approximate limit on the size of the replay buffer, in megabytes.
    max_buffer_size_megabytes: float = 2_048.0

    def configure(self, setting: ContinualRLSetting):
        super().configure(setting)
        # NOTE: Need to change some attributes depending on the maximal number of steps
        # in the environment allowed in the given Setting.
        if setting.steps_per_phase:
            ten_percent_of_step_budget = setting.steps_per_phase // 10
            if self.hparams.target_update_interval > ten_percent_of_step_budget:
                # Same for the 'update target network' interval.
                self.hparams.target_update_interval = ten_percent_of_step_budget // 2
                logger.info(
                    f"Reducing the target network update interval to "
                    f"{self.hparams.target_update_interval}, because of the limit on "
                    f"training steps imposed by the Setting."
                )

    def create_model(self, train_env: gym.Env, valid_env: gym.Env) -> DQN_ARModel:
        return self.Model(env=train_env, **self.hparams.to_dict())

    def fit(self, train_env: gym.Env, valid_env: gym.Env):
        super().fit(train_env=train_env, valid_env=valid_env)

    def get_actions(
            self, observations: ContinualRLSetting.Observations, action_space: spaces.Space
    ) -> ContinualRLSetting.Actions:
        obs = observations.x
        # Temp fix for monsterkong and DQN:
        if obs.shape == (64, 64, 3):
            obs = ChannelsFirst.apply(obs)
        predictions = self.model.predict(obs)
        action, _ = predictions
        assert action in action_space, (observations, action, action_space)
        return action

    def on_task_switch(self, task_id: Optional[int]) -> None:
        """ Called when switching tasks in a CL setting.

        If task labels are available, `task_id` will correspond to the index of
        the new task. Otherwise, if task labels aren't available, `task_id` will
        be `None`.

        todo: use this to customize how your method handles task transitions.
        """


if __name__ == "__main__":
    results = DQN_ARMethod.main()
    print(results)
