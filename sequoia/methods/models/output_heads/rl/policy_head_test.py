from functools import partial
from typing import Optional, Callable, Sequence, Any

import gym
import numpy as np
import pytest
import torch
from gym import spaces
from torch import Tensor, nn

from sequoia.common.gym_wrappers import (AddDoneToObservation, ConvertToFromTensors,
                                 EnvDataset)
from sequoia.common.gym_wrappers.batch_env import BatchedVectorEnv
from sequoia.common.gym_wrappers.batch_env.worker import FINAL_STATE_KEY
from sequoia.common.loss import Loss
from sequoia.conftest import DummyEnvironment, xfail_param
from sequoia.methods.models.forward_pass import ForwardPass
from sequoia.settings.active.continual import ContinualRLSetting

from .policy_head import Categorical, PolicyHead, PolicyHeadOutput
from gym.vector import VectorEnv, SyncVectorEnv
from gym.vector.utils import batch_space


from sequoia.common.metrics.rl_metrics import EpisodeMetrics, RLMetrics


class FakeEnvironment(SyncVectorEnv):
    def __init__(self,
                 env_fn: Callable[[], gym.Env],
                 batch_size: int,
                 new_episode_length: Callable[[int], int],
                 episode_lengths: Sequence[int] = None,
                ):
        super().__init__([
            env_fn for _ in range(batch_size)
        ])
        self.new_episode_length = new_episode_length
        self.batch_size = batch_size
        self.episode_lengths = np.array(episode_lengths or [
            new_episode_length(i) for i in range(self.num_envs)
        ])
        self.steps_left_in_episode = self.episode_lengths.copy()
        
        reward_space = spaces.Box(*self.reward_range, shape=())
        self.single_reward_space = reward_space
        self.reward_space = batch_space(reward_space, batch_size)
 
    def step(self, actions):
        self.steps_left_in_episode[:] -= 1
        
        # obs, reward, done, info = super().step(actions)
        obs = self.observation_space.sample()
        reward = np.ones(self.batch_size)
        
        assert not any(self.steps_left_in_episode < 0)
        done = self.steps_left_in_episode == 0

        info = np.array([{} for _ in range(self.batch_size)])

        for env_index, env_done in enumerate(done):
            if env_done:
                next_episode_length = self.new_episode_length(env_index)
                self.episode_lengths[env_index] = next_episode_length
                self.steps_left_in_episode[env_index] = next_episode_length

        return obs, reward, done, info



from sequoia.common.layers import Flatten
from gym.spaces.utils import flatdim, flatten


@pytest.mark.parametrize("batch_size",
[
    2,
    5,
])
def test_with_controllable_episode_lengths(batch_size: int, monkeypatch):
    """ TODO: Test out the PolicyHead in a very controlled environment, where we
    know exactly the lengths of each episode.
    """
    env = FakeEnvironment(
        partial(gym.make, "CartPole-v0"),
        batch_size=batch_size,
        episode_lengths=[5, *(10 for _ in range(batch_size-1))],
        new_episode_length=lambda env_index: 10,
    )
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)
    
    
    obs_space = env.single_observation_space
    x_dim = flatdim(obs_space[0])
    # Create some dummy encoder.
    encoder = nn.Linear(x_dim, x_dim)
    representation_space = obs_space[0]

    output_head = PolicyHead(
        input_space=representation_space,
        action_space=env.single_action_space,
        reward_space=env.single_reward_space,
        hparams=PolicyHead.HParams(max_episode_window_length=100, min_episodes_before_update=1)
    )

    # Simplify the loss function so we know exactly what the loss should be at
    # each step.
    
    def mock_policy_gradient(rewards: Sequence[float], log_probs: Sequence[float], gamma: float = 0.95) -> Optional[Loss]:
        log_probs = (log_probs - log_probs.clone()) + 1
        # Return the length of the episode, but with a "gradient" flowing back into log_probs.
        return len(rewards) * log_probs.mean()
    
    monkeypatch.setattr(output_head, "policy_gradient", mock_policy_gradient)

    batch_size = env.batch_size 
        
    obs = env.reset()
    step_done = np.zeros(batch_size, dtype=np.bool)
    
    for step in range(200):
        x, obs_done = obs
        
        # The done from the obs should always be the same as the 'done' from the 'step' function.
        assert np.array_equal(obs_done, step_done)

        representations = encoder(x)
        observations = ContinualRLSetting.Observations(
            x=x,
            done=obs_done,
        )

        actions_obj = output_head(observations, representations)
        actions = actions_obj.y_pred
      
        # TODO: kinda useless to wrap a single tensor in an object..
        forward_pass = ForwardPass(
            observations=observations,
            representations=representations,
            actions=actions,
        )
        obs, rewards, step_done, info = env.step(actions)
        
        rewards_obj = ContinualRLSetting.Rewards(y=rewards)
        loss = output_head.get_loss(
            forward_pass=forward_pass,
            actions=actions,
            rewards=rewards_obj,
        )
        print(f"Step {step}")
        print(f"num episodes since update: {output_head.num_episodes_since_update}")
        print(f"Tensors with gradients: {output_head.num_grad_tensors}")
        print(f"Tensors without gradients: {output_head.num_detached_tensors}")
        print(f"steps left in episode: {env.steps_left_in_episode}")
        print(f"Loss for that step: {loss}")

        if any(obs_done):
            assert loss != 0.
        
        if step == 5.:
            # Env 0 first episode from steps 0 -> 5
            assert loss.loss == 5. 
            assert loss.metrics["gradient_usage"].used_gradients == 5.
            assert loss.metrics["gradient_usage"].wasted_gradients == 0.            
        elif step % 10 == 0 and step != 0:
            # Env 1 to batch_size, first episode, from steps 0 -> 10
            assert loss.loss == 10. * (batch_size-1) 
            assert loss.metrics["gradient_usage"].used_gradients == 10. * (batch_size-1)
            assert loss.metrics["gradient_usage"].wasted_gradients == 0.   
        elif step % 10 == 5:
            # Env 0 second episode from steps 5 -> 15
            assert loss.loss == 10.
            assert loss.metrics["gradient_usage"].used_gradients == 5
            assert loss.metrics["gradient_usage"].wasted_gradients == 5
        else:
            assert loss.loss == 0.


@pytest.mark.xfail(reason="Older, confusing test.")
@pytest.mark.parametrize("batch_size",
[
    1,
    2,
    5,
])
def test_loss_is_nonzero_at_episode_end(batch_size: int):
    """ Test that when stepping through the env, when the episode ends, a
    non-zero loss is returned by the output head.
    """
    with gym.make("CartPole-v0") as temp_env:
        temp_env = AddDoneToObservation(temp_env)
        obs_space = temp_env.observation_space
        action_space = temp_env.action_space
        reward_space = getattr(temp_env, "reward_space",
                               spaces.Box(*temp_env.reward_range, shape=())) 

    env = gym.vector.make("CartPole-v0", num_envs=batch_size, asynchronous=False)
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)

    head = PolicyHead(
        input_space=obs_space[0],
        action_space=action_space,
        reward_space=reward_space,
    )
    head.train()

    env.seed(123)
    obs = env.reset()

    # obs = torch.as_tensor(obs, dtype=torch.float32)

    done = torch.zeros(batch_size, dtype=bool)
    info = np.array([{} for _ in range(batch_size)])
    loss = None
    
    non_zero_losses = 0
    
    encoder = nn.Linear(4, 4)
    encoder.train()
    
    for i in range(100):
        representations = encoder(obs[0])
        
        observations = ContinualRLSetting.Observations(
            x=obs[0],
            done=done,
            # info=info,
        )
        head_output = head.forward(observations, representations=representations)
        actions = head_output.actions.numpy().tolist()
        # actions = np.zeros(batch_size, dtype=int).tolist()

        obs, rewards, done, info = env.step(actions)
        done = torch.as_tensor(done, dtype=bool)
        rewards = ContinualRLSetting.Rewards(rewards)
        assert len(info) == batch_size

        print(f"Step {i}, obs: {obs}, done: {done}, info: {info}")

        forward_pass = ForwardPass(         
            observations=observations,
            representations=representations,
            actions=head_output,
        )
        loss = head.get_loss(forward_pass, actions=head_output, rewards=rewards)
        print("loss:", loss)

        for env_index, env_is_done in enumerate(observations.done):
            if env_is_done:
                print(f"Episode ended for env {env_index} at step {i}")
                assert loss.loss != 0.
                non_zero_losses += 1
                break
        else:
            print(f"No episode ended on step {i}, expecting no loss.")
            assert loss is None or loss.loss == 0.

    assert non_zero_losses > 0


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_done_is_sometimes_True_when_iterating_through_env(batch_size: int):
    """ Test that when *iterating* through the env, done is sometimes 'True'.
    """
    env = gym.vector.make("CartPole-v0", num_envs=batch_size, asynchronous=True)
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)
    for i, obs in zip(range(100), env):
        print(i, obs[1])
        reward = env.send(env.action_space.sample())
        if any(obs[1]):
            break
    else:
        assert False, "Never encountered done=True!"


@pytest.mark.parametrize("batch_size",
[
    1,
    xfail_param(2, reason="doesn't work with batched envs yet."),
    xfail_param(5, reason="doesn't work with batched envs yet."),
])
def test_loss_is_nonzero_at_episode_end_iterate(batch_size: int):
    """ Test that when *iterating* through the env (active-dataloader style),
    when the episode ends, a non-zero loss is returned by the output head.
    """
    with gym.make("CartPole-v0") as temp_env:
        temp_env = AddDoneToObservation(temp_env)
        
        obs_space = temp_env.observation_space
        action_space = temp_env.action_space
        reward_space = getattr(temp_env, "reward_space",
                               spaces.Box(*temp_env.reward_range, shape=())) 

    env = gym.vector.make("CartPole-v0", num_envs=batch_size, asynchronous=False)
    env = AddDoneToObservation(env)
    env = ConvertToFromTensors(env)
    env = EnvDataset(env)

    head = PolicyHead(
        # observation_space=obs_space,
        input_space=obs_space[0],
        action_space=action_space,
        reward_space=reward_space,
    )

    env.seed(123)
    non_zero_losses = 0
    
    for i, obs in zip(range(100), env):
        print(i, obs)
        x = obs[0]
        done = obs[1]
        representations = x
        assert isinstance(x, Tensor)
        assert isinstance(done, Tensor)
        observations = ContinualRLSetting.Observations(
            x=x,
            done=done,
            # info=info,
        )
        head_output = head.forward(observations, representations=representations)
        
        actions = head_output.actions.numpy().tolist()
        # actions = np.zeros(batch_size, dtype=int).tolist()

        rewards = env.send(actions) 

        print(f"Step {i}, obs: {obs}, done: {done}")
        assert isinstance(representations, Tensor)
        forward_pass = ForwardPass(         
            observations=observations,
            representations=representations,
            actions=head_output,
        )
        rewards = ContinualRLSetting.Rewards(rewards)
        loss = head.get_loss(forward_pass, actions=head_output, rewards=rewards)
        print("loss:", loss)

        for env_index, env_is_done in enumerate(observations.done):
            if env_is_done:
                print(f"Episode ended for env {env_index} at step {i}")
                assert loss.total_loss != 0.
                non_zero_losses += 1
                break
        else:
            print(f"No episode ended on step {i}, expecting no loss.")
            assert loss.total_loss == 0.

    assert non_zero_losses > 0


@pytest.mark.xfail(reason="TODO: Fix this test")
def test_buffers_are_stacked_correctly(monkeypatch):
    """TODO: Test that when "de-synced" episodes, when fed to the output head,
    get passed, re-stacked correctly, to the get_episode_loss function.
    """
    batch_size = 5
    
    starting_values = [i for i in range(batch_size)]
    targets = [10 for i in range(batch_size)]
    
    env = BatchedVectorEnv([
        partial(DummyEnvironment, start=start, target=target, max_value=10 * 2)
        for start, target in zip(starting_values, targets)
    ])
    obs = env.reset()
    assert obs.tolist() == list(range(batch_size))
    
    reward_space = spaces.Box(*env.reward_range, shape=())
    output_head = PolicyHead(#observation_space=spaces.Tuple([env.observation_space,
                             #              spaces.Box(False, True, [batch_size], np.bool)]),
                             input_space=spaces.Box(0, 1, (1,)),
                             action_space=env.single_action_space,
                             reward_space=reward_space)
    # Set the max window length, for testing.
    output_head.hparams.max_episode_window_length = 100
    
    obs = initial_obs = env.reset()
    done = np.zeros(batch_size, dtype=bool)

    obs = torch.from_numpy(obs)
    done = torch.from_numpy(done)

    def mock_get_episode_loss(self: PolicyHead,
                              env_index: int,
                              inputs: Tensor,
                              actions: ContinualRLSetting.Observations,
                              rewards: ContinualRLSetting.Rewards,
                              done: bool) -> Optional[Loss]:
        print(f"Environment at index {env_index}, episode ended: {done}")
        if done:
            print(f"Full episode: {inputs}")
        else:
            print(f"Episode so far: {inputs}")

        n_observations = len(inputs)

        assert inputs.flatten().tolist() == (env_index + np.arange(n_observations)).tolist()
        if done:
            # Unfortunately, we don't get the final state, because of how
            # VectorEnv works atm.
            assert inputs[-1] == targets[env_index] - 1

    monkeypatch.setattr(PolicyHead, "get_episode_loss", mock_get_episode_loss)

    # perform 10 iterations, incrementing each DummyEnvironment's counter at
    # each step (action of 1).
    # Therefore, at first, the counters should be [0, 1, 2, ... batch-size-1].
    info = [{} for _ in range(batch_size)]
    
    for step in range(10):
        print(f"Step {step}.")
        # Wrap up the obs to pretend that this is the data coming from a
        # ContinualRLSetting.
        observations = ContinualRLSetting.Observations(x=obs, done=done)#, info=info)
        # We don't use an encoder for testing, so the representations is just x.
        representations = obs.reshape([batch_size, 1])
        assert observations.task_labels is None
        
        actions = output_head(observations.float(), representations.float())

        # Wrap things up to pretend like the output head is being used in the
        # BaselineModel:
                
        forward_pass = ForwardPass(
            observations = observations,
            representations = representations,
            actions = actions,
        )

        action_np = actions.actions_np
        
        obs, rewards, done, info = env.step(action_np)
        
        obs = torch.from_numpy(obs)
        rewards = torch.from_numpy(rewards)
        done = torch.from_numpy(done)
        
        rewards = ContinualRLSetting.Rewards(y=rewards)
        loss = output_head.get_loss(forward_pass, actions=actions, rewards=rewards)
        
        # Check the contents of the episode buffers.

        assert len(output_head.representations) == batch_size
        for env_index in range(batch_size):
            
            # obs_buffer = output_head.observations[env_index]
            representations_buffer = output_head.representations[env_index]
            action_buffer = output_head.actions[env_index]
            reward_buffer = output_head.rewards[env_index]
            
            if step >= batch_size:
                if step + env_index == targets[env_index]:
                    assert len(representations_buffer) == 1 and output_head.done[env_index] == False
                # if env_index == step - batch_size:
                continue
            assert len(representations_buffer) == step + 1
            # Check to see that the last entry in the episode buffer for this
            # environment corresponds to the slice of the most recent
            # observations/actions/rewards at the index corresponding to this
            # environment.
            
            # observation_tuple = input_buffer[-1]
            step_action = action_buffer[-1]
            step_reward = reward_buffer[-1]
            # assert observation_tuple.x == observations.x[env_index]
            # assert observation_tuple.task_labels is None
            # assert observation_tuple.done == observations.done[env_index]

            # The last element in the buffer should be the slice in the batch
            # for that environment. 
            assert step_action.y_pred == actions.y_pred[env_index]
            assert step_reward.y == rewards.y[env_index]

        if step < batch_size:
            assert obs.tolist() == (np.arange(batch_size) + step + 1).tolist()
        # if step >= batch_size:
        #     if step + env_index == targets[env_index]:
        #         assert done
                
    # assert False, (obs, rewards, done, info)
    # loss: Loss = output_head.get_loss(forward_pass, actions=actions, rewards=rewards)
from sequoia.common.gym_wrappers import PixelObservationWrapper
from sequoia.settings.active.continual.make_env import make_batched_env


def test_sanity_check_cartpole_done_vector(monkeypatch):
    """TODO: Sanity check, make sure that cartpole has done=True at some point
    when using a BatchedEnv.
    """
    batch_size = 5
    
    starting_values = [i for i in range(batch_size)]
    targets = [10 for i in range(batch_size)]
    
    env = make_batched_env("CartPole-v0", batch_size=5, wrappers=[PixelObservationWrapper])
    env = AddDoneToObservation(env)
    # env = AddInfoToObservation(env)
    
    # env = BatchedVectorEnv([
    #     partial(gym.make, "CartPole-v0") for i in range(batch_size)
    # ])
    obs = env.reset()
    
    for i in range(100):
        obs, rewards, done, info = env.step(env.action_space.sample())
        assert all(obs[1] == done), i
        if any(done):
            
            break
    else:
        assert False, "Should have had at least one done=True, over the 100 steps!"
