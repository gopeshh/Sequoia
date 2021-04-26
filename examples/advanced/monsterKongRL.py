import sys
sys.path.extend([".", ".."])
from sequoia.settings.active import ContinualRLSetting,IncrementalRLSetting, TaskIncrementalRLSetting
from sequoia.methods.stable_baselines3_methods import DQNMethod, A2CMethod, DQNModel, DQNMERMethod, DQNMERModel

if __name__=="__main__":

    setting = IncrementalRLSetting(
        dataset="monsterkong",
        steps_per_task=10_000,
        test_steps_per_task=10_000,
        train_task_schedule={
           0: {"level": 0},
           1: {"level": 4},
           2: {"level": 10},
           3: {"level": 14},
           4: {"level": 20},
           5: {"level": 24},
           6: {"level": 30},
           7: {"level": 34},
                            }
    )

    method = DQNMERMethod(train_steps_per_task=1000)
    results = setting.apply(method)
    print(results.summary())
    results.make_plots()

