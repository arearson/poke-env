import asyncio
import os

import gym
import tensorflow
import tensorflow.compat.v1 as tf
from gym.spaces import Space, Box, MultiBinary, Discrete
from gym.utils.env_checker import check_env
# from rl.agents.dqn import DQNAgent
# from rl.memory import SequentialMemory
# from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from tabulate import tabulate
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
from tf_agents import drivers, train
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.agents.categorical_dqn import categorical_dqn_agent
from tf_agents.networks import sequential, categorical_q_network, NestFlatten
from tf_agents.policies import policy_saver
from tf_agents.policies.random_py_policy import RandomPyPolicy
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.replay_buffers import TFUniformReplayBuffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.environments import tf_py_environment, gym_wrapper, py_environment, FlattenObservationsWrapper, \
    validate_py_environment, parallel_py_environment, batched_py_environment

from examples.ai_players import SimpleRLPlayer, SimpleGen9RLPlayer
# import tensorflow.compat.v1 as tf


from poke_env.environment import AbstractBattle, MoveCategory
from poke_env.player import (
    background_evaluate_player,
    background_cross_evaluate,
    Gen8EnvSinglePlayer,
    RandomPlayer,
    MaxBasePowerPlayer,
    ObservationType,
    SimpleHeuristicsPlayer,
    cross_evaluate,
)

import numpy as np

from poke_env.data import GenData


# Sequential = tf.keras.Sequential
# Model = tf.keras.Model


async def main():
    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    # opponent = RandomPlayer(battle_format="gen8randombattle")
    # test_env = SimpleRLPlayer(battle_format="gen8randombattle", opponent=opponent, start_challenging=True)
    # check_env(test_env)
    # test_env.close()

    # Create one environment for training and one for evaluation
    # opponent = MaxBasePowerPlayer(battle_format="gen8randombattle", max_concurrent_battles=10)
    # train_env = SimpleRLPlayer(
    #     battle_format="gen8randombattle",
    #     opponent=opponent, start_challenging=True,
    #     # use_old_gym_api=False
    # )
    opponent2 = RandomPlayer(battle_format="gen9randombattle")
    eval_env = SimpleGen9RLPlayer(
        battle_format="gen9randombattle",
        opponent=opponent2, start_challenging=True,
        # use_old_gym_api=False
    )

    # Compute dimensions
    # n_action = train_env.action_space.n
    # print(n_action)
    # input_shape = (1,) + train_env.observation_space.shape

    # Create model
    # model = Sequential()
    # model.add(Dense(128, activation="elu", input_shape=input_shape))
    # model.add(Flatten())
    # model.add(Dense(64, activation="elu"))
    # model.add(Dense(n_action, activation="linear"))

    # inputs = tf.keras.Input(shape=input_shape)
    # x = Flatten()(inputs)
    # x = Dense(64, activation='elu')(x)
    # outputs = Dense(n_action, activation="linear")(x)
    # model = Model(inputs=inputs, outputs=outputs)
    # space = train_env.observation_space.__dict__
    # train_env.observation_space = gym.spaces.Box(high=space['high'], low=space['low'])
    # train_env = gym_wrapper.GymWrapper(train_env)
    # # train_env._observation_spec._shape = (10,)
    # # print(train_env.time_step_spec())
    # # train_env = batched_py_environment.BatchedPyEnvironment([train_env])
    # validate_py_environment(train_env)
    # train_env = tf_py_environment.TFPyEnvironment(train_env)

    def createEnv(pyenv):
        # space = env.observation_space.__dict__
        # env.observation_space = gym.spaces.Box(high=space['high'], low=space['low'])
        pyenv = gym_wrapper.GymWrapper(pyenv)
        # env = parallel_py_environment.ParallelPyEnvironment([para for _ in range(10)])
        # train_env._observation_spec._shape = (10,)
        # print(train_env.time_step_spec())
        # train_env = batched_py_environment.BatchedPyEnvironment([train_env])
        # validate_py_environment(env)
        pyenv = tf_py_environment.TFPyEnvironment(pyenv, isolation=True)
        # print(tensor_spec.from_spec(train_env.observation_spec()))
        return pyenv

    # train_env = batched_py_environment.BatchedPyEnvironment([
    #     gym_wrapper.GymWrapper(SimpleRLPlayer(
    #         battle_format="gen8randombattle",
    #         opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True,
    #         # use_old_gym_api=False
    #     )),
    #     gym_wrapper.GymWrapper(SimpleRLPlayer(
    #         battle_format="gen8randombattle",
    #         opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True,
    #         # use_old_gym_api=False
    #     )),
    #     gym_wrapper.GymWrapper(SimpleRLPlayer(
    #         battle_format="gen8randombattle",
    #         opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True,
    #         # use_old_gym_api=False
    #     )),
    #     gym_wrapper.GymWrapper(SimpleRLPlayer(
    #         battle_format="gen8randombattle",
    #         opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True,
    #         # use_old_gym_api=False
    #     )), gym_wrapper.GymWrapper(SimpleRLPlayer(
    #         battle_format="gen8randombattle",
    #         opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True,
    #         # use_old_gym_api=False
    #     )),
    #     gym_wrapper.GymWrapper(SimpleRLPlayer(
    #         battle_format="gen8randombattle",
    #         opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True,
    #         # use_old_gym_api=False
    #     )),
    #     gym_wrapper.GymWrapper(SimpleRLPlayer(
    #         battle_format="gen8randombattle",
    #         opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True,
    #         # use_old_gym_api=False
    #     )),
    #     gym_wrapper.GymWrapper(SimpleRLPlayer(
    #         battle_format="gen8randombattle",
    #         opponent=RandomPlayer(battle_format="gen8randombattle"), start_challenging=True,
    #         # use_old_gym_api=False
    #     ))])

    train_env = batched_py_environment.BatchedPyEnvironment([
        gym_wrapper.GymWrapper(SimpleGen9RLPlayer(
            battle_format="gen9randombattle",
            opponent=RandomPlayer(battle_format="gen9randombattle"), start_challenging=True,
            # use_old_gym_api=False
        )),
        gym_wrapper.GymWrapper(SimpleGen9RLPlayer(
            battle_format="gen9randombattle",
            opponent=RandomPlayer(battle_format="gen9randombattle"), start_challenging=True,
            # use_old_gym_api=False
        )),
        gym_wrapper.GymWrapper(SimpleGen9RLPlayer(
            battle_format="gen9randombattle",
            opponent=RandomPlayer(battle_format="gen9randombattle"), start_challenging=True,
            # use_old_gym_api=False
        )),
        gym_wrapper.GymWrapper(SimpleGen9RLPlayer(
            battle_format="gen9randombattle",
            opponent=RandomPlayer(battle_format="gen9randombattle"), start_challenging=True,
            # use_old_gym_api=False
        )),
    ])

    train_env = tf_py_environment.TFPyEnvironment(train_env, check_dims=True, isolation=False)

    # train_env = createEnv(train_env)
    eval_env = createEnv(eval_env)

    # fc_layer_params = (128, 64)
    # action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
    # num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    # def dense_layer(num_units):
    #     return tf.keras.layers.Dense(
    #         num_units,
    #         activation=tf.keras.activations.relu,
    #         kernel_initializer=tf.keras.initializers.VarianceScaling(
    #             scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    # dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    # q_values_layer = tf.keras.layers.Dense(
    #     num_actions,
    #     activation=None,
    #     kernel_initializer=tf.compat.v1.keras.initializers.RandomUniform(
    #         minval=0.03, maxval=1.0),
    #     bias_initializer=tf.keras.initializers.Constant(-0.2))
    # q_net = sequential.Sequential([*dense_layers, q_values_layer])

    # Defining the DQN
    # memory = SequentialMemory(limit=10000, window_length=1)
    #
    # policy = LinearAnnealedPolicy(
    #     EpsGreedyQPolicy(),
    #     attr="eps",
    #     value_max=1.0,
    #     value_min=0.05,
    #     value_test=0.0,
    #     nb_steps=10000,
    # )

    # dqn = DQNAgent(
    #     model=model,
    #     nb_actions=n_action,
    #     policy=policy,
    #     memory=memory,
    #     nb_steps_warmup=1000,
    #     gamma=0.5,
    #     target_model_update=1,
    #     delta_clip=0.01,
    #     enable_double_dqn=True,
    # )
    # dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    num_iterations = 15000  # @param {type:"integer"}

    initial_collect_steps = 1000  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_capacity = 10000  # @param {type:"integer"}

    fc_layer_params = (256, 128)

    batch_size = 256  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    epsilon_greedy = 0.3
    gamma = 0.8
    log_interval = 200  # @param {type:"integer"}

    num_atoms = 51  # @param {type:"integer"}
    min_q_value = -20  # @param {type:"integer"}
    max_q_value = 20  # @param {type:"integer"}
    n_step_update = 2  # @param {type:"integer"}

    num_eval_episodes = 1  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    # dqn = DqnAgent(
    #     # tf_py_environment.TFPyEnvironment(train_env).time_step_spec(),
    #     train_env.time_step_spec(),
    #     train_env.action_spec(),
    #     # tf_py_environment.TFPyEnvironment(train_env).action_spec(),
    #     q_network=q_net,
    #     optimizer=optimizer,
    #     n_step_update=n_step_update,
    #     td_errors_loss_fn=common.element_wise_huber_loss,
    #     gamma=gamma,
    #     train_step_counter=train_step_counter)
    preprocessing_layers = {
        # 'image': tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, 4),
        #                                      tf.keras.layers.Flatten()]),
        # 'vector': tf.keras.layers.Dense(5),
        'events': tf.keras.layers.Dense(2),
        'mons': tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(32)]),
        'mon_types': tf.keras.layers.Flatten(),
        'hp': tf.keras.layers.Dense(6),
        'o_mons': tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(32)]),
        'o_mon_types': tf.keras.layers.Flatten(),
        'o_hp': tf.keras.layers.Dense(6),
        'moves': tf.keras.models.Sequential([tf.keras.layers.Flatten(), tf.keras.layers.Dense(32)]),
        'old': tf.keras.layers.Flatten(),
    }
    preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    categorical_q_net = categorical_q_network.CategoricalQNetwork(
        input_tensor_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        num_atoms=num_atoms,
        fc_layer_params=fc_layer_params,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
    )

    dqn = categorical_dqn_agent.CategoricalDqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        categorical_q_network=categorical_q_net,
        optimizer=optimizer,
        min_q_value=min_q_value,
        max_q_value=max_q_value,
        n_step_update=n_step_update,
        epsilon_greedy=epsilon_greedy,
        td_errors_loss_fn=common.element_wise_huber_loss,
        gamma=gamma,
        train_step_counter=train_step_counter)

    dqn.initialize()

    policy_dir = os.path.join('policy')
    check_dir = os.path.join('checkpoint')
    checkpointer = common.Checkpointer(check_dir, max_to_keep=1, agent=dqn)
    checkpointer.initialize_or_restore()
    tf_policy_saver = policy_saver.PolicySaver(dqn.policy)

    replay_buffer = TFUniformReplayBuffer(
        data_spec=dqn.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity,
        device='CPU:*'
    )

    driver = drivers.dynamic_step_driver.DynamicStepDriver(
        env=train_env, policy=dqn.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps
    )

    # Training the model
    dataset = replay_buffer.as_dataset(
        sample_batch_size=batch_size,
        num_steps=n_step_update + 1,
        single_deterministic_pass=False,
        num_parallel_calls=3,
    )

    driver.run()
    iterator = iter(dataset)
    for trainers in [1]:
        for envs in train_env.envs:
            if trainers == 1:
                trainer = MaxBasePowerPlayer(battle_format="gen9randombattle")
            elif trainers == 2:
                trainer = SimpleHeuristicsPlayer(battle_format="gen9randombattle")
            else:
                continue
            envs.reset_env(opponent=trainer)
        for _ in range([128, 1024, 64][trainers]):
            for _ in range(512):
                drivers.dynamic_step_driver.DynamicStepDriver(
                    env=train_env, policy=dqn.collect_policy,
                    observers=[replay_buffer.add_batch], num_steps=collect_steps_per_iteration
                ).run()
                trajectories, _ = next(iterator)
                dqn.train(experience=trajectories)
            checkpointer.save(train_step_counter)
            print(train_step_counter.numpy(), '\t')
            for envs in train_env.envs:
                print(envs.n_finished_battles, f'{envs.n_won_battles / envs.n_finished_battles * 100:0.1f}%',
                      end='\t\t')
            print()
        tf_policy_saver.save(policy_dir)

    tf_policy_saver.save(policy_dir)
    for env in train_env.envs:
        env.close()

    # EVALUATION!
    saved_policy = tensorflow.saved_model.load(policy_dir)
    # Evaluating the model
    print("Results against random player:")
    driver_eval = drivers.dynamic_episode_driver.DynamicEpisodeDriver(
        env=eval_env, policy=saved_policy, num_episodes=num_eval_episodes
    )
    driver_eval.run()

    # dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.envs[0].n_won_battles} victories "
        f"out of {eval_env.envs[0].n_finished_battles} episodes"
    )

    # Second eval
    second_opponent = MaxBasePowerPlayer(battle_format="gen9randombattle")
    eval_env.envs[0].reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    driver_eval1 = drivers.dynamic_episode_driver.DynamicEpisodeDriver(
        env=eval_env, policy=saved_policy, num_episodes=num_eval_episodes
    )
    driver_eval1.run()
    # dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.envs[0].n_won_battles} victories"
        f" out of {eval_env.envs[0].n_finished_battles} episodes"
    )

    # Reset env
    eval_env.envs[0].reset_env(restart=False)

    # # Evaluate the player with included util method
    # n_challenges = 250
    # placement_battles = 40
    # eval_task = background_evaluate_player(
    #     eval_env.envs[0].agent, n_battles=n_challenges, n_placement_battles=placement_battles
    # )
    # # dqn.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    # driver_eval2 = drivers.dynamic_episode_driver.DynamicEpisodeDriver(
    #     env=eval_env, policy=saved_policy, num_episodes=n_challenges - 2
    # )
    # driver_eval2.run()
    # time_step = eval_env.reset()
    # while not time_step.is_last():
    #     time_step = eval_env.step(saved_policy.action(time_step))
    #
    # print("Evaluation with included method:", eval_task.result())
    # eval_env.envs[0].reset_env(restart=False)

    # Cross evaluate the player with included util method
    n_challenges = 50
    players = [
        eval_env.envs[0].agent,
        RandomPlayer(battle_format="gen9randombattle"),
        MaxBasePowerPlayer(battle_format="gen9randombattle"),
        # SimpleHeuristicsPlayer(battle_format="gen9randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    # dqn.test(
    #     eval_env,
    #     nb_episodes=n_challenges * (len(players) - 1),
    #     verbose=False,
    #     visualize=False,
    # )
    driver_eval3 = drivers.dynamic_episode_driver.DynamicEpisodeDriver(
        env=eval_env, policy=saved_policy, num_episodes=n_challenges * (len(players) - 1) - 2
    )
    driver_eval3.run()

    time_step = eval_env.reset()
    while not time_step.is_last():
        time_step = eval_env.step(saved_policy.action(time_step))

    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    for env in eval_env.envs:
        env.close()


if __name__ == "__main__":
    # asyncio.get_event_loop().run_until_complete(main())
    asyncio.run(main())
