import asyncio

import numpy as np
from gym import Space
from gym.spaces import Box
from tf_agents.environments import TFPyEnvironment
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.policies.random_tf_policy import RandomTFPolicy

from poke_env.environment import AbstractBattle, MoveCategory
from poke_env.player import (
    Gen8EnvSinglePlayer,
    Gen9EnvSinglePlayer,
    SimpleHeuristicsPlayer,
    RandomPlayer,
    ObservationType,
)


class SimpleRLPlayer(Gen8EnvSinglePlayer, SimpleHeuristicsPlayer):
    def calc_reward(self, last_battle, current_battle) -> float:
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle: AbstractBattle) -> ObservationType:
        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6

        # Main mons shortcuts
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        # Rough estimation of damage ratio
        p_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(opponent, "def")
        s_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(opponent, "spd")

        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        moves_setup_hazard = -np.ones(4)
        moves_removal_hazard = -np.ones(4)
        moves_setup = -np.ones(4)

        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                    move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = ((1.5 if move.type in active.types else 1) *
                                           (p_ratio if move.category == MoveCategory.PHYSICAL else s_ratio) *
                                           move.accuracy *
                                           move.expected_hits *
                                           opponent.damage_multiplier(move))
            moves_dmg_multiplier = (moves_dmg_multiplier - np.min(moves_dmg_multiplier)) / (
                    np.max(moves_dmg_multiplier) - np.min(moves_dmg_multiplier) + .000001) * 4
            # Entry hazard...
            if (fainted_mon_opponent < 0.5
                    and move.id in self.ENTRY_HAZARDS
                    and self.ENTRY_HAZARDS[move.id]
                    not in battle.opponent_side_conditions):
                moves_setup_hazard[i] = 1
            else:
                moves_setup_hazard[i] = 0
            # remove hazard
            if (battle.side_conditions
                    and move.id in self.ANTI_HAZARDS_MOVES
                    and fainted_mon_team < 2 / 6):
                moves_removal_hazard[i] = 1
            else:
                moves_removal_hazard[i] = 0
            if (
                    (
                            active.current_hp_fraction == 1 and
                            self._estimate_matchup(active, opponent) > 0
                    ) and
                    (
                            move.boosts and
                            sum(move.boosts.values()) >= 2 and
                            move.target == "self" and
                            min([active.boosts[s] for s, v in move.boosts.items() if v > 0]) < 6
                    )
            ):
                moves_setup[i] = 1
            else:
                moves_setup[i] = 0

        # Switch Score
        best_switch = -np.ones(6)
        best_score = np.zeros(6)
        for i, mon in enumerate(battle.available_switches):
            best_score[i] = self._estimate_matchup(mon, opponent)
        best_switch = (best_score - np.min(best_score)) / (
                np.max(best_score) - np.min(best_score) + .00001) * 2 + best_switch

        other = np.array(
            [self._should_switch_out(battle),
             self._should_dynamax(battle, int((1 - fainted_mon_team) * 6))]).astype(int)

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                moves_setup_hazard,
                moves_removal_hazard,
                moves_setup,
                best_switch,
                other,
                [fainted_mon_team, fainted_mon_opponent],
            ],
        )
        # print(final_vector)
        return np.float32(np.array(final_vector))

    def describe_embedding(self) -> Space:
        low = [-1, -1, -1, -1,
               0, 0, 0, 0,
               -1, -1, -1, -1,
               -1, -1, -1, -1,
               -1, -1, -1, -1,
               -1, -1, -1, -1, -1, -1,
               0, 0,
               0, 0]
        high = [3, 3, 3, 3,
                4, 4, 4, 4,
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1, 1, 1,
                1, 1,
                1, 1]
        return Box(
            # np.array([low], dtype=np.float32),
            # np.array([high], dtype=np.float32),
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            # shape=(1,),
            dtype=np.float32,
        )


class SimpleGen9RLPlayer(Gen9EnvSinglePlayer, SimpleRLPlayer):
    pass
    # def calc_reward(self, last_battle, current_battle) -> float:
    #     return self.reward_computing_helper(
    #         current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
    #     )
    #
    # def embed_battle(self, battle: AbstractBattle) -> ObservationType:
    #     # We count how many pokemons have fainted in each team
    #     fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
    #     fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
    #
    #     # Main mons shortcuts
    #     active = battle.active_pokemon
    #     opponent = battle.opponent_active_pokemon
    #
    #     # Rough estimation of damage ratio
    #     p_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(opponent, "def")
    #     s_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(opponent, "spd")
    #
    #     # -1 indicates that the move does not have a base power
    #     # or is not available
    #     moves_base_power = -np.ones(4)
    #     moves_dmg_multiplier = np.ones(4)
    #     moves_setup_hazard = -np.ones(4)
    #     moves_removal_hazard = -np.ones(4)
    #     moves_setup = -np.ones(4)
    #
    #     for i, move in enumerate(battle.available_moves):
    #         moves_base_power[i] = (
    #                 move.base_power / 100
    #         )  # Simple rescaling to facilitate learning
    #         if move.type:
    #             moves_dmg_multiplier[i] = ((1.5 if move.type in active.types else 1) *
    #                                        (p_ratio if move.category == MoveCategory.PHYSICAL else s_ratio) *
    #                                        move.accuracy *
    #                                        move.expected_hits *
    #                                        opponent.damage_multiplier(move))
    #         moves_dmg_multiplier = (moves_dmg_multiplier - np.min(moves_dmg_multiplier)) / (
    #                 np.max(moves_dmg_multiplier) - np.min(moves_dmg_multiplier) + .000001) * 4
    #         # Entry hazard...
    #         if (fainted_mon_opponent < 0.5
    #                 and move.id in self.ENTRY_HAZARDS
    #                 and self.ENTRY_HAZARDS[move.id]
    #                 not in battle.opponent_side_conditions):
    #             moves_setup_hazard[i] = 1
    #         else:
    #             moves_setup_hazard[i] = 0
    #         # remove hazard
    #         if (battle.side_conditions
    #                 and move.id in self.ANTI_HAZARDS_MOVES
    #                 and fainted_mon_team < 2 / 6):
    #             moves_removal_hazard[i] = 1
    #         else:
    #             moves_removal_hazard[i] = 0
    #         if (
    #                 (
    #                         active.current_hp_fraction == 1 and
    #                         self._estimate_matchup(active, opponent) > 0
    #                 ) and
    #                 (
    #                         move.boosts and
    #                         sum(move.boosts.values()) >= 2 and
    #                         move.target == "self" and
    #                         min([active.boosts[s] for s, v in move.boosts.items() if v > 0]) < 6
    #                 )
    #         ):
    #             moves_setup[i] = 1
    #         else:
    #             moves_setup[i] = 0
    #
    #     # Switch Score
    #     best_switch = -np.ones(6)
    #     best_score = np.zeros(6)
    #     for i, mon in enumerate(battle.available_switches):
    #         best_score[i] = self._estimate_matchup(mon, opponent)
    #     best_switch = (best_score - np.min(best_score)) / (
    #             np.max(best_score) - np.min(best_score) + .00001) * 2 + best_switch
    #
    #     other = np.array(
    #         [self._should_switch_out(battle),
    #          self._should_dynamax(battle, int((1 - fainted_mon_team) * 6))]).astype(int)
    #
    #     # Final vector with 10 components
    #     final_vector = np.concatenate(
    #         [
    #             moves_base_power,
    #             moves_dmg_multiplier,
    #             moves_setup_hazard,
    #             moves_removal_hazard,
    #             moves_setup,
    #             best_switch,
    #             other,
    #             [fainted_mon_team, fainted_mon_opponent],
    #         ],
    #     )
    #     # print(final_vector)
    #     return np.float32(np.array(final_vector))
    #
    # def describe_embedding(self) -> Space:
    #     low = [-1, -1, -1, -1,
    #            0, 0, 0, 0,
    #            -1, -1, -1, -1,
    #            -1, -1, -1, -1,
    #            -1, -1, -1, -1,
    #            -1, -1, -1, -1, -1, -1,
    #            0, 0,
    #            0, 0]
    #     high = [3, 3, 3, 3,
    #             4, 4, 4, 4,
    #             1, 1, 1, 1,
    #             1, 1, 1, 1,
    #             1, 1, 1, 1,
    #             1, 1, 1, 1, 1, 1,
    #             1, 1,
    #             1, 1]
    #     return Box(
    #         # np.array([low], dtype=np.float32),
    #         # np.array([high], dtype=np.float32),
    #         np.array(low, dtype=np.float32),
    #         np.array(high, dtype=np.float32),
    #         # shape=(1,),
    #         dtype=np.float32,
    #     )


def run_env(test_env):
    test_env = TFPyEnvironment(GymWrapper(test_env), check_dims=True, isolation=False)
    policy = RandomTFPolicy(time_step_spec=test_env.time_step_spec(), action_spec=test_env.action_spec())
    time_step = test_env.reset()
    while not time_step.is_last():
        time_step = test_env.step(policy.action(time_step))
    print('finished:', test_env.envs[0].n_finished_battles, 'battles')
    test_env.envs[0].close()


async def main():
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle",
        opponent=opponent,
        start_challenging=True,
    )
    run_env(test_env)


async def mainGen9():
    opponent = RandomPlayer(battle_format="gen9randombattle")
    test_env_gen9 = SimpleGen9RLPlayer(
        battle_format="gen9randombattle",
        opponent=opponent,
        start_challenging=True,
    )
    run_env(test_env_gen9)


if __name__ == '__main__':
    asyncio.run(main())
    asyncio.run(mainGen9())
