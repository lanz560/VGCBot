import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from poke_env.battle import Pokemon
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.side_condition import SideCondition
from poke_env.player.battle_order import SingleBattleOrder

from bot import features as F


def mon(species='hatterene', speed=50, moves=()):
    pokemon = Pokemon(9, species=species)
    pokemon._current_hp = pokemon._max_hp = 200
    pokemon._stats = dict(hp=200, atk=100, spa=100, spd=100, spe=speed)
    pokemon._stats['def'] = 100
    pokemon.moves.update({name: Move(name, 9) for name in moves})
    return pokemon


def battle(mine=None, foes=None):
    mine = mine or [mon(), mon()]
    foes = foes or [mon(speed=100), mon(speed=110)]
    return SimpleNamespace(active_pokemon=mine, opponent_active_pokemon=foes,
                           side_conditions={}, opponent_side_conditions={},
                           fields={}, weather={}, force_switch=[False, False],
                           team={str(i): m for i, m in enumerate(mine)},
                           opponent_team={str(i): m for i, m in enumerate(foes)},
                           max_team_size=4, turn=1)


def features(b, slot, name, target=1):
    return F.single_features(b, slot, SingleBattleOrder(Move(name, 9), move_target=target))[1]


class ContextualFeatureTests(unittest.TestCase):
    def test_trick_room_gain_changes_sign_and_repeated_tailwind_is_neutral(self):
        b = battle()
        self.assertEqual(F.speed_control_gain(b, [Move('trickroom', 9)]), 1)
        b.fields = {Field.TRICK_ROOM: 1}
        self.assertEqual(F.speed_control_gain(b, [Move('trickroom', 9)]), -1)
        self.assertEqual(F.speed_control_gain(b, [Move('trickroom', 9)] * 2), 0)
        b.fields = {}
        b.side_conditions = {SideCondition.TAILWIND: 1}
        self.assertEqual(F.speed_control_gain(b, [Move('tailwind', 9)]), 0)

    def test_helping_hand_only_rewards_extra_damage_and_new_kos(self):
        b = battle()
        with patch.object(F, 'move_frac', return_value=0.7):
            attack = features(b, 1, 'tackle')
            hand = features(b, 0, 'helpinghand', -2)
            contextual = F.contextual_pair_features(b, hand, attack)
            self.assertAlmostEqual(contextual[F.P['helping_hand_damage']], 0.3)
            self.assertEqual(contextual[F.P['helping_hand_ko']], 1)
            protect = features(b, 1, 'protect')
            no_attack = F.contextual_pair_features(b, hand, protect)
            self.assertEqual(no_attack[F.P['helping_hand_damage']], 0)
            self.assertEqual(no_attack[F.P['helping_hand_ko']], 0)

    def test_redirection_distinguishes_single_target_from_spread_threat(self):
        b = battle(foes=[mon(moves=['tackle']), None])
        with patch.object(F, 'move_frac', return_value=0.8), \
                patch.object(F, 'threat_to', return_value=0.2):
            redirect = features(b, 0, 'followme')
            attack = features(b, 1, 'tackle')
            result = F.contextual_pair_features(b, redirect, attack)
            self.assertAlmostEqual(result[F.P['redirect_protection']], 0.64)
            b.opponent_active_pokemon[0].moves.clear()
            b.opponent_active_pokemon[0].moves['heatwave'] = Move('heatwave', 9)
            result = F.contextual_pair_features(b, redirect, attack)
            self.assertEqual(result[F.P['redirect_protection']], 0)

    def test_setup_rewards_safety_and_respects_stat_caps(self):
        b = battle(mine=[mon(moves=['swordsdance', 'tackle']), mon()])
        setup = features(b, 0, 'swordsdance')
        protect = features(b, 1, 'protect')
        with patch.object(F, 'threat_to', return_value=0.1):
            safe = F.contextual_pair_features(b, setup, protect)
            self.assertAlmostEqual(safe[F.P['setup_safety']], 0.9)
            self.assertAlmostEqual(safe[F.P['setup_risk']], 0.1)
            b.active_pokemon[0]._boosts['atk'] = 6
            capped = F.contextual_pair_features(b, setup, protect)
            self.assertEqual(capped[F.P['setup_safety']], 0)
        with patch.object(F, 'threat_to', return_value=1):
            unsafe = F.contextual_pair_features(b, setup, protect)
            self.assertEqual(unsafe[F.P['setup_safety']], 0)
            self.assertEqual(unsafe[F.P['setup_risk']], 1)

    def test_expanding_force_scores_both_foes_and_single_target_has_no_spread_penalty(self):
        b = battle()
        b.fields = {Field.PSYCHIC_TERRAIN: 1}
        with patch.object(F, 'move_frac', return_value=0.4):
            values, aux = F.single_features(b, 0, SingleBattleOrder(Move('expandingforce', 9), move_target=1))
            self.assertAlmostEqual(values[F.S['dmg_frac']], 0.6)
            self.assertEqual(len(aux['hits']), 2)
            self.assertIsNone(aux['tgt'])
            b.opponent_active_pokemon[1] = None
            values, aux = F.single_features(b, 0, SingleBattleOrder(Move('expandingforce', 9), move_target=1))
            self.assertAlmostEqual(values[F.S['dmg_frac']], 0.4)

    def test_candidate_pipeline_and_value_features_are_finite(self):
        b = battle()
        b.valid_orders = [
            [SingleBattleOrder(Move('helpinghand', 9), move_target=-2),
             SingleBattleOrder(Move('trickroom', 9))],
            [SingleBattleOrder(Move('tackle', 9), move_target=1)],
        ]
        orders, phi = F.build_candidates(b)
        self.assertEqual(len(orders), 2)
        self.assertEqual(phi.shape, (2, F.D_MOVE))
        self.assertTrue(np.all(np.isfinite(phi)))
        self.assertGreater(phi[0, F.F_S + F.P['helping_hand_damage']], 0)
        psi = F.state_summary(b)['psi']
        self.assertEqual(psi.shape, (F.D_VALUE,))
        self.assertTrue(np.all(np.isfinite(psi)))


if __name__ == '__main__':
    unittest.main()
