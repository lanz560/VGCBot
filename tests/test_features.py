import unittest
from types import SimpleNamespace

from poke_env.battle import PokemonType
from poke_env.battle.field import Field
from poke_env.battle.move import Move

from bot.features import _effective_base_power, _effective_move


class EffectiveBasePowerTests(unittest.TestCase):
    def test_eruption_power_scales_with_current_hp(self):
        eruption = SimpleNamespace(id="eruption", base_power=150, category=None)

        self.assertEqual(
            _effective_base_power(
                SimpleNamespace(current_hp_fraction=1.0), eruption
            ),
            150,
        )
        self.assertEqual(
            _effective_base_power(
                SimpleNamespace(current_hp_fraction=0.5), eruption
            ),
            75,
        )

    def test_eruption_power_is_at_least_one(self):
        eruption = SimpleNamespace(id="eruption", base_power=150, category=None)

        self.assertEqual(
            _effective_base_power(
                SimpleNamespace(current_hp_fraction=0.0), eruption
            ),
            1,
        )

    def test_other_moves_keep_existing_base_power(self):
        move = SimpleNamespace(id="heatwave", base_power=95, category=None)

        self.assertEqual(
            _effective_base_power(
                SimpleNamespace(current_hp_fraction=0.5), move
            ),
            95,
        )

    def test_terrain_pulse_changes_type_and_doubles_power(self):
        terrain_types = (
            (Field.ELECTRIC_TERRAIN, PokemonType.ELECTRIC, 1.3),
            (Field.GRASSY_TERRAIN, PokemonType.GRASS, 1.3),
            (Field.MISTY_TERRAIN, PokemonType.FAIRY, 1.0),
            (Field.PSYCHIC_TERRAIN, PokemonType.PSYCHIC, 1.3),
        )
        user = SimpleNamespace(
            current_hp_fraction=1.0, types=(), ability=None, item=None
        )
        move = Move("terrainpulse", 9)

        for terrain, expected_type, expected_multiplier in terrain_types:
            with self.subTest(terrain=terrain):
                battle = SimpleNamespace(fields={terrain})
                bp, move_type, multiplier = _effective_move(
                    user, move, user, battle
                )
                self.assertEqual(bp, 100)
                self.assertEqual(move_type, expected_type)
                self.assertEqual(multiplier, expected_multiplier)

    def test_expanding_force_and_rising_voltage_use_terrain_rules(self):
        grounded = SimpleNamespace(
            current_hp_fraction=1.0, types=(), ability=None, item=None
        )
        psychic_battle = SimpleNamespace(fields={Field.PSYCHIC_TERRAIN})
        bp, move_type, multiplier = _effective_move(
            grounded, Move("expandingforce", 9), grounded, psychic_battle
        )
        self.assertEqual((bp, move_type, multiplier), (120, PokemonType.PSYCHIC, 1.3))

        electric_battle = SimpleNamespace(fields={Field.ELECTRIC_TERRAIN})
        bp, move_type, multiplier = _effective_move(
            grounded, Move("risingvoltage", 9), grounded, electric_battle
        )
        self.assertEqual((bp, move_type, multiplier), (140, PokemonType.ELECTRIC, 1.3))

        airborne = SimpleNamespace(
            current_hp_fraction=1.0,
            types=(PokemonType.FLYING,),
            ability=None,
            item=None,
        )
        bp, _, _ = _effective_move(
            grounded, Move("risingvoltage", 9), airborne, electric_battle
        )
        self.assertEqual(bp, 70)

    def test_misty_explosion_is_boosted_only_for_grounded_user_in_misty_terrain(self):
        grounded = SimpleNamespace(
            current_hp_fraction=1.0, types=(), ability=None, item=None
        )
        move = Move("mistyexplosion", 9)
        battle = SimpleNamespace(fields={Field.MISTY_TERRAIN})
        bp, move_type, multiplier = _effective_move(grounded, move, grounded, battle)
        self.assertEqual((bp, move_type, multiplier), (150, PokemonType.FAIRY, 1.0))

        airborne = SimpleNamespace(
            current_hp_fraction=1.0,
            types=(PokemonType.FLYING,),
            ability=None,
            item=None,
        )
        bp, _, _ = _effective_move(airborne, move, airborne, battle)
        self.assertEqual(bp, 100)


if __name__ == "__main__":
    unittest.main()
