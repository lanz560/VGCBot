import unittest
from types import SimpleNamespace

from bot.features import _effective_base_power


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


if __name__ == "__main__":
    unittest.main()
