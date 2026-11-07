import unittest

import numpy as np

from bot.features import F_S, S, _apply_helping_hand_bonus


class HelpingHandDamageTests(unittest.TestCase):
    def test_bonus_adds_spread_damage_and_new_ko_estimates(self):
        features = np.zeros(F_S)
        attack_aux = {
            "ko": 0,
            "dmg": 1.375,
            "damage_records": [
                (True, 0.4, 0.8, 1.0, 0),
                (True, 0.7, 0.8, 0.9, 1),
                (False, 0.2, 0.5, 1.0, 0),
            ],
        }

        updated_aux = _apply_helping_hand_bonus(features, attack_aux)

        self.assertAlmostEqual(features[S["dmg_frac"]], 0.3625)
        self.assertEqual(features[S["ko"]], 1)
        self.assertEqual(features[S["priority_ko"]], 1)
        self.assertAlmostEqual(features[S["ally_dmg"]], 0.2)
        self.assertEqual(updated_aux["ko"], 1)
        self.assertAlmostEqual(updated_aux["dmg"], 2.0625)
        self.assertEqual(attack_aux["ko"], 0)
        self.assertAlmostEqual(attack_aux["dmg"], 1.375)


if __name__ == "__main__":
    unittest.main()
