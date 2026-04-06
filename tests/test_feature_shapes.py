import unittest

import pandas as pd

from src.build_features import build_features_and_labels


class TestFeatureBuilder(unittest.TestCase):
    def test_feature_and_label_rows(self):
        config = {
            "n_tuples": 5,
            "window_size": 4,
            "tiering": {"hot_capacity_ratio": 0.4},
        }

        access_log = pd.DataFrame(
            {
                "tx_id": [0, 1, 2, 3, 4, 5],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "tuple_id": [0, 1, 0, 2, 0, 3],
                "column_id": [0, 1, 2, 1, 0, 0],
            }
        )

        features, labels = build_features_and_labels(access_log, config)
        self.assertEqual(len(features), 5)
        self.assertEqual(len(labels), 5)


if __name__ == "__main__":
    unittest.main()
