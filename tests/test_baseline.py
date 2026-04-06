import unittest

import pandas as pd

from src.baseline_hfa import predict_hot_cold_heuristic


class TestBaselineHeuristic(unittest.TestCase):
    def test_predict_hot_cold(self):
        df = pd.DataFrame(
            {
                "recent_access_count": [10, 2, 7],
                "recent_unique_columns": [4, 4, 1],
            }
        )

        pred = predict_hot_cold_heuristic(df, tuple_access_threshold=5, column_access_threshold=3)
        self.assertListEqual(pred.tolist(), [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
