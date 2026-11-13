import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sign_language.landmarks import FEATURE_SIZE, _landmarks_to_features
from sign_language.model import SignSequenceClassifier


class LandmarkTests(unittest.TestCase):
    def test_missing_landmarks_produce_zero_feature_vector(self):
        results = SimpleNamespace(
            pose_landmarks=None,
            left_hand_landmarks=None,
            right_hand_landmarks=None,
        )

        features = _landmarks_to_features(results)

        self.assertEqual(features.shape, (FEATURE_SIZE,))
        np.testing.assert_array_equal(features, np.zeros(FEATURE_SIZE, dtype=np.float32))


class ModelTests(unittest.TestCase):
    def test_classifier_returns_one_score_per_sign(self):
        model = SignSequenceClassifier(num_classes=4)
        model.eval()

        with torch.no_grad():
            scores = model(torch.zeros((2, 5, FEATURE_SIZE)))

        self.assertEqual(tuple(scores.shape), (2, 4))
        self.assertTrue(torch.isfinite(scores).all().item())


if __name__ == "__main__":
    unittest.main()
