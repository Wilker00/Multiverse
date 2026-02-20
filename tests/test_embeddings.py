import unittest

from memory.embeddings import cosine_similarity, obs_to_universal_vector, obs_to_vector, project_vector


class TestEmbeddings(unittest.TestCase):
    def test_obs_to_vector_flattens_nested_dict_and_list(self):
        obs = {
            "b": [2, {"x": 3}],
            "a": 1,
        }
        self.assertEqual(obs_to_vector(obs), [1.0, 2.0, 3.0])

    def test_obs_to_vector_accepts_top_level_tuple(self):
        self.assertEqual(obs_to_vector((1, [2, 3])), [1.0, 2.0, 3.0])

    def test_obs_to_vector_rejects_bool(self):
        with self.assertRaises(TypeError):
            obs_to_vector({"x": True})

    def test_project_vector_normalizes_and_is_fixed_dim(self):
        v = project_vector([1.0, 2.0, 3.0, 4.0], dim=32)
        self.assertEqual(len(v), 32)
        norm = sum(x * x for x in v) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_obs_to_universal_vector_handles_dimension_drift(self):
        old_obs = {"a": 1, "b": [2, 3]}
        new_obs = {"a": 1, "b": [2, 3], "extra": [0, 0]}
        u1 = obs_to_universal_vector(old_obs, dim=64)
        u2 = obs_to_universal_vector(new_obs, dim=64)
        self.assertEqual(len(u1), 64)
        self.assertEqual(len(u2), 64)
        # Added zero-valued fields should preserve high similarity in projected space.
        self.assertGreater(cosine_similarity(u1, u2), 0.95)


if __name__ == "__main__":
    unittest.main()
