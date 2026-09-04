from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import torch

from experiments.prob_head_structure_full_v1 import distributions as distribution_module
from experiments.prob_head_structure_full_v1.distributions import (
    NegativeBinomialDistribution,
    ShiftedHurdleNegativeBinomialDistribution,
    TweedieDistribution,
    pooled_cdf_quantile,
    simplex_grid,
)


class DistributionTests(unittest.TestCase):
    def test_negative_binomial_has_exact_mass_mean_and_variance(self):
        """Catches a parameterization other than NB(mean=mu, total_count=r)."""
        dist = NegativeBinomialDistribution(
            mu=torch.tensor([[2.0, 4.0]]), r=torch.tensor([[3.0, 2.0]])
        )
        y = torch.tensor([[0.0, 3.0]])
        expected = (
            torch.lgamma(y + dist.r)
            - torch.lgamma(dist.r)
            - torch.lgamma(y + 1)
            + dist.r * torch.log(dist.r / (dist.r + dist.mu))
            + y * torch.log(dist.mu / (dist.r + dist.mu))
        )
        self.assertTrue(torch.allclose(dist.log_prob(y), expected, atol=1e-6))
        self.assertTrue(torch.equal(dist.mean(), dist.mu))
        self.assertTrue(
            torch.allclose(dist.variance(), dist.mu + dist.mu.square() / dist.r)
        )
        self.assertTrue(torch.allclose(dist.p_zero(), torch.exp(dist.log_prob(torch.zeros_like(y)))))

    def test_negative_binomial_mass_and_seeded_moments_match_independent_contract(self):
        """Catches a self-consistent accessor formula whose PMF or sampler is not normalized."""
        mu = torch.tensor([[0.3, 2.0, 10.0]], dtype=torch.float64)
        r = torch.tensor([[0.7, 3.0, 1.5]], dtype=torch.float64)
        dist = NegativeBinomialDistribution(mu, r)
        support = torch.arange(0, 2001, dtype=torch.float64)[:, None, None]
        mass = torch.exp(dist.log_prob(support)).sum(dim=0)
        self.assertTrue(torch.allclose(mass, torch.ones_like(mass), atol=1e-10, rtol=0.0))

        samples = dist.sample(200_000, seed=2026090541)
        expected_variance = mu + mu.square() / r
        mean_error_bound = 6.0 * torch.sqrt(expected_variance / samples.shape[0]) + 0.01
        self.assertTrue(torch.all(torch.abs(samples.mean(dim=0) - mu) <= mean_error_bound))
        empirical_variance = samples.var(dim=0, unbiased=True)
        self.assertTrue(torch.all(torch.abs(empirical_variance / expected_variance - 1.0) < 0.05))

    def test_shifted_hurdle_nb_has_atom_and_shifted_positive_support(self):
        """Catches mixing a zero-truncated NB with the specified 1+NB tail."""
        dist = ShiftedHurdleNegativeBinomialDistribution(
            pi=torch.tensor([[0.7]]), mu=torch.tensor([[2.0]]), r=torch.tensor([[3.0]])
        )
        self.assertTrue(torch.allclose(dist.p_zero(), torch.tensor([[0.3]])))
        self.assertTrue(torch.allclose(dist.mean(), torch.tensor([[2.1]])))
        self.assertTrue(torch.allclose(dist.log_prob(torch.zeros((1, 1))), torch.log(torch.tensor([[0.3]]))))
        expected_one = math.log(0.7) + float(
            NegativeBinomialDistribution(torch.tensor([[2.0]]), torch.tensor([[3.0]])).log_prob(torch.zeros((1, 1)))
        )
        self.assertTrue(torch.allclose(dist.log_prob(torch.ones((1, 1))), torch.tensor([[expected_one]]), atol=1e-6))

    def test_shifted_hurdle_nb_total_mass_sample_support_and_empirical_mean(self):
        """Catches an unnormalized hurdle mixture or sampler that emits the wrong support."""
        pi = torch.tensor([[0.2, 0.6, 0.9]], dtype=torch.float64)
        mu = torch.tensor([[0.5, 3.0, 10.0]], dtype=torch.float64)
        r = torch.tensor([[2.0, 0.8, 5.0]], dtype=torch.float64)
        dist = ShiftedHurdleNegativeBinomialDistribution(pi, mu, r)
        support = torch.arange(0, 2001, dtype=torch.float64)[:, None, None]
        mass = torch.exp(dist.log_prob(support)).sum(dim=0)
        self.assertTrue(torch.allclose(mass, torch.ones_like(mass), atol=1e-10, rtol=0.0))

        samples = dist.sample(200_000, seed=2026090541)
        self.assertTrue(torch.all((samples == 0) | ((samples >= 1) & (samples == torch.floor(samples)))))
        self.assertTrue(torch.all(torch.abs((samples == 0).double().mean(dim=0) - (1 - pi)) < 0.005))
        expected_mean = pi * (1 + mu)
        mean_error_bound = 6.0 * torch.sqrt(dist.variance() / samples.shape[0]) + 0.01
        self.assertTrue(torch.all(torch.abs(samples.mean(dim=0) - expected_mean) <= mean_error_bound))

    def test_common_cdf_quantile_and_seeded_sampling_interface(self):
        """Catches distributions that cannot produce monotone CDF/quantiles or reproducible draws."""
        distributions = [
            NegativeBinomialDistribution(torch.tensor([[2.0]]), torch.tensor([[3.0]])),
            ShiftedHurdleNegativeBinomialDistribution(torch.tensor([[0.6]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])),
            TweedieDistribution(torch.tensor([[2.0]]), torch.tensor([[0.8]]), torch.tensor([[1.5]])),
        ]
        y = torch.tensor([[[0.0]], [[1.0]], [[3.0]], [[8.0]]])
        q = torch.tensor([0.01, 0.25, 0.5, 0.9, 0.99])
        for dist in distributions:
            cdf = dist.cdf(y)
            self.assertTrue(torch.all(cdf[1:] >= cdf[:-1]))
            quantiles = dist.quantile(q)
            self.assertTrue(torch.all(quantiles[1:] >= quantiles[:-1]))
            self.assertTrue(torch.all(dist.cdf(quantiles) >= q[:, None, None] - 2e-4))
            self.assertTrue(torch.equal(dist.sample(11, seed=2026090541), dist.sample(11, seed=2026090541)))
            self.assertEqual(tuple(dist.sample(3, seed=1).shape), (3, 1, 1))

    def test_cdf_scalar_and_query_grid_broadcast_over_batch_horizon(self):
        """Catches support-axis construction from an unexpanded scalar CDF query."""
        mu = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=torch.float64)
        r = torch.full_like(mu, 2.5)
        distributions = (
            NegativeBinomialDistribution(mu, r),
            ShiftedHurdleNegativeBinomialDistribution(
                torch.full_like(mu, 0.6), mu, r
            ),
            TweedieDistribution(mu, torch.full_like(mu, 0.8), torch.full_like(mu, 1.5)),
        )
        query_grid = torch.tensor([0.0, 1.0, 5.0, 20.0], dtype=torch.float64)
        for distribution in distributions:
            scalar = distribution.cdf(5.0)
            grid = distribution.cdf(query_grid)
            explicit = distribution.cdf(query_grid[:, None, None])
            self.assertEqual(tuple(scalar.shape), (2, 3))
            self.assertEqual(tuple(grid.shape), (4, 2, 3))
            torch.testing.assert_close(grid, explicit)
            self.assertTrue(bool((grid[1:] >= grid[:-1]).all()))

    def test_tweedie_has_compound_poisson_zero_atom(self):
        """Catches an implementation that treats a 1<p<2 Tweedie as fully continuous."""
        mu = torch.tensor([[2.0]], dtype=torch.float64)
        phi = torch.tensor([[0.75]], dtype=torch.float64)
        p = torch.tensor([[1.5]], dtype=torch.float64)
        dist = TweedieDistribution(mu, phi, p)
        expected = torch.exp(-mu.pow(2 - p) / (phi * (2 - p)))
        self.assertTrue(torch.allclose(dist.p_zero(), expected, rtol=1e-10, atol=1e-12))
        self.assertTrue(torch.allclose(dist.log_prob(torch.zeros_like(mu)), torch.log(expected), rtol=1e-10, atol=1e-12))

    def test_tweedie_mixed_zero_and_positive_values_have_finite_parameter_gradients(self):
        """Catches masked log(0) work that poisons a full Tweedie likelihood gradient."""
        mu = torch.tensor([[2.0, 3.0]], dtype=torch.float64, requires_grad=True)
        phi = torch.tensor([[0.7, 1.1]], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([[1.3, 1.7]], dtype=torch.float64, requires_grad=True)
        loss = -TweedieDistribution(mu, phi, p).log_prob(torch.tensor([[0.0, 1.0]], dtype=torch.float64)).sum()
        loss.backward()
        for gradient in (mu.grad, phi.grad, p.grad):
            self.assertTrue(torch.isfinite(gradient).all())

    def test_tweedie_tail_density_uses_y_aware_series_terms(self):
        """Catches a rate-only truncation before the positive-density series mode."""
        dist = TweedieDistribution(
            torch.tensor([[20.0]], dtype=torch.float64),
            torch.tensor([[0.1]], dtype=torch.float64),
            torch.tensor([[1.1]], dtype=torch.float64),
        )
        self.assertAlmostEqual(float(dist.log_prob(torch.tensor([[100.0]], dtype=torch.float64))), -567.7045020410478, places=5)

    def test_quantile_endpoints_and_zero_atom_are_mathematically_defined(self):
        """Catches bisection artifacts at q=0, q=p_zero, and q=1."""
        distributions = [
            NegativeBinomialDistribution(torch.tensor([[2.0]]), torch.tensor([[3.0]])),
            ShiftedHurdleNegativeBinomialDistribution(torch.tensor([[0.6]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])),
            TweedieDistribution(torch.tensor([[2.0]]), torch.tensor([[0.8]]), torch.tensor([[1.5]])),
        ]
        for dist in distributions:
            q = torch.cat((torch.tensor([0.0]), dist.p_zero().reshape(-1), torch.tensor([1.0])))
            actual = dist.quantile(q).reshape(-1)
            self.assertEqual(float(actual[0]), 0.0)
            self.assertEqual(float(actual[1]), 0.0)
            self.assertTrue(torch.isinf(actual[2]) and actual[2] > 0)

    def test_discrete_quantiles_are_smallest_integer_cdf_crossings(self):
        """Catches applying ceil to a continuous NB/HSNB bisection point, which is one count too high."""
        fixtures = [
            (NegativeBinomialDistribution(torch.tensor([[2.0]]), torch.tensor([[3.0]])), torch.tensor([0.30]), 1.0),
            (ShiftedHurdleNegativeBinomialDistribution(torch.tensor([[0.6]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])), torch.tensor([0.50]), 1.0),
        ]
        for distribution, probabilities, expected in fixtures:
            quantiles = distribution.quantile(probabilities)
            self.assertEqual(float(quantiles.reshape(-1)[0]), expected)
            self.assertTrue(torch.all(distribution.cdf(quantiles) >= probabilities[:, None, None]))
            self.assertTrue(torch.all(distribution.cdf(quantiles - 1) < probabilities[:, None, None]))

    def test_cdf_infinite_endpoints_compose_with_q_one_quantiles_for_all_heads_and_precisions(self):
        """Catches flooring +inf into an integer sentinel before the NB/HSNB CDF endpoint is handled."""
        for dtype in (torch.float32, torch.float64):
            distributions = [
                NegativeBinomialDistribution(torch.tensor([[2.0]], dtype=dtype), torch.tensor([[3.0]], dtype=dtype)),
                ShiftedHurdleNegativeBinomialDistribution(torch.tensor([[0.6]], dtype=dtype), torch.tensor([[2.0]], dtype=dtype), torch.tensor([[3.0]], dtype=dtype)),
                TweedieDistribution(torch.tensor([[2.0]], dtype=dtype), torch.tensor([[0.8]], dtype=dtype), torch.tensor([[1.5]], dtype=dtype)),
            ]
            endpoints = torch.tensor([[[-torch.inf]], [[torch.inf]]], dtype=dtype)
            for distribution in distributions:
                self.assertTrue(torch.equal(distribution.cdf(endpoints).reshape(-1), torch.tensor([0.0, 1.0], dtype=dtype)))
                self.assertTrue(torch.equal(distribution.cdf(distribution.quantile(torch.tensor([1.0], dtype=dtype))).reshape(-1), torch.ones(1, dtype=dtype)))

    def test_nb_cdf_rejects_huge_finite_counts_before_integer_overflow(self):
        """Catches casting a very large finite CDF argument to the negative integer sentinel used for +inf."""
        dist = NegativeBinomialDistribution(torch.tensor([[2.0]]), torch.tensor([[3.0]]))
        with self.assertRaisesRegex(RuntimeError, "resource guard"):
            dist.cdf(torch.tensor([[1e30]]))

    def test_nb_cdf_does_not_allocate_a_support_by_query_tensor(self):
        """Catches O(max_count*queries) support expansion during pool inversion."""
        dist = NegativeBinomialDistribution(
            torch.full((448, 1), 50.0, dtype=torch.float64),
            torch.full((448, 1), 2.0, dtype=torch.float64),
        )
        queries = torch.full((1, 448, 1), 20_000.0, dtype=torch.float64)
        with patch.object(
            distribution_module.torch,
            "arange",
            side_effect=AssertionError("support allocation is forbidden"),
        ):
            result = dist.cdf(queries)
        self.assertEqual(tuple(result.shape), (1, 448, 1))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(((result >= 0.0) & (result <= 1.0)).all())

    def test_nb_and_hsnb_cdf_stay_monotone_in_unit_interval_at_float32_extremes(self):
        """Catches float32 PMF accumulation above one propagating into CDF pools."""
        for dtype in (torch.float32, torch.float64):
            mu = torch.tensor([[0.1, 20.0, 1000.0]], dtype=dtype)
            r = torch.tensor([[100.0, 10.0, 1000.0]], dtype=dtype)
            variance = mu + mu.square() / r
            queries = torch.stack(
                [
                    torch.zeros_like(mu),
                    torch.floor(mu),
                    torch.ceil(mu + 20.0 * torch.sqrt(variance)),
                ],
                dim=0,
            )
            for distribution in (
                NegativeBinomialDistribution(mu, r),
                ShiftedHurdleNegativeBinomialDistribution(
                    torch.tensor([[0.2, 0.6, 0.9]], dtype=dtype), mu, r
                ),
            ):
                cdf = distribution.cdf(queries)
                self.assertTrue(torch.isfinite(cdf).all())
                self.assertTrue(((cdf >= 0) & (cdf <= 1)).all())
                self.assertTrue((cdf[1:] >= cdf[:-1]).all())
                self.assertTrue(
                    torch.allclose(
                        cdf[-1], torch.ones_like(cdf[-1]), atol=2e-6, rtol=0.0
                    )
                )

    def test_cdf_roundoff_guard_rejects_material_probability_error(self):
        """Catches an unconditional clamp hiding a broken mass calculation."""
        within = torch.tensor([-1e-12, 1.0 + 1e-12], dtype=torch.float64)
        checked = distribution_module._checked_cdf_probability(within, family="NB")
        torch.testing.assert_close(checked, torch.tensor([0.0, 1.0], dtype=torch.float64))
        for broken in (
            torch.tensor([-1e-4], dtype=torch.float64),
            torch.tensor([1.0001], dtype=torch.float64),
            torch.tensor([float("nan")], dtype=torch.float64),
        ):
            with self.assertRaisesRegex(RuntimeError, "CDF probability error"):
                distribution_module._checked_cdf_probability(broken, family="HSNB")

    def test_constructor_rejects_mixed_parameter_dtypes(self):
        """Catches implicit type promotion that makes the predictive-head contract ambiguous."""
        with self.assertRaisesRegex(ValueError, "dtype"):
            NegativeBinomialDistribution(torch.tensor([[1.0]], dtype=torch.float32), torch.tensor([[1.0]], dtype=torch.float64))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable on this validation host")
    def test_constructor_rejects_mixed_parameter_devices(self):
        """Catches a CPU/CUDA parameter mix being deferred to an opaque tensor-operation error."""
        with self.assertRaisesRegex(ValueError, "device"):
            NegativeBinomialDistribution(torch.tensor([[1.0]]), torch.tensor([[1.0]], device="cuda"))

    def test_tweedie_float32_tail_gradients_are_finite(self):
        """Catches a float64 internal summation that disconnects or destabilizes float32 training gradients."""
        mu = torch.tensor([[20.0]], dtype=torch.float32, requires_grad=True)
        phi = torch.tensor([[0.1]], dtype=torch.float32, requires_grad=True)
        p = torch.tensor([[1.1]], dtype=torch.float32, requires_grad=True)
        (-TweedieDistribution(mu, phi, p).log_prob(torch.tensor([[100.0]], dtype=torch.float32))).backward()
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in (mu.grad, phi.grad, p.grad)))

    def test_simplex_grid_and_pooled_cdf_inverse_are_complete_and_monotone(self):
        """Catches incomplete 0.1-simplex enumeration or a non-inverting pooled CDF helper."""
        weights = simplex_grid(3, step=0.5)
        self.assertEqual(tuple(weights.shape), (6, 3))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(6, dtype=weights.dtype)))
        cdfs = torch.tensor([[[0.3], [0.7], [1.0]], [[0.5], [0.9], [1.0]]])
        y = torch.tensor([0.0, 1.0, 2.0])
        quantiles = pooled_cdf_quantile(cdfs, y, torch.tensor([0.25, 0.5, 0.8]), torch.tensor([0.5, 0.5]))
        self.assertTrue(torch.all(quantiles[1:] >= quantiles[:-1]))
        self.assertTrue(torch.equal(quantiles.reshape(-1), torch.tensor([0.0, 1.0, 2.0])))

    def test_pooled_cdf_inverse_rejects_invalid_support_or_probabilities(self):
        """Catches invalid mixture CDFs being silently inverted as if they were probabilities."""
        valid = torch.tensor([[[0.2], [1.0]], [[0.4], [1.0]]])
        with self.assertRaises(ValueError):
            pooled_cdf_quantile(valid, torch.tensor([1.0, 1.0]), torch.tensor([0.5]), torch.tensor([0.5, 0.5]))
        with self.assertRaises(ValueError):
            pooled_cdf_quantile(torch.tensor([[[0.2], [1.1]], [[0.4], [1.0]]]), torch.tensor([0.0, 1.0]), torch.tensor([0.5]), torch.tensor([0.5, 0.5]))
        with self.assertRaises(ValueError):
            pooled_cdf_quantile(torch.tensor([[[0.8], [0.7]], [[0.4], [1.0]]]), torch.tensor([0.0, 1.0]), torch.tensor([0.5]), torch.tensor([0.5, 0.5]))
        for invalid_support in (
            torch.tensor([0.0, 0.5]),
            torch.tensor([0.0, 2.0]),
            torch.tensor([1.0, 2.0]),
        ):
            with self.assertRaisesRegex(ValueError, "complete count support"):
                pooled_cdf_quantile(
                    valid,
                    invalid_support,
                    torch.tensor([0.5]),
                    torch.tensor([0.5, 0.5]),
                )
        with self.assertRaisesRegex(ValueError, "simplex"):
            pooled_cdf_quantile(
                valid,
                torch.tensor([0.0, 1.0]),
                torch.tensor([0.5]),
                torch.tensor([0.50001, 0.5]),
            )
        with self.assertRaisesRegex(ValueError, "terminal CDF mass"):
            pooled_cdf_quantile(
                torch.tensor([[[0.2], [0.99]], [[0.4], [1.0]]]),
                torch.tensor([0.0, 1.0]),
                torch.tensor([0.5]),
                torch.tensor([0.5, 0.5]),
            )

    def test_cdf_and_quantile_queries_reject_nan_instead_of_returning_finite_values(self):
        """Catches corrupted query values being disguised as valid probabilities or quantiles."""
        distributions = [
            NegativeBinomialDistribution(torch.tensor([[2.0]]), torch.tensor([[3.0]])),
            ShiftedHurdleNegativeBinomialDistribution(
                torch.tensor([[0.6]]), torch.tensor([[2.0]]), torch.tensor([[3.0]])
            ),
            TweedieDistribution(
                torch.tensor([[2.0]]), torch.tensor([[0.8]]), torch.tensor([[1.5]])
            ),
        ]
        for distribution in distributions:
            with self.assertRaisesRegex(ValueError, "finite"):
                distribution.cdf(torch.tensor([[float("nan")]]))
            with self.assertRaisesRegex(ValueError, "finite"):
                distribution.quantile(torch.tensor([float("nan")]))

        valid_cdfs = torch.tensor([[[0.2], [1.0]], [[0.4], [1.0]]])
        with self.assertRaisesRegex(ValueError, "finite"):
            pooled_cdf_quantile(
                valid_cdfs,
                torch.tensor([0.0, 1.0]),
                torch.tensor([float("nan")]),
                torch.tensor([0.5, 0.5]),
            )

    def test_invalid_parameters_and_support_are_rejected(self):
        """Catches silent acceptance of invalid distribution parameters or non-[B,H] inputs."""
        with self.assertRaises(ValueError):
            NegativeBinomialDistribution(torch.tensor([1.0]), torch.tensor([1.0]))
        with self.assertRaises(ValueError):
            NegativeBinomialDistribution(torch.tensor([[0.0]]), torch.tensor([[1.0]]))
        with self.assertRaises(ValueError):
            ShiftedHurdleNegativeBinomialDistribution(torch.tensor([[1.1]]), torch.tensor([[1.0]]), torch.tensor([[1.0]]))
        with self.assertRaises(ValueError):
            TweedieDistribution(torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([[2.0]]))
        with self.assertRaises(ValueError):
            TweedieDistribution(torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([[1.01]]))
        with self.assertRaises(ValueError):
            TweedieDistribution(torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([[1.99]]))


if __name__ == "__main__":
    unittest.main()
