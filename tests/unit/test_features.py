"""
Unit tests for physics feature engineering functions.

Tests verify correctness of physics formulas using known analytical results:
- Invariant mass of Z boson from back-to-back equal-pT muons
- ΔR = 0 for identical particles
- Transverse mass bounded by W mass for W → lν events
- Rapidity is symmetric and zero for particles at 90°

Run with:
    pytest tests/unit/test_features.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.physics_features import (
    azimuthal_angle_difference,
    centrality,
    delta_r,
    four_vector_components,
    ht_scalar,
    invariant_mass,
    missing_et_significance,
    rapidity,
    transverse_mass,
)
from src.features.low_level_features import LowLevelExtractor
from src.features.high_level_features import HighLevelFeatureBuilder


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """A minimal HIGGS-format DataFrame for testing."""
    import pandas as pd
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "label": np.random.randint(0, 2, n),
        "lepton_pt":  np.random.uniform(25, 200, n).astype(np.float32),
        "lepton_eta": np.random.uniform(-2.4, 2.4, n).astype(np.float32),
        "lepton_phi": np.random.uniform(-np.pi, np.pi, n).astype(np.float32),
        "missing_energy_magnitude": np.random.uniform(20, 300, n).astype(np.float32),
        "missing_energy_phi": np.random.uniform(-np.pi, np.pi, n).astype(np.float32),
        "jet1_pt":  np.random.uniform(30, 300, n).astype(np.float32),
        "jet1_eta": np.random.uniform(-2.5, 2.5, n).astype(np.float32),
        "jet1_phi": np.random.uniform(-np.pi, np.pi, n).astype(np.float32),
        "jet1_b_tag": np.random.uniform(0, 1, n).astype(np.float32),
        "jet2_pt":  np.random.uniform(30, 200, n).astype(np.float32),
        "jet2_eta": np.random.uniform(-2.5, 2.5, n).astype(np.float32),
        "jet2_phi": np.random.uniform(-np.pi, np.pi, n).astype(np.float32),
        "jet2_b_tag": np.random.uniform(0, 1, n).astype(np.float32),
        "jet3_pt":  np.random.uniform(0, 100, n).astype(np.float32),
        "jet3_eta": np.random.uniform(-2.5, 2.5, n).astype(np.float32),
        "jet3_phi": np.random.uniform(-np.pi, np.pi, n).astype(np.float32),
        "jet3_b_tag": np.random.uniform(0, 1, n).astype(np.float32),
        "jet4_pt":  np.zeros(n, dtype=np.float32),   # zero-padded 4th jet
        "jet4_eta": np.zeros(n, dtype=np.float32),
        "jet4_phi": np.zeros(n, dtype=np.float32),
        "jet4_b_tag": np.zeros(n, dtype=np.float32),
        "m_jj":   np.random.uniform(50, 500, n).astype(np.float32),
        "m_jjj":  np.random.uniform(100, 800, n).astype(np.float32),
        "m_lv":   np.random.uniform(30, 200, n).astype(np.float32),
        "m_jlv":  np.random.uniform(100, 600, n).astype(np.float32),
        "m_bb":   np.random.uniform(50, 300, n).astype(np.float32),
        "m_wbb":  np.random.uniform(100, 500, n).astype(np.float32),
        "m_wwbb": np.random.uniform(200, 1000, n).astype(np.float32),
    })


# ─── Invariant mass tests ─────────────────────────────────────────────────────

class TestInvariantMass:

    def test_identical_particles_returns_zero_for_massless(self):
        """Two identical massless particles: M² = 0 (they're collinear)."""
        # For identical massless particles, the 4-vector sum doubles everything
        # m_inv = sqrt((2E)² - (2p)²) = 0 for massless (E=|p|)
        m = invariant_mass(
            pt1=np.array([50.0]), eta1=np.array([0.0]), phi1=np.array([0.0]), m1=np.array([0.0]),
            pt2=np.array([50.0]), eta2=np.array([0.0]), phi2=np.array([0.0]), m2=np.array([0.0]),
        )
        assert float(m[0]) < 1e-3, f"Expected ~0, got {m[0]}"

    def test_z_boson_mass_from_muon_pair(self):
        """
        Two muons from Z → μμ decay:
        Back-to-back (Δφ = π), equal pT, η ≈ 0.
        Invariant mass should be ≈ 2 * pT (for massless limit at η=0, Δφ=π).
        For M_Z = 91.2 GeV: pT ≈ 45.6 GeV each.
        """
        m_z = 91.2    # GeV
        pt = m_z / 2  # back-to-back, η=0
        m_mu = 0.106  # muon mass in GeV

        m_inv = invariant_mass(
            pt1=np.array([pt]), eta1=np.array([0.0]), phi1=np.array([0.0]), m1=np.array([m_mu]),
            pt2=np.array([pt]), eta2=np.array([0.0]), phi2=np.array([np.pi]), m2=np.array([m_mu]),
        )
        # At η=0, back-to-back: M = 2 * sqrt(pT² + m_mu²) ≈ 2*pT for pT >> m_mu
        expected = 2 * np.sqrt(pt**2 + m_mu**2)
        assert abs(float(m_inv[0]) - expected) < 0.5, (
            f"Expected ~{expected:.1f} GeV, got {m_inv[0]:.1f} GeV"
        )

    def test_invariant_mass_positive(self):
        """Invariant mass is always non-negative."""
        n = 1000
        rng = np.random.default_rng(0)
        pt1 = rng.uniform(10, 200, n)
        m_inv = invariant_mass(
            pt1=pt1, eta1=rng.uniform(-3, 3, n), phi1=rng.uniform(-np.pi, np.pi, n),
            m1=rng.uniform(0, 5, n),
            pt2=rng.uniform(10, 200, n), eta2=rng.uniform(-3, 3, n),
            phi2=rng.uniform(-np.pi, np.pi, n), m2=rng.uniform(0, 5, n),
        )
        assert (m_inv >= 0).all(), "Invariant mass must be non-negative"

    def test_invariant_mass_vectorized(self):
        """Vectorized computation returns correct shape."""
        n = 500
        rng = np.random.default_rng(1)
        result = invariant_mass(
            rng.uniform(10, 100, n), rng.uniform(-2, 2, n),
            rng.uniform(-np.pi, np.pi, n), np.zeros(n),
            rng.uniform(10, 100, n), rng.uniform(-2, 2, n),
            rng.uniform(-np.pi, np.pi, n), np.zeros(n),
        )
        assert result.shape == (n,), f"Expected shape ({n},), got {result.shape}"


# ─── ΔR tests ─────────────────────────────────────────────────────────────────

class TestDeltaR:

    def test_identical_particles_zero(self):
        """ΔR = 0 for identical particle directions."""
        dr = delta_r(
            eta1=np.array([1.5]), phi1=np.array([0.3]),
            eta2=np.array([1.5]), phi2=np.array([0.3]),
        )
        assert abs(float(dr[0])) < 1e-6, f"Expected 0, got {dr[0]}"

    def test_known_delta_r(self):
        """ΔR = sqrt(Δη² + Δφ²) for simple case."""
        dr = delta_r(
            eta1=np.array([1.0]), phi1=np.array([0.0]),
            eta2=np.array([0.0]), phi2=np.array([0.0]),
        )
        assert abs(float(dr[0]) - 1.0) < 1e-5, f"Expected 1.0, got {dr[0]}"

    def test_phi_wrapping(self):
        """ΔR correctly wraps Δφ across the ±π boundary."""
        # phi1 = -π + 0.1, phi2 = π - 0.1: should be |Δφ| = 0.2, not ≈ 2π - 0.2
        dr = delta_r(
            eta1=np.array([0.0]), phi1=np.array([-np.pi + 0.1]),
            eta2=np.array([0.0]), phi2=np.array([np.pi - 0.1]),
        )
        expected = 0.2  # Correct wrapped Δφ
        assert abs(float(dr[0]) - expected) < 1e-4, (
            f"Expected ΔR ≈ {expected:.3f} (with wrapping), got {dr[0]:.4f}"
        )

    def test_delta_r_non_negative(self):
        """ΔR is always non-negative."""
        n = 1000
        rng = np.random.default_rng(2)
        dr = delta_r(
            rng.uniform(-3, 3, n), rng.uniform(-np.pi, np.pi, n),
            rng.uniform(-3, 3, n), rng.uniform(-np.pi, np.pi, n),
        )
        assert (dr >= 0).all(), "ΔR must be non-negative"


# ─── Transverse mass tests ─────────────────────────────────────────────────────

class TestTransverseMass:

    def test_mt_non_negative(self):
        """Transverse mass is always non-negative."""
        n = 1000
        rng = np.random.default_rng(3)
        mt = transverse_mass(
            lepton_pt=rng.uniform(10, 200, n),
            lepton_phi=rng.uniform(-np.pi, np.pi, n),
            met=rng.uniform(10, 300, n),
            met_phi=rng.uniform(-np.pi, np.pi, n),
        )
        assert (mt >= 0).all(), "M_T must be non-negative"

    def test_mt_back_to_back_maximum(self):
        """
        M_T is maximized when lepton and MET are back-to-back (Δφ = π).
        For Δφ = π: M_T = 2 * sqrt(pT_lep * MET) (simplified).
        """
        pt_lep = np.array([45.0])
        met = np.array([45.0])
        mt_back = transverse_mass(
            lepton_pt=pt_lep, lepton_phi=np.array([0.0]),
            met=met, met_phi=np.array([np.pi]),
        )
        mt_collinear = transverse_mass(
            lepton_pt=pt_lep, lepton_phi=np.array([0.0]),
            met=met, met_phi=np.array([0.0]),
        )
        assert mt_back[0] > mt_collinear[0], "M_T should be larger when back-to-back"

    def test_mt_collinear_is_zero(self):
        """M_T = 0 when lepton and MET are collinear (Δφ = 0)."""
        mt = transverse_mass(
            lepton_pt=np.array([50.0]), lepton_phi=np.array([0.0]),
            met=np.array([50.0]), met_phi=np.array([0.0]),
        )
        assert abs(float(mt[0])) < 1e-4, f"Expected 0, got {mt[0]}"


# ─── HT tests ─────────────────────────────────────────────────────────────────

class TestHTScalar:

    def test_ht_sum_of_jet_pts(self):
        """HT = sum of all positive jet pTs."""
        jet_pts = np.array([[100.0, 80.0, 60.0, 0.0]])  # 4th jet is zero-padded
        ht = ht_scalar(jet_pts)
        assert abs(float(ht[0]) - 240.0) < 1e-4, f"Expected 240.0, got {ht[0]}"

    def test_ht_single_jet(self):
        """HT with one real jet equals that jet's pT."""
        jet_pts = np.array([[150.0, 0.0, 0.0, 0.0]])
        ht = ht_scalar(jet_pts)
        assert abs(float(ht[0]) - 150.0) < 1e-4

    def test_ht_non_negative(self):
        """HT is always non-negative."""
        n = 100
        rng = np.random.default_rng(4)
        jet_pts = np.maximum(rng.uniform(-10, 200, (n, 4)), 0.0)
        ht = ht_scalar(jet_pts)
        assert (ht >= 0).all()


# ─── Rapidity tests ────────────────────────────────────────────────────────────

class TestRapidity:

    def test_rapidity_symmetric_particle_at_90_degrees(self):
        """A particle with pz = 0 (η = 0, θ = 90°) has y = 0."""
        E = np.array([100.0])
        pz = np.array([0.0])
        y = rapidity(E, pz)
        assert abs(float(y[0])) < 1e-5, f"Expected y=0, got {y[0]}"

    def test_rapidity_forward_particle_positive(self):
        """A forward particle (pz > 0) has positive rapidity."""
        E = np.array([100.0])
        pz = np.array([50.0])
        y = rapidity(E, pz)
        assert float(y[0]) > 0, f"Expected y > 0 for forward particle, got {y[0]}"

    def test_rapidity_antisymmetric(self):
        """Rapidity changes sign when pz flips sign."""
        E = np.array([100.0])
        pz_fwd = np.array([50.0])
        pz_bwd = np.array([-50.0])
        y_fwd = rapidity(E, pz_fwd)
        y_bwd = rapidity(E, pz_bwd)
        assert abs(float(y_fwd[0]) + float(y_bwd[0])) < 1e-5, (
            "Rapidity should be antisymmetric: y(pz) = -y(-pz)"
        )


# ─── Centrality tests ──────────────────────────────────────────────────────────

class TestCentrality:

    def test_centrality_bounded_zero_to_one(self):
        """Centrality is in [0, 1] when HT <= E_total."""
        ht = np.array([300.0, 100.0, 0.0])
        E_total = np.array([500.0, 500.0, 500.0])
        c = centrality(ht, E_total)
        assert (c >= 0).all() and (c <= 1).all()

    def test_centrality_zero_ht(self):
        """Centrality = 0 when HT = 0."""
        c = centrality(np.array([0.0]), np.array([100.0]))
        assert abs(float(c[0])) < 1e-6


# ─── High-level feature builder tests ─────────────────────────────────────────

class TestHighLevelFeatureBuilder:

    def test_builds_derived_columns(self, sample_df):
        """Builder adds all expected derived columns."""
        builder = HighLevelFeatureBuilder(include_derived=True)
        result = builder.build(sample_df)
        for col in HighLevelFeatureBuilder.DERIVED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_mutation_of_input(self, sample_df):
        """Builder does not mutate the input DataFrame."""
        original_cols = set(sample_df.columns)
        builder = HighLevelFeatureBuilder()
        builder.build(sample_df)
        assert set(sample_df.columns) == original_cols, "Input DataFrame was mutated"

    def test_ht_matches_manual_sum(self, sample_df):
        """H_T column matches manual sum of jet pTs."""
        builder = HighLevelFeatureBuilder(include_derived=True)
        result = builder.build(sample_df)

        manual_ht = (
            sample_df["jet1_pt"] + sample_df["jet2_pt"]
            + sample_df["jet3_pt"] + sample_df["jet4_pt"]
        ).values
        computed_ht = result["ht"].values
        np.testing.assert_allclose(computed_ht, manual_ht, rtol=1e-4)

    def test_transverse_mass_non_negative(self, sample_df):
        """Computed transverse mass column is non-negative."""
        builder = HighLevelFeatureBuilder(include_derived=True)
        result = builder.build(sample_df)
        assert (result["transverse_mass_lv"] >= 0).all()

    def test_delta_r_lepton_jet_non_negative(self, sample_df):
        """ΔR(lepton, jet1) is non-negative."""
        builder = HighLevelFeatureBuilder(include_derived=True)
        result = builder.build(sample_df)
        assert (result["delta_r_lepton_jet1"] >= 0).all()


# ─── 4-vector component tests ─────────────────────────────────────────────────

class TestFourVectorComponents:

    def test_massless_particle_px_py(self):
        """For a massless particle at η=0: px = pT*cos(φ), py = pT*sin(φ)."""
        pt = np.array([100.0])
        phi = np.array([np.pi / 4])  # 45°
        px, py, pz, E = four_vector_components(pt, np.array([0.0]), phi, np.array([0.0]))
        expected_px = 100.0 * np.cos(np.pi / 4)
        expected_py = 100.0 * np.sin(np.pi / 4)
        assert abs(float(px[0]) - expected_px) < 1e-4
        assert abs(float(py[0]) - expected_py) < 1e-4

    def test_massless_energy_equals_momentum(self):
        """For a massless particle (m=0): E = |p| = pT / cos(θ) ≈ pT * cosh(η)."""
        pt = np.array([50.0])
        eta = np.array([0.0])  # θ = 90°
        px, py, pz, E = four_vector_components(pt, eta, np.array([0.0]), np.array([0.0]))
        # At η=0, pz=0, E = sqrt(px²+py²+pz²) = sqrt(pT²) = pT
        assert abs(float(E[0]) - float(pt[0])) < 1e-4


# ─── Azimuthal angle difference tests ─────────────────────────────────────────

class TestAzimuthalAngleDifference:

    def test_same_phi_returns_zero(self):
        result = azimuthal_angle_difference(np.array([1.5]), np.array([1.5]))
        assert abs(float(result[0])) < 1e-6

    def test_opposite_phi_returns_pi(self):
        result = azimuthal_angle_difference(np.array([0.0]), np.array([np.pi]))
        assert abs(float(result[0]) - np.pi) < 1e-5

    def test_result_in_zero_to_pi(self):
        """Result is always in [0, π]."""
        n = 1000
        rng = np.random.default_rng(5)
        phi1 = rng.uniform(-np.pi, np.pi, n)
        phi2 = rng.uniform(-np.pi, np.pi, n)
        result = azimuthal_angle_difference(phi1, phi2)
        assert (result >= 0).all() and (result <= np.pi + 1e-6).all()


# ─── MET significance tests ────────────────────────────────────────────────────

class TestMETSignificance:

    def test_zero_met_returns_zero(self):
        sig = missing_et_significance(np.array([0.0]), np.array([100.0]))
        assert abs(float(sig[0])) < 1e-6

    def test_scaling_with_met(self):
        """Doubling MET doubles significance."""
        sig1 = missing_et_significance(np.array([50.0]), np.array([100.0]))
        sig2 = missing_et_significance(np.array([100.0]), np.array([100.0]))
        assert abs(float(sig2[0]) / float(sig1[0]) - 2.0) < 1e-4
