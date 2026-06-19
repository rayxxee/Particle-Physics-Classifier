"""
Physics-aware feature engineering for particle physics events.

Implements the core kinematic observables used in High Energy Physics (HEP)
analyses. These computed features carry genuine discriminating power grounded
in the Standard Model — they are not arbitrary engineered features.

All functions operate on NumPy arrays for vectorized processing over full
datasets. Each function is documented with its physics motivation.

Reference:
    Peskin & Schroeder, "An Introduction to Quantum Field Theory"
    Baldi et al. (2014) — HIGGS dataset paper

Usage:
    from src.features.physics_features import (
        invariant_mass, delta_r, transverse_mass, rapidity, ht_scalar
    )

    m_inv = invariant_mass(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2)
"""

from __future__ import annotations

import numpy as np


# ─── 4-vector helpers ─────────────────────────────────────────────────────────

def four_vector_components(
    pt: np.ndarray,
    eta: np.ndarray,
    phi: np.ndarray,
    mass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (pT, eta, phi, mass) to Cartesian (px, py, pz, E).

    Physics:
        px = pT * cos(phi)
        py = pT * sin(phi)
        pz = pT * sinh(eta)
        E  = sqrt(px² + py² + pz² + m²)

    Args:
        pt:   Transverse momentum [GeV]
        eta:  Pseudorapidity (= -ln(tan(θ/2)))
        phi:  Azimuthal angle [rad]
        mass: Particle mass [GeV]

    Returns:
        (px, py, pz, E) arrays all in GeV.
    """
    pt = np.asarray(pt, dtype=np.float64)
    eta = np.asarray(eta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return px, py, pz, E


def invariant_mass(
    pt1: np.ndarray, eta1: np.ndarray, phi1: np.ndarray, m1: np.ndarray,
    pt2: np.ndarray, eta2: np.ndarray, phi2: np.ndarray, m2: np.ndarray,
) -> np.ndarray:
    """
    Compute the Lorentz-invariant mass of a two-particle system.

    Physics:
        M² = (E₁ + E₂)² - (p⃗₁ + p⃗₂)²

    This is the central observable in HEP — a peak in M_inv at the known
    particle mass (e.g., m_Z = 91.2 GeV, m_H = 125 GeV) is how particle
    discoveries are made. The Higgs discovery (2012) used this quantity.

    Returns:
        Invariant mass M [GeV]. Returns 0.0 for unphysical configurations
        (m² < 0 due to numerical precision).

    Example:
        # Two muons from Z → μμ should peak at ~91.2 GeV
        m_z = invariant_mass(mu1_pt, mu1_eta, mu1_phi, 0.106,
                             mu2_pt, mu2_eta, mu2_phi, 0.106)
    """
    px1, py1, pz1, E1 = four_vector_components(pt1, eta1, phi1, m1)
    px2, py2, pz2, E2 = four_vector_components(pt2, eta2, phi2, m2)

    # Sum of 4-vectors
    E_tot = E1 + E2
    px_tot = px1 + px2
    py_tot = py1 + py2
    pz_tot = pz1 + pz2

    # M² = E² - |p|²
    m_sq = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)

    # Guard against numerical noise (small negative m²)
    return np.sqrt(np.maximum(m_sq, 0.0)).astype(np.float32)


def delta_r(
    eta1: np.ndarray,
    phi1: np.ndarray,
    eta2: np.ndarray,
    phi2: np.ndarray,
) -> np.ndarray:
    """
    Compute ΔR — the angular separation between two particles in η-φ space.

    Physics:
        ΔR = √(Δη² + Δφ²)    where Δφ is wrapped to [-π, π]

    ΔR is the standard HEP measure of how "close" two objects are in the
    detector. It is approximately Lorentz-invariant under boosts along the
    beam axis. Used to:
    - Define jet cones (anti-kT jet algorithm uses R=0.4 or 0.8)
    - Measure lepton isolation (require no jet within ΔR < 0.4)
    - Build GNN edges (connect particles within ΔR < threshold)

    Returns:
        ΔR values [dimensionless], shape same as input arrays.

    Example:
        # ΔR = 0 for identical particles
        assert delta_r(1.0, 0.5, 1.0, 0.5) == 0.0
    """
    eta1 = np.asarray(eta1, dtype=np.float64)
    eta2 = np.asarray(eta2, dtype=np.float64)
    phi1 = np.asarray(phi1, dtype=np.float64)
    phi2 = np.asarray(phi2, dtype=np.float64)

    d_eta = eta1 - eta2
    d_phi = phi1 - phi2

    # Wrap Δφ to [-π, π] — angles are periodic
    d_phi = np.where(d_phi > np.pi, d_phi - 2 * np.pi, d_phi)
    d_phi = np.where(d_phi < -np.pi, d_phi + 2 * np.pi, d_phi)

    return np.sqrt(d_eta**2 + d_phi**2).astype(np.float32)


def transverse_mass(
    lepton_pt: np.ndarray,
    lepton_phi: np.ndarray,
    met: np.ndarray,
    met_phi: np.ndarray,
) -> np.ndarray:
    """
    Compute the transverse mass M_T of the lepton + neutrino system.

    Physics:
        M_T = √(2 · pT_lep · MET · (1 - cos(Δφ)))

    M_T is bounded above by the W boson mass (~80.4 GeV) for events where
    the lepton and neutrino come from W → lν decay. The "Jacobian peak" at
    M_T ≈ M_W is a classic W mass measurement technique.

    For Higgs events (H → WW*), the M_T distribution differs from QCD
    background, providing discriminating power.

    Returns:
        Transverse mass values [GeV].
    """
    lepton_pt = np.asarray(lepton_pt, dtype=np.float64)
    met = np.asarray(met, dtype=np.float64)
    d_phi = np.asarray(lepton_phi, dtype=np.float64) - np.asarray(met_phi, dtype=np.float64)

    # Wrap to [-π, π]
    d_phi = np.where(d_phi > np.pi, d_phi - 2 * np.pi, d_phi)
    d_phi = np.where(d_phi < -np.pi, d_phi + 2 * np.pi, d_phi)

    m_t_sq = 2 * lepton_pt * met * (1 - np.cos(d_phi))
    return np.sqrt(np.maximum(m_t_sq, 0.0)).astype(np.float32)


def rapidity(E: np.ndarray, pz: np.ndarray) -> np.ndarray:
    """
    Compute the rapidity y of a particle.

    Physics:
        y = (1/2) · ln((E + pz) / (E - pz))

    Rapidity is a Lorentz-covariant measure of a particle's "forward-ness"
    (how much it moves along the beam axis). Unlike the angle θ, rapidity
    differences Δy are invariant under longitudinal Lorentz boosts, making
    them physically meaningful at a hadron collider where the center-of-mass
    frame is not at rest in the lab.

    Pseudorapidity η ≈ y for massless particles. For massive particles,
    true rapidity is more physically correct.

    Returns:
        Rapidity values [dimensionless].
    """
    E = np.asarray(E, dtype=np.float64)
    pz = np.asarray(pz, dtype=np.float64)

    # Protect against division by zero (E = |pz| for massless particles
    # moving exactly along beam axis — very rare in practice)
    numerator = np.maximum(E + pz, 1e-10)
    denominator = np.maximum(E - pz, 1e-10)

    return (0.5 * np.log(numerator / denominator)).astype(np.float32)


def ht_scalar(jet_pts: np.ndarray) -> np.ndarray:
    """
    Compute H_T: scalar sum of jet transverse momenta.

    Physics:
        H_T = Σᵢ pT_i   (sum over all jets)

    H_T measures the overall "activity" or hardness of the collision event.
    High H_T events are typically associated with high-mass processes or
    multi-body final states. QCD events at a given center-of-mass energy
    tend to have lower H_T than signal processes producing heavy particles.

    Args:
        jet_pts: Array of shape (n_events, n_jets) with jet pT values [GeV].
                 Zero-padded jets (pT=0) are excluded from the sum.

    Returns:
        H_T values [GeV], shape (n_events,).
    """
    jet_pts = np.asarray(jet_pts, dtype=np.float64)

    # Handle both 1D (single event) and 2D (batch) inputs
    if jet_pts.ndim == 1:
        return np.sum(jet_pts[jet_pts > 0]).astype(np.float32)

    # Sum only jets with pT > 0 (non-padded)
    return np.sum(np.maximum(jet_pts, 0.0), axis=1).astype(np.float32)


def centrality(ht: np.ndarray, E_total: np.ndarray) -> np.ndarray:
    """
    Compute event centrality: ratio of H_T to total energy.

    Physics:
        C = H_T / E_total

    Centrality measures how "central" the event energy is in the transverse
    plane. Isotropic events (like decays of heavy particles at rest) have
    higher centrality than boosted events where most energy is along the beam.

    Returns:
        Centrality values [dimensionless, 0 to 1].
    """
    ht = np.asarray(ht, dtype=np.float64)
    E_total = np.asarray(E_total, dtype=np.float64)
    # Guard against division by zero
    return (ht / np.maximum(E_total, 1e-10)).astype(np.float32)


def azimuthal_angle_difference(phi1: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """
    Compute Δφ between two particles, wrapped to [0, π].

    Physics:
        |Δφ| = |φ₁ - φ₂|    (wrapped to [0, π])

    The azimuthal angle difference between two objects (e.g., lepton and MET)
    is a powerful discriminant: back-to-back objects (Δφ ≈ π) are typical
    of W decays, while collinear objects suggest QCD radiation.

    Returns:
        |Δφ| values in [0, π].
    """
    phi1 = np.asarray(phi1, dtype=np.float64)
    phi2 = np.asarray(phi2, dtype=np.float64)

    d_phi = np.abs(phi1 - phi2)
    # Fold to [0, π]
    d_phi = np.where(d_phi > np.pi, 2 * np.pi - d_phi, d_phi)
    return d_phi.astype(np.float32)


def pseudorapidity_from_theta(theta: np.ndarray) -> np.ndarray:
    """
    Convert polar angle θ to pseudorapidity η.

    Physics:
        η = -ln(tan(θ/2))

    Pseudorapidity is a standard coordinate in HEP detectors, replacing
    the polar angle θ. It equals true rapidity y in the massless limit.
    Most detector coverage is specified in η (e.g., |η| < 2.5 for CMS tracker).

    Args:
        theta: Polar angle in radians, in (0, π).

    Returns:
        Pseudorapidity η values.
    """
    theta = np.asarray(theta, dtype=np.float64)
    # Clamp to avoid log(0)
    theta = np.clip(theta, 1e-7, np.pi - 1e-7)
    return (-np.log(np.tan(theta / 2))).astype(np.float32)


def missing_et_significance(
    met: np.ndarray, ht: np.ndarray
) -> np.ndarray:
    """
    Compute MET significance: MET / √H_T.

    Physics:
        MET_sig = MET / √H_T

    Raw MET in hadronic events scales with √H_T due to calorimeter resolution.
    MET/√H_T normalizes this and is a better discriminant for true missing
    energy (from neutrinos or new physics) vs fake MET from detector noise.

    Returns:
        MET significance [dimensionless, in units of √GeV].
    """
    met = np.asarray(met, dtype=np.float64)
    ht = np.asarray(ht, dtype=np.float64)
    return (met / np.sqrt(np.maximum(ht, 1.0))).astype(np.float32)
