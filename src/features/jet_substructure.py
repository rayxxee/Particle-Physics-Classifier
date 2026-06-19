"""
Jet substructure features: n-subjettiness, jet mass, energy correlation functions.

These are advanced jet observables that require access to jet constituent
particles (the individual tracks and calorimeter towers that form the jet).
They are extremely powerful for discriminating:
- W/Z/H jets from QCD jets (using τ₂₁ = τ₂/τ₁)
- Top quark jets from QCD (using τ₃₂)
- Quark-initiated vs gluon-initiated jets

Status in Phase 1:
    The HIGGS CSV dataset does not include jet constituent information.
    This module provides the mathematical framework and placeholder
    implementations that are activated when:
    a) ROOT data with jet constituents is loaded (Phase 2+), or
    b) A particle-level dataset with constituent access is used.

    All functions return NaN arrays when constituent data is unavailable,
    which downstream models handle gracefully.

Physics references:
    - Thaler & Van Tilburg (2011) — N-subjettiness
    - Larkoski, Salam, Thaler (2013) — Energy Correlation Functions
    - Moult et al. (2016) — D2 observable

Usage:
    from src.features.jet_substructure import JetSubstructureCalculator

    calc = JetSubstructureCalculator()
    tau21 = calc.n_subjettiness_ratio(jet_constituents, N1=1, N2=2)
"""

from __future__ import annotations

import warnings

import numpy as np


class JetConstituentData:
    """
    Container for jet constituent particles.

    Attributes:
        pt:    Constituent pT values, shape (n_constituents,)
        eta:   Constituent η values, shape (n_constituents,)
        phi:   Constituent φ values, shape (n_constituents,)
        E:     Constituent energy values, shape (n_constituents,)
    """

    def __init__(
        self,
        pt: np.ndarray,
        eta: np.ndarray,
        phi: np.ndarray,
        E: np.ndarray | None = None,
    ) -> None:
        self.pt = np.asarray(pt, dtype=np.float64)
        self.eta = np.asarray(eta, dtype=np.float64)
        self.phi = np.asarray(phi, dtype=np.float64)
        self.E = np.asarray(E, dtype=np.float64) if E is not None else self.pt.copy()

    @property
    def n_constituents(self) -> int:
        return len(self.pt)


class JetSubstructureCalculator:
    """
    Computes jet substructure observables from jet constituent particles.

    When constituent data is not available, returns NaN arrays with a
    warning rather than raising an exception — this allows the pipeline
    to run end-to-end without constituent data.

    Example (with constituent data):
        constituents = JetConstituentData(pt=..., eta=..., phi=...)
        calc = JetSubstructureCalculator()
        tau1 = calc.n_subjettiness(constituents, N=1, R=0.4)
        tau2 = calc.n_subjettiness(constituents, N=2, R=0.4)
        tau21 = tau2 / np.maximum(tau1, 1e-10)  # τ₂₁ discriminant
    """

    def n_subjettiness(
        self,
        constituents: JetConstituentData | None,
        N: int,
        R: float = 0.4,
        beta: float = 1.0,
    ) -> float:
        """
        Compute τ_N (N-subjettiness) for a jet.

        Physics:
            τ_N = (1 / d₀) · Σᵢ pT_i · min(ΔR_{i,1}, ..., ΔR_{i,N})^β

        where the min is over N candidate subjet axes found by exclusive
        kT clustering, and d₀ = Σᵢ pT_i · R^β is a normalization.

        τ₁ ≈ 0: jet has 1 prong (quark/gluon jet)
        τ₂ ≈ 0: jet has ≤ 2 prongs (W/Z/H → qq̄)
        τ₂/τ₁ ≈ 0: good W/Z/H tagger
        τ₃/τ₂ ≈ 0: good top tagger

        Args:
            constituents: Jet constituent particles.
            N:            Number of subjets (N=1 for 1-prong, N=2 for 2-prong).
            R:            Jet radius parameter.
            beta:         Angular exponent (β=1 default, β=2 is also common).

        Returns:
            τ_N value (float).
        """
        if constituents is None or constituents.n_constituents == 0:
            return np.nan

        # Find N subjet axes using exclusive kT clustering (simplified: use N hardest)
        # Production version: use pyjet or fastjet bindings
        axes_idx = np.argsort(constituents.pt)[-N:][::-1]
        axes_eta = constituents.eta[axes_idx]
        axes_phi = constituents.phi[axes_idx]

        # Compute d₀
        d0 = np.sum(constituents.pt) * (R**beta)
        if d0 < 1e-10:
            return np.nan

        # Compute τ_N
        tau = 0.0
        for i in range(constituents.n_constituents):
            # Minimum ΔR to any of the N axes
            dr_to_axes = np.array([
                self._delta_r(constituents.eta[i], constituents.phi[i],
                              axes_eta[k], axes_phi[k])
                for k in range(len(axes_eta))
            ])
            min_dr = np.min(dr_to_axes)
            tau += constituents.pt[i] * (min_dr**beta)

        return tau / d0

    def n_subjettiness_ratio(
        self,
        constituents: JetConstituentData | None,
        N1: int,
        N2: int,
        R: float = 0.4,
    ) -> float:
        """
        Compute τ_{N2} / τ_{N1} — the ratio discriminant.

        τ₂₁ = τ₂/τ₁: W/Z/H tagger (2-prong vs 1-prong)
        τ₃₂ = τ₃/τ₂: Top quark tagger (3-prong vs 2-prong)

        Returns:
            Ratio value, or NaN if constituent data unavailable.
        """
        tau_n1 = self.n_subjettiness(constituents, N=N1, R=R)
        tau_n2 = self.n_subjettiness(constituents, N=N2, R=R)
        if np.isnan(tau_n1) or np.isnan(tau_n2) or tau_n1 < 1e-10:
            return np.nan
        return tau_n2 / tau_n1

    def jet_mass(self, constituents: JetConstituentData | None) -> float:
        """
        Compute the invariant mass of a jet from its constituents.

        Physics:
            M_jet² = (Σᵢ Eᵢ)² - |Σᵢ p⃗ᵢ|²

        Jet mass is a powerful discriminant for boosted heavy objects:
        - QCD jets: M_jet ~ 0-30 GeV
        - W/Z jets: M_jet ~ M_W/Z ≈ 80-91 GeV
        - H → bb jets: M_jet ~ M_H ≈ 125 GeV

        Returns:
            Jet mass in GeV, or NaN if constituent data unavailable.
        """
        if constituents is None or constituents.n_constituents == 0:
            return np.nan

        E_tot = np.sum(constituents.E)
        px_tot = np.sum(constituents.pt * np.cos(constituents.phi))
        py_tot = np.sum(constituents.pt * np.sin(constituents.phi))
        pz_tot = np.sum(constituents.pt * np.sinh(constituents.eta))

        m_sq = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
        return float(np.sqrt(max(m_sq, 0.0)))

    def energy_correlation_function(
        self,
        constituents: JetConstituentData | None,
        N: int = 2,
        beta: float = 1.0,
    ) -> float:
        """
        Compute the N-point energy correlation function e_N.

        Physics (N=2):
            e₂ = (1/pT_jet²) · Σᵢ<ⱼ pTᵢ · pTⱼ · ΔRᵢⱼ^β

        Higher-order ECFs provide more discriminating power but are more
        computationally expensive (O(n^N) scaling with constituents).

        C₂ = e₃ / e₂² and D₂ = e₃ / e₂³ are derived discriminants
        used at CMS/ATLAS for W/Z/H vs QCD jet tagging.

        Returns:
            e_N value, or NaN if constituent data unavailable.
        """
        if constituents is None or constituents.n_constituents < N:
            return np.nan

        pt_jet = np.sum(constituents.pt)
        if pt_jet < 1e-10:
            return np.nan

        if N == 2:
            ecf = 0.0
            for i in range(constituents.n_constituents):
                for j in range(i + 1, constituents.n_constituents):
                    dr = self._delta_r(
                        constituents.eta[i], constituents.phi[i],
                        constituents.eta[j], constituents.phi[j],
                    )
                    ecf += constituents.pt[i] * constituents.pt[j] * (dr**beta)
            return ecf / (pt_jet**2)

        elif N == 3:
            ecf = 0.0
            for i in range(constituents.n_constituents):
                for j in range(i + 1, constituents.n_constituents):
                    for k in range(j + 1, constituents.n_constituents):
                        dr_ij = self._delta_r(
                            constituents.eta[i], constituents.phi[i],
                            constituents.eta[j], constituents.phi[j],
                        )
                        dr_ik = self._delta_r(
                            constituents.eta[i], constituents.phi[i],
                            constituents.eta[k], constituents.phi[k],
                        )
                        dr_jk = self._delta_r(
                            constituents.eta[j], constituents.phi[j],
                            constituents.eta[k], constituents.phi[k],
                        )
                        ecf += (
                            constituents.pt[i]
                            * constituents.pt[j]
                            * constituents.pt[k]
                            * (dr_ij * dr_ik * dr_jk) ** beta
                        )
            return ecf / (pt_jet**3)

        else:
            warnings.warn(f"ECF N={N} not implemented. Only N=2,3 supported.")
            return np.nan

    @staticmethod
    def _delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
        """Scalar ΔR for two particles."""
        d_eta = eta1 - eta2
        d_phi = phi1 - phi2
        if d_phi > np.pi:
            d_phi -= 2 * np.pi
        elif d_phi < -np.pi:
            d_phi += 2 * np.pi
        return float(np.sqrt(d_eta**2 + d_phi**2))

    def batch_unavailable(self, n_events: int) -> dict[str, np.ndarray]:
        """
        Return NaN arrays for all substructure features.
        Used when constituent data is not available (Phase 1).
        """
        nan_arr = np.full(n_events, np.nan, dtype=np.float32)
        return {
            "tau1": nan_arr.copy(),
            "tau2": nan_arr.copy(),
            "tau21": nan_arr.copy(),
            "tau3": nan_arr.copy(),
            "tau32": nan_arr.copy(),
            "jet_mass_j1": nan_arr.copy(),
            "ecf2": nan_arr.copy(),
            "ecf3": nan_arr.copy(),
        }
