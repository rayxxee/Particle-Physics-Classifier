from typing import Dict, List

# Map feature names to human-readable physics descriptions
PHYSICS_DESCRIPTIONS: Dict[str, str] = {
    "m_bb": "Invariant mass of the two b-tagged jets. A peak near 125 GeV strongly suggests a Higgs boson decay (H → bb).",
    "m_ww": "Invariant mass of the WW candidate. Reconstructs the Higgs mass for H → WW decays.",
    "deltaR_lep_jet": "Angular separation between the lepton and nearest jet. Small values indicate boosted decays or misidentified leptons.",
    "met": "Missing transverse energy. Represents neutrinos or undiscovered invisible particles escaping the detector.",
    "lepton_pt": "Transverse momentum of the primary lepton. High pT suggests a decay from a heavy parent particle (W/Z/Higgs).",
    "jet1_pt": "Transverse momentum of the leading jet. High activity events tend to have harder jets.",
    "ht_scalar": "Scalar sum of all transverse momenta. A global measure of the event's total energy scale.",
    "centrality": "Ratio of transverse energy to total energy. Signal events often deposit energy more centrally in the detector.",
    "n_jets": "Total number of reconstructed jets. Multi-jet events are typical for top quark pairs or complex signal topologies.",
    "n_bjets": "Number of b-tagged jets. Essential for discriminating top/Higgs events from light-flavor QCD backgrounds.",
}

def interpret_shap_physics(shap_values: dict, feature_names: List[str], top_k: int = 3) -> List[str]:
    """
    Map top SHAP features back to human-readable physics explanations.
    
    Args:
        shap_values: Dictionary mapping feature names to their SHAP importance value.
        feature_names: List of all feature names.
        top_k: Number of top features to explain.
        
    Returns:
        List of physics explanations for the most important features.
    """
    # Sort features by absolute SHAP value
    sorted_features = sorted(
        feature_names,
        key=lambda f: abs(shap_values.get(f, 0.0)),
        reverse=True
    )
    
    explanations = []
    for f in sorted_features[:top_k]:
        val = shap_values.get(f, 0.0)
        direction = "signal-like" if val > 0 else "background-like"
        desc = PHYSICS_DESCRIPTIONS.get(f, f"Feature {f} contributes to the prediction.")
        
        explanation = f"**{f}** pushed the prediction towards {direction}. {desc}"
        explanations.append(explanation)
        
    return explanations
