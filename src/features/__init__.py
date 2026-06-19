"""Feature engineering package for the Particle Physics Classifier."""

from src.features.physics_features import (
    invariant_mass,
    delta_r,
    transverse_mass,
    rapidity,
    ht_scalar,
    centrality,
    azimuthal_angle_difference,
    missing_et_significance,
    four_vector_components,
)
from src.features.low_level_features import LowLevelExtractor
from src.features.high_level_features import HighLevelFeatureBuilder

# FeatureStore and JetSubstructureCalculator are available but imported lazily
# to avoid pulling in pandera / torch at top-level import time.
def FeatureStore(*args, **kwargs):  # noqa: N802
    from src.features.feature_store import FeatureStore as _FS
    return _FS(*args, **kwargs)


def FeatureConfig(*args, **kwargs):  # noqa: N802
    from src.features.feature_store import FeatureConfig as _FC
    return _FC(*args, **kwargs)


def JetSubstructureCalculator(*args, **kwargs):  # noqa: N802
    from src.features.jet_substructure import JetSubstructureCalculator as _JSC
    return _JSC(*args, **kwargs)


def JetConstituentData(*args, **kwargs):  # noqa: N802
    from src.features.jet_substructure import JetConstituentData as _JCD
    return _JCD(*args, **kwargs)


__all__ = [
    # Physics primitives
    "invariant_mass",
    "delta_r",
    "transverse_mass",
    "rapidity",
    "ht_scalar",
    "centrality",
    "azimuthal_angle_difference",
    "missing_et_significance",
    "four_vector_components",
    # Feature builders
    "LowLevelExtractor",
    "HighLevelFeatureBuilder",
    # Feature store (lazy)
    "FeatureStore",
    "FeatureConfig",
    # Jet substructure (lazy)
    "JetSubstructureCalculator",
    "JetConstituentData",
]
