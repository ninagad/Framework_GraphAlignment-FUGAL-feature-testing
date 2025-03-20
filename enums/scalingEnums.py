from enum import Enum, auto


class ScalingEnums(Enum):
    NO_SCALING = auto()

    # Normalization
    NORMALIZE_FEATURES = auto()
    NORMALIZE_DIFFERENCES = auto()

    # Standardization
    STANDARDIZE_FEATURES = auto()

    # Robust normalization
    ROBUST_NORMALIZE_FEATURES = auto()
