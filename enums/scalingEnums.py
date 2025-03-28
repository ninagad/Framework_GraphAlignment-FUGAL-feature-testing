from enum import Enum, auto


class ScalingEnums(Enum):
    NO_SCALING = auto()

    # Normalization
    INDIVIDUAL_MM_NORMALIZATION = auto()
    COLLECTIVE_MM_NORMALIZATION = auto()
    NORMALIZE_DIFFERENCES = auto()

    # Standardization
    INDIVIDUAL_STANDARDIZATION = auto()
    COLLECTIVE_STANDARDIZATION = auto()

    # Robust normalization
    INDIVIDUAL_ROBUST_NORMALIZATION = auto()
    COLLECTIVE_ROBUST_NORMALIZATION = auto()
