from enum import Enum, auto


class NormalizationEnums(Enum):
    NO_NORMALIZATION = auto()

    # Normalization
    NORMALIZE_FEATURES = auto()
    NORMALIZE_DIFFERENCES = auto()

    # Standardization
    STANDARDIZE_FEATURES = auto()
