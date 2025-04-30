from enum import Enum, auto


class ScalingEnums(Enum):
    def __repr__(self):
        return self.name

    NO_SCALING = "no_scaling"

    # Normalization
    INDIVIDUAL_MM_NORMALIZATION = "individual_mm_normalization"
    COLLECTIVE_MM_NORMALIZATION = "collective_mm_normalization"

    # Standardization
    INDIVIDUAL_STANDARDIZATION = "individual_standardization"
    COLLECTIVE_STANDARDIZATION = "collective_standardization"

    # Robust normalization
    INDIVIDUAL_ROBUST_NORMALIZATION = "individual_robust_normalization"
    COLLECTIVE_ROBUST_NORMALIZATION = "collective_robust_normalization"
