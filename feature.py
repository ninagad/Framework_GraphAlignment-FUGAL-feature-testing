from enum import Enum, auto
import numpy as np
import matplotlib.pyplot as plt


class FeatureExtensions:
    @staticmethod
    def to_str(feature: 'Feature') -> str:
        return feature.name.lower()

    @staticmethod
    def to_feature_combination(left_feature: 'Feature', right_feature: 'Feature') -> str:
        return f"{right_feature},{left_feature}"

    @staticmethod
    def to_feature(name: str):  #-> 'Feature' | None:
        name = name.replace('[', '').replace('\'', '').replace(']', '')  # Remove brackets and '.
        try:
            return Feature[name.upper()]
        except:
            try:  # 2hop_edges and 2hop_neighbors
                name = name.replace('2', 'TWO').upper()
                return Feature[name]
            except:
                print(f'Feature: {name} is not an ENUM -> excluded from plot')
                return

    @staticmethod
    def to_label(feature: 'Feature') -> str:
        label_dict = {
            # NetSimile
            Feature.DEG: 'degree',
            Feature.CLUSTER: 'cluster coeff',
            Feature.AVG_EGO_DEG: 'avg ego deg',
            Feature.AVG_EGO_CLUSTER: 'avg ego cluster coeff',
            Feature.EGO_EDGES: 'ego edges',
            Feature.EGO_OUT_EDGES: 'ego out edges',
            Feature.EGO_NEIGHBORS: 'ego neighbors',

            # Miscellaneous
            Feature.SUM_EGO_CLUSTER: "sum ego cluster coeff",
            Feature.VAR_EGO_CLUSTER: "var ego cluster coeff",
            Feature.ASSORTATIVITY_EGO: "ego assortativity",
            Feature.INTERNAL_FRAC_EGO: "ego internal frac",

            # STATISTICAL
            Feature.MODE_EGO_DEGS: "mode ego degs",
            Feature.MEDIAN_EGO_DEGS: "median ego degs",
            Feature.MIN_EGO_DEGS: "min ego degs",
            Feature.MAX_EGO_DEGS: "max ego degs",
            Feature.RANGE_EGO_DEGS: "range ego degs",
            Feature.SKEWNESS_EGO_DEGS: "skewness ego degs",
            Feature.KURTOSIS_EGO_DEGS: "kurtosis ego degs",

            # CENTRALITY
            Feature.CLOSENESS_CENTRALITY: "closeness centrality",
            Feature.DEGREE_CENTRALITY: "degree centrality",
            Feature.EIGENVECTOR_CENTRALITY: "eigenvector centrality",
            Feature.PAGERANK: "pagerank",

            # 2 hop features
            Feature.AVG_2HOP_DEG: 'avg 2hop deg',
            Feature.AVG_2HOP_CLUSTER: 'avg 2hop cluster coeff',
            Feature.TWOHOP_EDGES: '2hop edges',
            Feature.TWOHOP_NEIGHBORS: '2hop neighbors',
            Feature.SUM_2HOP_CLUSTER: 'sum 2hop cluster coeff',
            Feature.VAR_2HOP_CLUSTER: 'var 2hop cluster coeff',
            Feature.ASSORTATIVITY_2HOP: '2hop assortativity',
            Feature.INTERNAL_FRAC_2HOP: '2hop internal frac',
            Feature.MEDIAN_2HOP_DEGS: 'median 2hop degs',
            Feature.MAX_2HOP_DEGS: 'max 2hop degs',
            Feature.RANGE_2HOP_DEGS: 'range 2hop degs',
            Feature.SKEWNESS_2HOP_DEGS: 'skewness 2hop degs'
        }

        return label_dict[feature]

    @staticmethod
    def to_labels(features: list['Feature']) -> str:
        labels = [FeatureExtensions.to_label(feature) for feature in features]
        label = ", ".join(labels)

        return label


class Feature(Enum):
    # NETSIMILE
    DEG = 0  # Force enums to start from 0
    CLUSTER = auto()
    AVG_EGO_DEG = auto()
    AVG_EGO_CLUSTER = auto()
    EGO_EDGES = auto()
    EGO_OUT_EDGES = auto()
    EGO_NEIGHBORS = auto()

    # MISCELLANEOUS
    SUM_EGO_CLUSTER = auto()
    VAR_EGO_CLUSTER = auto()
    ASSORTATIVITY_EGO = auto()
    INTERNAL_FRAC_EGO = auto()

    # STATISTICAL
    MODE_EGO_DEGS = auto()
    MEDIAN_EGO_DEGS = auto()
    MIN_EGO_DEGS = auto()
    MAX_EGO_DEGS = auto()
    RANGE_EGO_DEGS = auto()
    SKEWNESS_EGO_DEGS = auto()
    KURTOSIS_EGO_DEGS = auto()

    # CENTRALITY
    CLOSENESS_CENTRALITY = auto()
    DEGREE_CENTRALITY = auto()
    EIGENVECTOR_CENTRALITY = auto()
    PAGERANK = auto()

    # 2-hop features
    AVG_2HOP_DEG = AVG_EGO_DEG
    AVG_2HOP_CLUSTER = AVG_EGO_CLUSTER
    TWOHOP_EDGES = EGO_EDGES
    TWOHOP_NEIGHBORS = EGO_NEIGHBORS
    SUM_2HOP_CLUSTER = SUM_EGO_CLUSTER
    VAR_2HOP_CLUSTER = VAR_EGO_CLUSTER
    ASSORTATIVITY_2HOP = ASSORTATIVITY_EGO
    INTERNAL_FRAC_2HOP = INTERNAL_FRAC_EGO
    MEDIAN_2HOP_DEGS = MEDIAN_EGO_DEGS
    MAX_2HOP_DEGS = MAX_EGO_DEGS
    RANGE_2HOP_DEGS = RANGE_EGO_DEGS
    SKEWNESS_2HOP_DEGS = SKEWNESS_EGO_DEGS
