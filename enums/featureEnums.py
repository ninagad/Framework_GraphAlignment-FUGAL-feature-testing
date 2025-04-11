from aenum import Enum, NoAlias


class FeatureExtensions:
    @staticmethod
    def to_feature(name: str):  #-> 'Feature' | None:
        name = name.replace('[', '').replace('\'', '').replace(']', '')  # Remove brackets and '.
        try:
            return FeatureEnums[name.upper()]
        except:
            if name == 'katz': # Katz centrality
                return FeatureEnums['KATZ_CENTRALITY']

            try:  # 2hop_edges and 2hop_neighbors
                name = name.replace('2', 'TWO').upper()
                return FeatureEnums[name]
            except:
                print(f'Feature: {name} is not an ENUM -> excluded from plot')
                return

    @staticmethod
    def to_label(feature: 'FeatureEnums') -> str:
        label_dict = {
            # NetSimile
            FeatureEnums.DEG: 'degree',
            FeatureEnums.CLUSTER: 'cluster coeff',
            FeatureEnums.AVG_EGO_DEG: 'avg ego deg',
            FeatureEnums.AVG_EGO_CLUSTER: 'avg ego cluster coeff',
            FeatureEnums.EGO_EDGES: 'ego edges',
            FeatureEnums.EGO_OUT_EDGES: 'ego out edges',
            FeatureEnums.EGO_NEIGHBORS: 'ego neighbors',

            # Miscellaneous
            FeatureEnums.SUM_EGO_CLUSTER: "sum ego cluster coeff",
            FeatureEnums.VAR_EGO_CLUSTER: "var ego cluster coeff",
            FeatureEnums.ASSORTATIVITY_EGO: "ego assortativity",
            FeatureEnums.INTERNAL_FRAC_EGO: "ego internal frac",

            # STATISTICAL
            FeatureEnums.SUM_EGO_DEG: "sum ego degs",
            FeatureEnums.STD_EGO_DEG: "std ego degs",
            FeatureEnums.MODE_EGO_DEGS: "mode ego degs",
            FeatureEnums.MEDIAN_EGO_DEGS: "median ego degs",
            FeatureEnums.MIN_EGO_DEGS: "min ego degs",
            FeatureEnums.MAX_EGO_DEGS: "max ego degs",
            FeatureEnums.RANGE_EGO_DEGS: "range ego degs",
            FeatureEnums.SKEWNESS_EGO_DEGS: "skewness ego degs",
            FeatureEnums.KURTOSIS_EGO_DEGS: "kurtosis ego degs",

            FeatureEnums.STD_EGO_CLUSTER: "std ego cluster coeff",
            FeatureEnums.MEDIAN_EGO_CLUSTER: "median ego cluster coeff",
            FeatureEnums.MIN_EGO_CLUSTER: "min ego cluster coeff",
            FeatureEnums.MAX_EGO_CLUSTER: "max ego cluster coeff",
            FeatureEnums.RANGE_EGO_CLUSTER: "range ego cluster coeff",
            FeatureEnums.SKEWNESS_EGO_CLUSTER: "skewness ego cluster coeff",
            FeatureEnums.KURTOSIS_EGO_CLUSTER: "kurtosis ego cluster coeff",

            # CENTRALITY
            FeatureEnums.CLOSENESS_CENTRALITY: "closeness centrality",
            FeatureEnums.DEGREE_CENTRALITY: "degree centrality",
            FeatureEnums.EIGENVECTOR_CENTRALITY: "eigenvector centrality",
            FeatureEnums.PAGERANK: "pagerank",
            FeatureEnums.KATZ_CENTRALITY: "katz centrality",

            # 2 hop features
            FeatureEnums.AVG_2HOP_DEG: 'avg 2hop deg',
            FeatureEnums.AVG_2HOP_CLUSTER: 'avg 2hop cluster coeff',
            FeatureEnums.TWOHOP_EDGES: '2hop edges',
            FeatureEnums.TWOHOP_NEIGHBORS: '2hop neighbors',
            FeatureEnums.SUM_2HOP_CLUSTER: 'sum 2hop cluster coeff',
            FeatureEnums.VAR_2HOP_CLUSTER: 'var 2hop cluster coeff',
            FeatureEnums.ASSORTATIVITY_2HOP: '2hop assortativity',
            FeatureEnums.INTERNAL_FRAC_2HOP: '2hop internal frac',
            FeatureEnums.MEDIAN_2HOP_DEGS: 'median 2hop degs',
            FeatureEnums.MAX_2HOP_DEGS: 'max 2hop degs',
            FeatureEnums.RANGE_2HOP_DEGS: 'range 2hop degs',
            FeatureEnums.SKEWNESS_2HOP_DEGS: 'skewness 2hop degs'
        }

        return label_dict[feature]

    @staticmethod
    def to_labels(features: list['FeatureEnums']) -> str:
        labels = [FeatureExtensions.to_label(feature) for feature in features]
        label = ", ".join(labels)

        fugal_features = [FeatureEnums['DEG'], FeatureEnums['CLUSTER'], FeatureEnums['AVG_EGO_DEG'], FeatureEnums['AVG_EGO_CLUSTER']]
        fugal_label = ", ".join([FeatureExtensions.to_label(feature) for feature in fugal_features])

        if label == fugal_label:
            return 'FUGAL'
        return label

    @staticmethod
    def transform_feature_str_to_label(feature: str):
        fe = FeatureExtensions()

        if ',' not in feature:
            feature_enum = fe.to_feature(feature)
            if feature_enum is None:
                return None

            return fe.to_label(feature_enum)
        else:
            features_in_combination = feature.replace(' ', '').split(',')
            features = [fe.to_feature(name) for name in features_in_combination]
            return fe.to_labels(features)
class FeatureEnums(Enum):
    _settings_ = NoAlias

    def __repr__(self):
        return self.name

    # NETSIMILE
    DEG = 0  # Force enums to start from 0
    CLUSTER = 1
    AVG_EGO_DEG = 2
    AVG_EGO_CLUSTER = 3
    EGO_EDGES = 4
    EGO_OUT_EDGES = 5
    EGO_NEIGHBORS = 6

    # MISCELLANEOUS
    ASSORTATIVITY_EGO = 9
    INTERNAL_FRAC_EGO = 10
    VAR_EGO_CLUSTER = 31

    # STATISTICAL on ego degrees
    SUM_EGO_DEG = 23
    STD_EGO_DEG = 24
    MODE_EGO_DEGS = 11
    MEDIAN_EGO_DEGS = 12
    MIN_EGO_DEGS = 13
    MAX_EGO_DEGS = 14
    RANGE_EGO_DEGS = 15
    SKEWNESS_EGO_DEGS = 16
    KURTOSIS_EGO_DEGS = 17

    # Statistical on ego clustering coefficients
    SUM_EGO_CLUSTER = 7
    STD_EGO_CLUSTER = 8
    MEDIAN_EGO_CLUSTER = 25
    MIN_EGO_CLUSTER = 26
    MAX_EGO_CLUSTER = 27
    RANGE_EGO_CLUSTER = 28
    SKEWNESS_EGO_CLUSTER = 29
    KURTOSIS_EGO_CLUSTER = 30

    # CENTRALITY
    CLOSENESS_CENTRALITY = 18
    DEGREE_CENTRALITY = 19
    EIGENVECTOR_CENTRALITY = 20
    PAGERANK = 21
    KATZ_CENTRALITY = 22

    # 2-hop features
    AVG_2HOP_DEG = 2
    AVG_2HOP_CLUSTER = 3
    TWOHOP_EDGES = 4
    TWOHOP_NEIGHBORS = 6
    SUM_2HOP_CLUSTER = 7
    VAR_2HOP_CLUSTER = 8
    ASSORTATIVITY_2HOP = 9
    INTERNAL_FRAC_2HOP = 10
    MEDIAN_2HOP_DEGS = 12
    MAX_2HOP_DEGS = 14
    RANGE_2HOP_DEGS = 15
    SKEWNESS_2HOP_DEGS = 16
