from aenum import Enum, NoAlias


class FeatureExtensions:
    @staticmethod
    def to_feature(name: str):  #-> 'Feature' | None:
        name = name.replace('[', '').replace('\'', '').replace(']', '')  # Remove brackets and '.
        try:
            return Feature[name.upper()]
        except:
            if name == 'katz': # Katz centrality
                return Feature['KATZ_CENTRALITY']

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
            Feature.KATZ_CENTRALITY: "katz centrality",

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

        fugal_features = [Feature['DEG'], Feature['CLUSTER'], Feature['AVG_EGO_DEG'], Feature['AVG_EGO_CLUSTER']]
        fugal_label = ", ".join([FeatureExtensions.to_label(feature) for feature in fugal_features])

        if label == fugal_label:
            return 'FUGAL'
        return label


class Feature(Enum):
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
    SUM_EGO_CLUSTER = 7
    VAR_EGO_CLUSTER = 8
    ASSORTATIVITY_EGO = 9
    INTERNAL_FRAC_EGO = 10

    # STATISTICAL
    MODE_EGO_DEGS = 11
    MEDIAN_EGO_DEGS = 12
    MIN_EGO_DEGS = 13
    MAX_EGO_DEGS = 14
    RANGE_EGO_DEGS = 15
    SKEWNESS_EGO_DEGS = 16
    KURTOSIS_EGO_DEGS = 17

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
