from enums.featureEnums import FeatureEnums
import matplotlib.pyplot as plt
import numpy as np
from enums.featureEnums import FeatureExtensions as FE



class PlotUtils:
    def __init__(self):
        self.colorscale_dict = {}
        self.markers_dict = {}
        self.marker_options = ['o', '^', 's', 'x', 'D', 'P', 'd', '>', 'p', 'v', '1', '<', 'X']

        self.init_single_feature()
        self.init_combination_features()

    def init_single_feature(self):
        colormaps = ['Blues', 'Greys', 'Greens', 'Purples']  # OrRd']
        group_sizes = [7, 4, 7, 5]

        # Generate colorscale
        colorscale = np.empty((0, 4), float)
        markers = []
        for group_size, colormap in zip(group_sizes, colormaps):
            cmap = plt.get_cmap(colormap)  # Get the colormap
            colors = cmap(np.linspace(0.3, 0.9, group_size))  # Generate shades
            colorscale = np.vstack((colorscale, colors))

            markers = markers + self.marker_options[:group_size]

        for feature in FeatureEnums:
            name = FE.to_label(feature)
            idx = feature.value
            self.colorscale_dict[name] = colorscale[idx]

            self.markers_dict[name] = markers[idx]

    def init_combination_features(self):
        feature_combinations = []
        colormaps = []

        # Feature combinations
        combination_features = [
            # Euroroad combinations
            [FeatureEnums.DEG, FeatureEnums.EGO_EDGES],
            [FeatureEnums.DEG, FeatureEnums.MAX_EGO_DEGS],
            [FeatureEnums.EGO_EDGES, FeatureEnums.MAX_EGO_DEGS],
            [FeatureEnums.DEG, FeatureEnums.EGO_EDGES, FeatureEnums.MAX_EGO_DEGS],
            # Netscience
            [FeatureEnums.DEG, FeatureEnums.SUM_EGO_CLUSTER],
            [FeatureEnums.EGO_EDGES, FeatureEnums.SUM_EGO_CLUSTER],
            [FeatureEnums.DEG, FeatureEnums.EGO_EDGES, FeatureEnums.SUM_EGO_CLUSTER],
            # Voles
            [FeatureEnums.DEG, FeatureEnums.CLUSTER],
            [FeatureEnums.CLUSTER, FeatureEnums.EGO_EDGES],
            [FeatureEnums.DEG, FeatureEnums.CLUSTER, FeatureEnums.EGO_EDGES],
            # socfb-Bowdoin47
            [FeatureEnums.DEG, FeatureEnums.INTERNAL_FRAC_EGO],
            [FeatureEnums.SUM_EGO_CLUSTER, FeatureEnums.INTERNAL_FRAC_EGO],
            [FeatureEnums.DEG, FeatureEnums.SUM_EGO_CLUSTER, FeatureEnums.INTERNAL_FRAC_EGO],

        ]
        combination_colormap = 'pink_r'
        feature_combinations.append(combination_features)
        colormaps.append(combination_colormap)

        centrality_combinations = [
            [FeatureEnums.DEG, FeatureEnums.DEGREE_CENTRALITY],
            [FeatureEnums.DEG, FeatureEnums.PAGERANK],
            [FeatureEnums.DEG, FeatureEnums.KATZ_CENTRALITY]
        ]
        centrality_combinations_colormap = 'copper_r'

        feature_combinations.append(centrality_combinations)
        colormaps.append(centrality_combinations_colormap)

        other_algo_combinations = [
            [FeatureEnums.EGO_EDGES, FeatureEnums.PAGERANK],
            [FeatureEnums.DEG, FeatureEnums.EGO_EDGES, FeatureEnums.PAGERANK]
        ]
        other_algo_combinations_colormap = 'YlOrBr'
        feature_combinations.append(other_algo_combinations)
        colormaps.append(other_algo_combinations_colormap)

        for colormap, combinations in zip(colormaps, feature_combinations):
            if colormap == other_algo_combinations_colormap:
                upper = 0.4
            else:
                upper = 0.9

            cmap = plt.get_cmap(colormap)  # Get the colormap
            colors = cmap(np.linspace(0.3, upper, len(combinations)))  # Generate shades

            for idx, feature_comb in enumerate(combinations):
                name = FE.to_labels(feature_comb)
                self.colorscale_dict[name] = colors[idx]
                self.markers_dict[name] = self.marker_options[idx]

        FUGAL_combination = (
            [FeatureEnums.DEG, FeatureEnums.CLUSTER, FeatureEnums.AVG_EGO_DEG, FeatureEnums.AVG_EGO_CLUSTER]
        )
        fugal_name = FE.to_labels(FUGAL_combination)
        self.colorscale_dict[fugal_name] = plt.get_cmap("Set1")(4)

    def to_color(self, feature: FeatureEnums) -> str:
        name = FE.to_label(feature)
        return self.colorscale_dict[name]

    def to_colors(self, features: list[FeatureEnums]) -> str:
        name = FE.to_labels(features)
        return self.colorscale_dict[name]


    def to_marker(self, feature: FeatureEnums) -> str:
        name = FE.to_label(feature)
        return self.markers_dict[name]

    def to_markers(self, features: list[FeatureEnums]) -> str | None:
        name = FE.to_labels(features)
        try:
            return self.markers_dict[name]
        except:  # No marker defined, none returned
            return None

    def to_column_name(self, plottype: str) -> str:
        column_name_dict = {
            'p': 'variable',
            'External p': 'variable',
            'k': 'variable',
            'Noise-level': 'Noise-level',
            'n': 'variable'
        }

        return column_name_dict[plottype]




