from feature import Feature
import matplotlib.pyplot as plt
import numpy as np
from feature import FeatureExtensions as FE



class PlotUtils:
    def __init__(self):
        self.colorscale_dict = {}
        self.markers_dict = {}
        self.marker_options = ['o', '^', 's', 'x', 'D', 'P', 'd', '>', 'p', 'v', '1', '<']

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

        for feature in Feature:
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
            [Feature.DEG, Feature.EGO_EDGES],
            [Feature.DEG, Feature.MAX_EGO_DEGS],
            [Feature.EGO_EDGES, Feature.MAX_EGO_DEGS],
            [Feature.DEG, Feature.EGO_EDGES, Feature.MAX_EGO_DEGS],
            # Netscience
            [Feature.DEG, Feature.SUM_EGO_CLUSTER],
            [Feature.EGO_EDGES, Feature.SUM_EGO_CLUSTER],
            [Feature.DEG, Feature.EGO_EDGES, Feature.SUM_EGO_CLUSTER],
            # Voles
            [Feature.DEG, Feature.CLUSTER],
            [Feature.CLUSTER, Feature.EGO_EDGES],
            [Feature.DEG, Feature.CLUSTER, Feature.EGO_EDGES],
            # socfb-Bowdoin47
            [Feature.CLUSTER, Feature.SUM_EGO_CLUSTER],
            [Feature.DEG, Feature.CLUSTER, Feature.SUM_EGO_CLUSTER],

        ]
        combination_colormap = 'pink_r'
        feature_combinations.append(combination_features)
        colormaps.append(combination_colormap)

        centrality_combinations = [
            [Feature.DEG, Feature.DEGREE_CENTRALITY],
            [Feature.DEG, Feature.PAGERANK],
            [Feature.DEG, Feature.KATZ_CENTRALITY]
        ]
        centrality_combinations_colormap = 'copper_r'

        feature_combinations.append(centrality_combinations)
        colormaps.append(centrality_combinations_colormap)

        other_algo_combinations = [
            [Feature.EGO_EDGES, Feature.PAGERANK],
            [Feature.DEG, Feature.EGO_EDGES, Feature.PAGERANK]
        ]
        other_algo_combinations_colormap = 'YlOrBr'
        feature_combinations.append(other_algo_combinations)
        colormaps.append(other_algo_combinations_colormap)

        for colormap, combinations in zip(colormaps, feature_combinations):
            if colormap == other_algo_combinations_colormap:
                upper = 0.5
            else:
                upper = 0.9

            cmap = plt.get_cmap(colormap)  # Get the colormap
            colors = cmap(np.linspace(0.3, upper, len(combinations)))  # Generate shades

            for idx, feature_comb in enumerate(combinations):
                name = FE.to_labels(feature_comb)
                self.colorscale_dict[name] = colors[idx]
                self.markers_dict[name] = self.marker_options[idx]

        FUGAL_combination = (
            [Feature.DEG, Feature.CLUSTER, Feature.AVG_EGO_DEG, Feature.AVG_EGO_CLUSTER]
        )
        fugal_name = FE.to_labels(FUGAL_combination)
        self.colorscale_dict[fugal_name] = plt.get_cmap("Set1")(4)

    def to_color(self, feature: Feature) -> str:
        name = FE.to_label(feature)
        return self.colorscale_dict[name]

    def to_colors(self, features: list[Feature]) -> str:
        name = FE.to_labels(features)
        return self.colorscale_dict[name]


    def to_marker(self, feature: Feature) -> str:
        name = FE.to_label(feature)
        return self.markers_dict[name]

    def to_markers(self, features: list[Feature]) -> str | None:
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




