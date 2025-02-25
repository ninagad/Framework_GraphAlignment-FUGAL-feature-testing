from feature import Feature
import matplotlib.pyplot as plt
import numpy as np
from feature import FeatureExtensions as FE



class PlotUtils:
    def __init__(self):
        self.colorscale_dict = {}
        self.markers_dict = {}
        self.marker_options = ['o', '^', 's', 'x', 'D', 'P', 'd', '>', 'p', '1', 'v']

        self.init_single_feature()
        self.init_combination_features()

    def init_single_feature(self):
        colormaps = ['Blues', 'Greys', 'Greens', 'Purples']  # OrRd']
        group_sizes = [7, 4, 7, 4]

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
        # Feature combinations
        combination_features = [
            # Euroroad combinations
            [Feature.DEG, Feature.EGO_EDGES],
            [Feature.DEG, Feature.MAX_EGO_DEGS],
            [Feature.EGO_EDGES, Feature.MAX_EGO_DEGS],
            #[Feature.DEG, Feature.EGO_EDGES, Feature.MAX_EGO_DEGS],
            # Netscience
            [Feature.DEG, Feature.SUM_EGO_CLUSTER],
            [Feature.EGO_EDGES, Feature.SUM_EGO_CLUSTER],
            #[Feature.DEG, Feature.SUM_EGO_CLUSTER, Feature.EGO_EDGES],
            # socfb-Bowdoin47
            [Feature.DEG, Feature.CLUSTER],
            [Feature.CLUSTER, Feature.SUM_EGO_CLUSTER],
            #[Feature.DEG, Feature.CLUSTER, Feature.SUM_EGO_CLUSTER],
            # Voles
            [Feature.CLUSTER, Feature.EGO_EDGES],
        ]

        combination_colormap = 'pink'
        cmap = plt.get_cmap(combination_colormap)  # Get the colormap
        colors = cmap(np.linspace(0.1, 0.7, len(combination_features)))  # Generate shades

        for idx, feature_comb in enumerate(combination_features):
            name = FE.to_labels(feature_comb)
            self.colorscale_dict[name] = colors[idx]
            self.markers_dict[name] = self.marker_options[idx]

    def to_color(self, feature: Feature) -> str:
        name = FE.to_label(feature)
        return self.colorscale_dict[name]

    def to_colors(self, features: list[Feature]) -> str:
        name = FE.to_labels(features)
        return self.colorscale_dict[name]


    def to_marker(self, feature: Feature) -> str:
        name = FE.to_label(feature)
        return self.markers_dict[name]

    def to_markers(self, features: list[Feature]) -> str:
        name = FE.to_labels(features)
        return self.markers_dict[name]
