from feature import Feature
import matplotlib.pyplot as plt
import numpy as np

class PlotUtils:
    def __init__(self):
        colormaps = ['Blues', 'Greys', 'Greens', 'Purples']  # OrRd']
        group_sizes = [7, 4, 7, 4]
        marker_options = ['o', '^', 's', 'x', 'D', 'P', 'd']

        # Generate colorscale
        colorscale = np.empty((0, 4), float)
        markers = []
        for group_size, colormap in zip(group_sizes, colormaps):
            cmap = plt.get_cmap(colormap)  # Get the colormap
            colors = cmap(np.linspace(0.3, 0.9, group_size))  # Generate shades
            colorscale = np.vstack((colorscale, colors))

            markers = markers + marker_options[:group_size]

        self.colorscale = colorscale
        self.markers = markers

    def to_color(self, feature: Feature) -> str:
        return self.colorscale[feature.value]

    #def to_color(left_feature: Feature, right_feature: Feature) -> str:
       #return str(left_feature).lower() + "," + str(right_feature).lower()

    def to_marker(self, feature: Feature) -> str:
        return self.markers[feature.value]
