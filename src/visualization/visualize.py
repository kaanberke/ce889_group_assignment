import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from matplotlib import pyplot as plt

# png_renderer = pio.renderers["png"]
pio.renderers.default = "browser"


class DataVisualizer(object):
    def __init__(self, data: np.array, save_path: str = None) -> None:
        self.data = data
        self.save_path = save_path

    def visualize(self) -> None:
        ...

    def save(self, path: str) -> None:
        ...

    def heatmap(self) -> None:
        corr_matrix = self.data.corr()
        fig = px.imshow(corr_matrix)
        fig.show()

        if self.save_path is not None:
            fig.write_image(self.save_path)

        self.data.plot(kind="density", subplots=True, layout=(5, 5), sharex=False, legend=False, fontsize=1)

    def hist(self, x: str, y: str, color: str = None, pattern_shape: str = None, title: str = "") -> None:
        fig = px.histogram(
            self.data,
            x=x,
            y=y,
            color=color,
            pattern_shape=pattern_shape,
            title=title,
            marginal="violin",
            hover_data=self.data.columns
        )
        fig.show()

        if self.save_path is not None:
            fig.write_image(self.save_path)

    def cumulative(self, x: str, y: str, color: str = None, title: str = "") -> None:
        fig = go.Figure(data=[go.Histogram(
            x=self.data[x],
            y=self.data[y],
            cumulative_enabled=True)])
        fig.show()

if __name__ == "__main__":
    df = pd.read_csv("../../data/processed/train.csv")
    visualizer = DataVisualizer(df)
    visualizer.heatmap()