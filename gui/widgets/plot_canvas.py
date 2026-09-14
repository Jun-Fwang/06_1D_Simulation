"""
gui/widgets/plot_canvas.py
Matplotlib FigureCanvasQTAgg 래퍼.
"""
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout


class PlotCanvas(QWidget):
    def __init__(self, parent=None, nrows=1, ncols=1, figsize=(7, 4)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._nrows = nrows
        self._ncols = ncols
        self.axes = self.fig.subplots(nrows, ncols)

    def clear(self):
        """모든 axes 초기화."""
        self.fig.clear()
        self.axes = self.fig.subplots(self._nrows, self._ncols)

    def draw(self):
        self.canvas.draw_idle()
