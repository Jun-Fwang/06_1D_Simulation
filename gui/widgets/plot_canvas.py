"""
gui/widgets/plot_canvas.py
Matplotlib FigureCanvasQTAgg 래퍼.
범례 체크박스를 통한 데이터 계열(시리즈) On/Off 기능 지원.
"""
import platform
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame,
    QCheckBox, QPushButton, QLabel
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QBrush, QColor, QPen

if platform.system() == "Windows":
    matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False


def _make_color_icon(color_hex):
    """범례 색상 사각형 아이콘 생성."""
    pix = QPixmap(14, 14)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    try:
        p.setBrush(QBrush(QColor(color_hex)))
    except Exception:
        p.setBrush(QBrush(QColor("#1565C0")))
    p.setPen(QPen(QColor("#78909C"), 1))
    p.drawRoundedRect(1, 1, 12, 12, 2, 2)
    p.end()
    return QIcon(pix)


class PlotCanvas(QWidget):
    def __init__(self, parent=None, nrows=1, ncols=1, figsize=(7, 4)):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # ── 범례 체크박스 가로 스크롤 툴바 ──
        self.legend_scroll = QScrollArea()
        self.legend_scroll.setFixedHeight(36)
        self.legend_scroll.setWidgetResizable(True)
        self.legend_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.legend_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.legend_scroll.setStyleSheet(
            "QScrollArea { background: #FAFAFA; border-bottom: 1px solid #CFD8DC; }"
        )
        self.legend_scroll.setVisible(False)

        self.legend_container = QWidget()
        self.legend_container.setStyleSheet("background: transparent;")
        self.legend_layout = QHBoxLayout(self.legend_container)
        self.legend_layout.setContentsMargins(8, 2, 8, 2)
        self.legend_layout.setSpacing(10)
        self.legend_scroll.setWidget(self.legend_container)

        self._checkboxes = []
        self._current_ax = None
        self._current_lines = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.legend_scroll)
        layout.addWidget(self.canvas)

        self._nrows = nrows
        self._ncols = ncols
        self.axes = self.fig.subplots(nrows, ncols)

    def clear(self):
        """모든 axes 및 범례 툴바 초기화."""
        self.fig.clear()
        self.axes = self.fig.subplots(self._nrows, self._ncols)
        self._clear_legend_toolbar()

    def _clear_legend_toolbar(self):
        while self.legend_layout.count():
            item = self.legend_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._checkboxes.clear()
        self._current_lines.clear()
        self._current_ax = None
        self.legend_scroll.setVisible(False)

    def draw(self):
        self.canvas.draw_idle()

    def set_checkable_legend(self, ax, line_entries):
        """
        플롯의 데이터 계열에 대해 켜고 끌 수 있는 범례 체크박스들을 구성한다.
        line_entries: list of Line2D 또는 list of (Line2D, label, color)
        """
        self._clear_legend_toolbar()
        if not line_entries:
            return

        self._current_ax = ax
        self.legend_scroll.setVisible(True)

        lbl = QLabel("범례:")
        lbl.setStyleSheet("font-weight: bold; font-size: 11px; color: #455A64;")
        self.legend_layout.addWidget(lbl)

        for item in line_entries:
            if isinstance(item, (tuple, list)):
                line, label, color = item[0], item[1], item[2]
            else:
                line = item
                label = line.get_label()
                color = line.get_color()

            cb = QCheckBox(label)
            cb.setChecked(line.get_visible())
            cb.setIcon(_make_color_icon(color))
            cb.setCursor(Qt.CursorShape.PointingHandCursor)
            cb.setStyleSheet("""
                QCheckBox {
                    font-size: 11px;
                    font-weight: 500;
                    color: #263238;
                    spacing: 4px;
                    padding: 2px 4px;
                }
                QCheckBox:hover {
                    background-color: #ECEFF1;
                    border-radius: 3px;
                }
            """)

            def _make_handler(target_line):
                def _handler(checked):
                    target_line.set_visible(checked)
                    self._update_plot_visibility(ax)
                return _handler

            cb.toggled.connect(_make_handler(line))
            self.legend_layout.addWidget(cb)
            self._checkboxes.append(cb)
            self._current_lines.append(line)

        self.legend_layout.addStretch()

        btn_all = QPushButton("전체")
        btn_all.setFixedHeight(22)
        btn_all.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_all.setStyleSheet("""
            QPushButton {
                background: #ECEFF1; color: #37474F; font-size: 10px; font-weight: bold;
                border: 1px solid #CFD8DC; border-radius: 3px; padding: 0 6px;
            }
            QPushButton:hover { background: #CFD8DC; }
        """)
        btn_all.clicked.connect(self._select_all_series)
        self.legend_layout.addWidget(btn_all)

        btn_none = QPushButton("해제")
        btn_none.setFixedHeight(22)
        btn_none.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_none.setStyleSheet("""
            QPushButton {
                background: #ECEFF1; color: #37474F; font-size: 10px; font-weight: bold;
                border: 1px solid #CFD8DC; border-radius: 3px; padding: 0 6px;
            }
            QPushButton:hover { background: #CFD8DC; }
        """)
        btn_none.clicked.connect(self._deselect_all_series)
        self.legend_layout.addWidget(btn_none)

    def _select_all_series(self):
        for cb in self._checkboxes:
            cb.setChecked(True)

    def _deselect_all_series(self):
        for cb in self._checkboxes:
            cb.setChecked(False)

    def _update_plot_visibility(self, ax):
        # Matplotlib 내부 범례 업데이트
        handles, labels = ax.get_legend_handles_labels()
        vis = [(h, l) for h, l in zip(handles, labels) if h.get_visible()]
        if vis:
            h_list, l_list = zip(*vis)
            ax.legend(h_list, l_list, loc='best', ncol=2, fontsize=9,
                      frameon=True, framealpha=0.85, shadow=True)
        else:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        # 보이는 선이 있으면 Y축 자동 재스케일링
        if any(line.get_visible() for line in self._current_lines):
            try:
                ax.relim(visible_only=True)
                ax.autoscale_view(scalex=False, scaley=True)
            except Exception:
                pass

        self.canvas.draw_idle()
