"""
gui/tab_result_analysis.py
결과 분석 탭.
시뮬레이션 결과와 실험 데이터를 비교 분석 (MAPE / SMAPE / RMSE).
연결기 데이터 분석 포함.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QFileDialog, QSplitter, QTextEdit,
    QTableWidget, QTableWidgetItem, QTabWidget, QComboBox,
    QFormLayout, QFrame,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import numpy as np


class ResultAnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sim_result = None
        self._init_ui()

    def set_sim_result(self, result):
        """tab_simulation에서 완료된 결과를 전달받음."""
        self._sim_result = result
        self.lbl_sim_status.setText("  ✔  시뮬레이션 결과 수신됨  ")
        self.lbl_sim_status.setStyleSheet(
            "background:#2E7D32; color:white; font-weight:bold;"
            "padding:4px 8px; border-radius:4px; font-size:12px;")

    def _init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)

        sub_tabs = QTabWidget()
        sub_tabs.addTab(self._build_validation_tab(), "📊  수치 검증 (실험 vs 시뮬)")
        sub_tabs.addTab(self._build_coupling_analysis_tab(), "🔗  연결기 데이터 분석")
        main.addWidget(sub_tabs)

    # ─ 서브탭1: 수치 검증 ─────────────────────────────────────────────
    def _build_validation_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(8)

        # 시뮬 결과 상태 배지
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("시뮬레이션 결과:"))
        self.lbl_sim_status = QLabel("  ✖  없음 — ② 시뮬레이션 탭에서 먼저 실행하세요  ")
        self.lbl_sim_status.setStyleSheet(
            "background:#C62828; color:white; font-weight:bold;"
            "padding:4px 8px; border-radius:4px; font-size:12px;")
        status_row.addWidget(self.lbl_sim_status)
        status_row.addStretch()
        vl.addLayout(status_row)

        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        vl.addWidget(line)

        # 입력 폼
        input_box = QGroupBox("실험 데이터 설정")
        form = QFormLayout(input_box)
        form.setSpacing(8)

        exp_row = QHBoxLayout()
        self.edit_exp = QLineEdit()
        self.edit_exp.setReadOnly(True)
        self.edit_exp.setPlaceholderText("실험 데이터 CSV 파일 경로...")
        btn_exp = QPushButton("찾기")
        btn_exp.setFixedWidth(60)
        btn_exp.clicked.connect(lambda: self._browse(self.edit_exp))
        exp_row.addWidget(self.edit_exp)
        exp_row.addWidget(btn_exp)
        form.addRow("실험 데이터 CSV:", exp_row)

        self.edit_time_col = QLineEdit("time")
        form.addRow("시간 컬럼명:", self.edit_time_col)

        self.edit_disp_col = QLineEdit("displacement")
        form.addRow("변위 컬럼명:", self.edit_disp_col)

        vl.addWidget(input_box)

        # 실행 버튼
        btn_row = QHBoxLayout()
        self.btn_compare = QPushButton("▶  오차 분석 실행")
        self.btn_compare.setStyleSheet(
            "background:#F57C00; color:white; font-weight:bold;"
            "font-size:13px; border-radius:5px; padding:5px 16px;")
        self.btn_compare.setFixedHeight(36)
        self.btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare.clicked.connect(self._run_validation)
        btn_row.addWidget(self.btn_compare)
        btn_row.addStretch()
        vl.addLayout(btn_row)

        # 결과 표시 (좌: 지표 테이블 / 우: 비교 플롯)
        from gui.widgets.plot_canvas import PlotCanvas
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_w = QWidget()
        lv = QVBoxLayout(left_w)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.addWidget(QLabel("오차 지표"))
        self.metric_table = QTableWidget(0, 2)
        self.metric_table.setHorizontalHeaderLabels(["지표", "값"])
        self.metric_table.horizontalHeader().setStretchLastSection(True)
        lv.addWidget(self.metric_table)
        left_w.setMinimumWidth(220)
        left_w.setMaximumWidth(300)

        self.cmp_canvas = PlotCanvas(figsize=(7, 4))
        splitter.addWidget(left_w)
        splitter.addWidget(self.cmp_canvas)
        splitter.setSizes([260, 600])
        vl.addWidget(splitter, stretch=1)
        return w

    # ─ 서브탭2: 연결기 데이터 분석 ───────────────────────────────────
    def _build_coupling_analysis_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(8)

        input_box = QGroupBox("데이터 파일 설정")
        form = QFormLayout(input_box)
        form.setSpacing(8)

        file_row = QHBoxLayout()
        self.edit_coupler_csv = QLineEdit()
        self.edit_coupler_csv.setReadOnly(True)
        self.edit_coupler_csv.setPlaceholderText("연결기 데이터 CSV 파일 경로...")
        btn_c = QPushButton("찾기")
        btn_c.setFixedWidth(60)
        btn_c.clicked.connect(self._browse_coupler_csv)
        file_row.addWidget(self.edit_coupler_csv)
        file_row.addWidget(btn_c)
        form.addRow("연결기 데이터 CSV:", file_row)

        xy_row = QHBoxLayout()
        xy_row.setSpacing(8)
        xy_row.addWidget(QLabel("X축:"))
        self.combo_x = QComboBox()
        self.combo_x.setMinimumWidth(140)
        xy_row.addWidget(self.combo_x)
        xy_row.addSpacing(12)
        xy_row.addWidget(QLabel("Y축:"))
        self.combo_y = QComboBox()
        self.combo_y.setMinimumWidth(140)
        xy_row.addWidget(self.combo_y)
        xy_row.addStretch()
        form.addRow("축 선택:", xy_row)

        vl.addWidget(input_box)

        btn_row = QHBoxLayout()
        btn_plot = QPushButton("▶  그래프 표시")
        btn_plot.setStyleSheet(
            "background:#455A64; color:white; font-weight:bold;"
            "font-size:13px; border-radius:5px; padding:5px 16px;")
        btn_plot.setFixedHeight(36)
        btn_plot.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_plot.clicked.connect(self._plot_coupling_data)
        btn_row.addWidget(btn_plot)
        btn_row.addStretch()
        vl.addLayout(btn_row)

        from gui.widgets.plot_canvas import PlotCanvas
        self.couple_canvas = PlotCanvas(figsize=(9, 5))
        vl.addWidget(self.couple_canvas, stretch=1)
        return w

    # ──────────────────────────────────────────────────────────────────
    def _browse(self, edit: QLineEdit):
        p, _ = QFileDialog.getOpenFileName(self, "파일 선택", "",
                                           "CSV / Excel (*.csv *.xlsx)")
        if p:
            edit.setText(p)

    def _browse_coupler_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "파일 선택", "",
                                           "CSV / Excel (*.csv *.xlsx)")
        if p:
            self.edit_coupler_csv.setText(p)
            self._load_columns()     # 파일 선택 즉시 컬럼 자동 로드

    def _run_validation(self):
        from PyQt6.QtWidgets import QMessageBox
        if self._sim_result is None:
            QMessageBox.warning(self, "경고",
                                "먼저 ② 시뮬레이션 탭에서 시뮬레이션을 실행하세요.")
            return
        exp_path = self.edit_exp.text().strip()
        if not exp_path:
            QMessageBox.warning(self, "경고", "실험 데이터 파일을 선택하세요.")
            return
        try:
            import pandas as pd
            df_exp = pd.read_csv(exp_path)
            t_col = self.edit_time_col.text().strip()
            d_col = self.edit_disp_col.text().strip()

            exp_t = df_exp[t_col].values
            exp_d = df_exp[d_col].values

            sim_t = np.array(self._sim_result.get('time', []))
            sim_d = np.array(self._sim_result.get('displacement', []))
            if len(sim_d) == 0:
                disp_list = self._sim_result.get('displacement_list', [])
                if disp_list:
                    sim_d = np.array(disp_list[0])

            from scipy.interpolate import interp1d
            if len(sim_t) and len(exp_t):
                t_min = max(sim_t[0], exp_t[0])
                t_max = min(sim_t[-1], exp_t[-1])
                t_common = np.linspace(t_min, t_max, 500)
                f_sim = interp1d(sim_t, sim_d, fill_value='extrapolate')
                f_exp = interp1d(exp_t, exp_d, fill_value='extrapolate')
                s = f_sim(t_common)
                e = f_exp(t_common)
            else:
                s, e = sim_d, exp_d

            from core.simulation import f_test_simulation_numerical_validation
            metrics = f_test_simulation_numerical_validation(s, e)

            self.metric_table.setRowCount(0)
            for k, v in metrics.items():
                r = self.metric_table.rowCount()
                self.metric_table.insertRow(r)
                self.metric_table.setItem(r, 0, QTableWidgetItem(k))
                self.metric_table.setItem(r, 1, QTableWidgetItem(f"{v:.4f}"))

            ax = self.cmp_canvas.fig.clear()
            ax = self.cmp_canvas.fig.add_subplot(111)
            if len(sim_t):
                ax.plot(sim_t, sim_d, label='Simulation', color='royalblue')
            if len(exp_t):
                ax.plot(exp_t, exp_d, label='Experiment', color='tomato',
                        linestyle='--')
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Displacement (mm)")
            ax.set_title("시뮬레이션 vs 실험 비교")
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.cmp_canvas.canvas.draw()

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "분석 오류", str(e))

    def _load_columns(self):
        path = self.edit_coupler_csv.text().strip()
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path, nrows=0)
            cols = list(df.columns)
            for cb in (self.combo_x, self.combo_y):
                cb.clear()
                cb.addItems(cols)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "오류", str(e))

    def _plot_coupling_data(self):
        path = self.edit_coupler_csv.text().strip()
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            x_col = self.combo_x.currentText()
            y_col = self.combo_y.currentText()
            ax = self.couple_canvas.fig.clear()
            ax = self.couple_canvas.fig.add_subplot(111)
            ax.plot(df[x_col].values, df[y_col].values, marker='o',
                    markersize=3, linewidth=1, color='steelblue')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} vs {x_col}")
            ax.grid(True, alpha=0.3)
            self.couple_canvas.canvas.draw()
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "플롯 오류", str(e))
