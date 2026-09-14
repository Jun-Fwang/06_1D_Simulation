"""
gui/tab_result_analysis.py
결과 분석 탭.

서브탭 구성
  1. 수치 검증       - 실험 vs 시뮬 MAPE/SMAPE/RMSE 비교
  2. 결과 요약 테이블 - 에너지 수지 / 연결기 하중-변위-에너지 / 차량 운동 정보
  3. 결과 그래프     - Energy Balance / Time-Force / F-D / Time-Acceleration
  4. 연결기 데이터 분석 - CSV 파일 로드 -> 커스텀 플롯
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QFileDialog, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QComboBox,
    QFormLayout, QFrame, QHeaderView, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont
import numpy as np

_HDR_COLOR  = "#1565C0"
_HDR_FG     = "#FFFFFF"
_ROW_A      = "#EBF5FB"
_GRID_COLOR = "#D5D8DC"
_NO_DATA_STYLE = "color:#9E9E9E; font-size:14px; font-style:italic;"

_LINE_COLORS = [
    "#1565C0", "#E53935", "#43A047", "#FB8C00",
    "#8E24AA", "#00838F", "#6D4C41", "#039BE5",
    "#7CB342", "#F4511E", "#3949AB", "#00ACC1",
]


def _make_table(col_headers):
    t = QTableWidget(0, len(col_headers))
    t.setHorizontalHeaderLabels(col_headers)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    t.setAlternatingRowColors(True)
    t.setStyleSheet(f"""
        QTableWidget {{ gridline-color: {_GRID_COLOR}; font-size: 12px; }}
        QHeaderView::section {{
            background-color: {_HDR_COLOR}; color: {_HDR_FG};
            font-weight: bold; font-size: 12px; padding: 5px; border: none;
        }}
        QTableWidget::item {{ padding: 4px 8px; }}
        QTableWidget::item:alternate {{ background: {_ROW_A}; }}
    """)
    return t


def _fill_table(table, rows):
    table.setRowCount(0)
    for row_data in rows:
        r = table.rowCount()
        table.insertRow(r)
        for c, val in enumerate(row_data):
            item = QTableWidgetItem(str(val))
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter if c > 0
                else Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
            )
            table.setItem(r, c, item)


class ResultAnalysisTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sim_result = None
        self._mode = 1   # 0=Rigid Wall, 1=Coupling
        self._n_car = 0
        self._n_coupler = 0
        self._init_ui()

    def set_sim_result(self, result, mode: int = 1):
        self._sim_result = result
        self._mode = mode
        self._detect_topology(result)
        self.lbl_sim_status.setText("  OK  시뮬레이션 결과 수신됨  ")
        self.lbl_sim_status.setStyleSheet(
            "background:#2E7D32; color:white; font-weight:bold;"
            "padding:4px 8px; border-radius:4px; font-size:12px;")
        self.btn_save.setEnabled(True)
        try:
            import pandas as pd
            if isinstance(result, pd.DataFrame) and not result.empty:
                self._refresh_tables(result)
                self._refresh_plots(result)
                self.lbl_table_status.setText("")
                self.lbl_plot_status.setText("")
        except Exception as e:
            self.lbl_table_status.setText(f"테이블 갱신 실패: {e}")
            self.lbl_plot_status.setText(f"그래프 갱신 실패: {e}")

    def _init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)

        # 상단 상태 배지 + 저장 버튼
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("시뮬레이션 결과:"))
        self.lbl_sim_status = QLabel("  X  없음 - 시뮬레이션 탭에서 먼저 실행하세요  ")
        self.lbl_sim_status.setStyleSheet(
            "background:#C62828; color:white; font-weight:bold;"
            "padding:4px 8px; border-radius:4px; font-size:12px;")
        status_row.addWidget(self.lbl_sim_status)
        status_row.addStretch()
        self.btn_save = QPushButton("💾  결과 저장 (CSV)")
        self.btn_save.setFixedHeight(30)
        self.btn_save.setEnabled(False)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet(
            "background:#455A64; color:white; font-weight:bold;"
            "border-radius:4px; font-size:12px; padding:0 10px;")
        self.btn_save.clicked.connect(self._save_result)
        status_row.addWidget(self.btn_save)
        main.addLayout(status_row)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        main.addWidget(line)

        self.sub_tabs = QTabWidget()
        self.tab_table = self._build_table_tab()
        self.tab_plot = self._build_plot_tab()
        self.tab_val = self._build_validation_tab()
        self.tab_coupling = self._build_coupling_analysis_tab()

        self.sub_tabs.addTab(self.tab_table,    "결과 요약 테이블")
        self.sub_tabs.addTab(self.tab_plot,     "결과 그래프")
        self.sub_tabs.addTab(self.tab_val,      "수치 검증 (실험 vs 시뮬)")
        self.sub_tabs.addTab(self.tab_coupling, "연결기 데이터 분석")
        main.addWidget(self.sub_tabs, stretch=1)

    def set_ai_tabs_visible(self, visible: bool):
        """AI 모델링 토글 상태에 따라 수치 검증 및 연결기 데이터 분석 탭을 숨기거나 표시한다."""
        val_idx = self.sub_tabs.indexOf(self.tab_val)
        coup_idx = self.sub_tabs.indexOf(self.tab_coupling)
        if val_idx != -1:
            self.sub_tabs.setTabVisible(val_idx, visible)
        if coup_idx != -1:
            self.sub_tabs.setTabVisible(coup_idx, visible)
        if not visible and self.sub_tabs.currentIndex() in (val_idx, coup_idx):
            self.sub_tabs.setCurrentIndex(0)

    def _build_validation_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(8)
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
        btn_row = QHBoxLayout()
        self.btn_compare = QPushButton("  오차 분석 실행")
        self.btn_compare.setStyleSheet(
            "background:#F57C00; color:white; font-weight:bold;"
            "font-size:13px; border-radius:5px; padding:5px 16px;")
        self.btn_compare.setFixedHeight(36)
        self.btn_compare.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_compare.clicked.connect(self._run_validation)
        btn_row.addWidget(self.btn_compare)
        btn_row.addStretch()
        vl.addLayout(btn_row)
        from gui.widgets.plot_canvas import PlotCanvas
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_w = QWidget()
        lv = QVBoxLayout(left_w)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.addWidget(QLabel("오차 지표"))
        self.metric_table = _make_table(["지표", "값"])
        lv.addWidget(self.metric_table)
        left_w.setMinimumWidth(220)
        left_w.setMaximumWidth(300)
        self.cmp_canvas = PlotCanvas(figsize=(7, 4))
        splitter.addWidget(left_w)
        splitter.addWidget(self.cmp_canvas)
        splitter.setSizes([260, 600])
        vl.addWidget(splitter, stretch=1)
        return w

    def _build_table_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(6, 6, 6, 6)
        self.lbl_table_status = QLabel("시뮬레이션 결과가 수신되면 자동으로 표시됩니다.")
        self.lbl_table_status.setStyleSheet(_NO_DATA_STYLE)
        self.lbl_table_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(self.lbl_table_status)
        btn_refresh = QPushButton("새로고침")
        btn_refresh.setFixedWidth(120)
        btn_refresh.setFixedHeight(30)
        btn_refresh.setStyleSheet(
            "background:#546E7A; color:white; font-weight:bold;"
            "border-radius:4px; font-size:12px;")
        btn_refresh.clicked.connect(self._on_refresh_tables)
        vl.addWidget(btn_refresh, alignment=Qt.AlignmentFlag.AlignRight)
        inner = QTabWidget()
        self.tbl_energy = _make_table(["항목", "값"])
        tab_e = QWidget(); le = QVBoxLayout(tab_e); le.setContentsMargins(4,4,4,4)
        le.addWidget(self.tbl_energy)
        inner.addTab(tab_e, "에너지 수지")
        self.tbl_coupler = _make_table(
            ["Coupler", "Max Force [kN]", "Max Stroke [mm]", "Max Energy [kJ]"])
        tab_c = QWidget(); lc = QVBoxLayout(tab_c); lc.setContentsMargins(4,4,4,4)
        lc.addWidget(self.tbl_coupler)
        inner.addTab(tab_c, "연결기 하중/변위/에너지")
        self.tbl_car = _make_table(
            ["Car", "Max Travel [mm]", "Max Vel [km/h]",
             "Max Acc [g]", "Max Acc 30ms [g]", "Max Acc 120ms [g]"])
        tab_k = QWidget(); lk = QVBoxLayout(tab_k); lk.setContentsMargins(4,4,4,4)
        lk.addWidget(self.tbl_car)
        inner.addTab(tab_k, "차량 운동 정보")
        vl.addWidget(inner, stretch=1)
        return w

    def _on_refresh_tables(self):
        if self._sim_result is None:
            QMessageBox.information(self, "알림",
                "먼저 시뮬레이션 탭에서 시뮬레이션을 실행하세요.")
            return
        try:
            import pandas as pd
            if isinstance(self._sim_result, pd.DataFrame):
                self._refresh_tables(self._sim_result)
                self.lbl_table_status.setText("")
        except Exception as e:
            QMessageBox.critical(self, "오류", str(e))

    def _refresh_tables(self, df):
        n_car = self._n_car
        n_coupler = self._n_coupler
        try:
            ke0 = df.iloc[0].get('Kinetic_Energy [kJ]', 0)
            ke1 = df.iloc[-1].get('Kinetic_Energy [kJ]', 0)
            ge0 = df.iloc[0].get('Global_Energy [kJ]', 1)
            ge1 = df.iloc[-1].get('Global_Energy [kJ]', 1)
            ie1 = df.iloc[-1].get('Internal_Energy [kJ]', 0)
            fe1 = df.iloc[-1].get('Friction_Energy [kJ]', 0)
            gap = f"{round((ge1/ge0 - 1)*100, 1)} %" if ge0 != 0 else "N/A"
            _fill_table(self.tbl_energy, [
                ["Global Energy Increase [%]",          gap],
                ["Starting Kinetic Energy [kJ]",         round(ke0, 1)],
                ["Final Kinetic Energy [kJ]",            round(ke1, 1)],
                ["Energy Dissipation of Couplers [kJ]", round(ie1, 1)],
                ["Energy Dissipation of Brakes [kJ]",   round(fe1, 1)],
            ])
        except Exception:
            pass
        try:
            rows = []
            for i in range(n_coupler):
                tag = str(i).zfill(2)
                f_col = f'Coupler_Force_{tag} [kN]'
                d_col = f'Coupler_Displacement_{tag} [mm]'
                e_col = f'Coupler_Internal_Energy_{tag} [kJ]'
                max_f = round(df[f_col].max(), 1) if f_col in df.columns else "N/A"
                max_d = round(df[d_col].max(), 1) if d_col in df.columns else "N/A"
                max_e = round(df[e_col].max(), 1) if e_col in df.columns else "N/A"
                rows.append([f"Coupler_{tag}", max_f, max_d, max_e])
            _fill_table(self.tbl_coupler, rows)
        except Exception:
            pass
        try:
            time_step = (df['Time [ms]'].iloc[1]
                         if 'Time [ms]' in df.columns and len(df) > 1 else 1)
            rows = []
            for i in range(n_car):
                tag = str(i).zfill(2)
                tr_col  = f'Car_Displacement_Difference_{tag} [mm]'
                vel_col = f'Car_Velocity_{tag} [km/h]'
                acc_col = f'Car_Acceleration_{tag} [g]'
                def _peak(col):
                    if col not in df.columns:
                        return "N/A"
                    mn, mx = df[col].min(), df[col].max()
                    return round(mn if abs(mn) > abs(mx) else mx, 1)
                max_tr  = round(df[tr_col].max(), 1)  if tr_col  in df.columns else "N/A"
                max_vel = round(df[vel_col].max(), 1) if vel_col in df.columns else "N/A"
                max_acc = _peak(acc_col)
                if acc_col in df.columns and time_step > 0:
                    s = df[acc_col]
                    w30  = max(1, int(30  / time_step))
                    w120 = max(1, int(120 / time_step))
                    def _ps(sr):
                        mn, mx = sr.min(), sr.max()
                        return round(mn if abs(mn) > abs(mx) else mx, 1)
                    max_acc30  = _ps(s.rolling(w30,  min_periods=1).mean())
                    max_acc120 = _ps(s.rolling(w120, min_periods=1).mean())
                else:
                    max_acc30 = max_acc120 = "N/A"
                rows.append([f"Car_{tag}", max_tr, max_vel,
                              max_acc, max_acc30, max_acc120])
            _fill_table(self.tbl_car, rows)
        except Exception:
            pass

    def _build_plot_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(6, 6, 6, 6)
        self.lbl_plot_status = QLabel("시뮬레이션 결과가 수신되면 자동으로 표시됩니다.")
        self.lbl_plot_status.setStyleSheet(_NO_DATA_STYLE)
        self.lbl_plot_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(self.lbl_plot_status)
        btn_refresh = QPushButton("새로고침")
        btn_refresh.setFixedWidth(120)
        btn_refresh.setFixedHeight(30)
        btn_refresh.setStyleSheet(
            "background:#546E7A; color:white; font-weight:bold;"
            "border-radius:4px; font-size:12px;")
        btn_refresh.clicked.connect(self._on_refresh_plots)
        vl.addWidget(btn_refresh, alignment=Qt.AlignmentFlag.AlignRight)
        from gui.widgets.plot_canvas import PlotCanvas
        inner = QTabWidget()

        # ── 기존 4개 그래프 ────────────────────────────────────────────
        self.canvas_energy     = PlotCanvas(figsize=(10, 5))
        self.canvas_time_force = PlotCanvas(figsize=(10, 5))
        self.canvas_fd         = PlotCanvas(figsize=(10, 5))
        self.canvas_time_acc   = PlotCanvas(figsize=(10, 5))
        inner.addTab(self.canvas_energy,     "Energy Balance")
        inner.addTab(self.canvas_time_force, "Time - Force")
        inner.addTab(self.canvas_fd,         "Force - Displacement")
        inner.addTab(self.canvas_time_acc,   "Time - Acceleration")

        # ── 시뮬 결과 개별 탭 4개 (구 '2×2 개요' 대체) ────────────────
        self.canvas_ov0 = PlotCanvas(figsize=(10, 5))
        self.canvas_ov1 = PlotCanvas(figsize=(10, 5))
        self.canvas_ov2 = PlotCanvas(figsize=(10, 5))
        self.canvas_ov3 = PlotCanvas(figsize=(10, 5))
        self._inner_plot_tabs = inner   # 모드 변경 시 탭 이름 업데이트용
        inner.addTab(self.canvas_ov0, "시뮬 결과 1")
        inner.addTab(self.canvas_ov1, "시뮬 결과 2")
        inner.addTab(self.canvas_ov2, "시뮬 결과 3")
        inner.addTab(self.canvas_ov3, "시뮬 결과 4")
        # 실제 지수는 개요 탭 4개가 다음 4개 탭 이후에 시작됨 (인덱스 4–7)
        self._ov_tab_start = 4   # Energy/Force/FD/Acc 다음

        vl.addWidget(inner, stretch=1)
        return w

    def _on_refresh_plots(self):
        if self._sim_result is None:
            QMessageBox.information(self, "알림",
                "먼저 시뮬레이션 탭에서 시뮬레이션을 실행하세요.")
            return
        try:
            import pandas as pd
            if isinstance(self._sim_result, pd.DataFrame):
                self._refresh_plots(self._sim_result)
                self.lbl_plot_status.setText("")
        except Exception as e:
            QMessageBox.critical(self, "오류", str(e))

    def _save_result(self):
        if self._sim_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", "simulation_result.csv", "CSV (*.csv)")
        if path:
            try:
                self._sim_result.to_csv(path, index=False, encoding='utf-8-sig')
            except Exception as e:
                QMessageBox.critical(self, "저장 오류", str(e))

    def _refresh_plots(self, df):
        n_car = self._n_car
        n_coupler = self._n_coupler
        time = df['Time [ms]'].values if 'Time [ms]' in df.columns else None

        def _styled_ax(ax, title, xlabel, ylabel):
            ax.set_title(title, fontweight='bold', fontsize=13, pad=8)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', ncol=2, fontsize=9,
                      frameon=True, framealpha=0.85, shadow=True)
            ax.tick_params(axis='both', labelsize=9)

        try:
            self.canvas_energy.fig.clear()
            ax = self.canvas_energy.fig.add_subplot(111)
            pairs = [
                ('Global Energy [kJ]',   '#1565C0', 'Global_Energy [kJ]'),
                ('Kinetic Energy [kJ]',  '#E53935', 'Kinetic_Energy [kJ]'),
                ('Internal Energy [kJ]', '#43A047', 'Internal_Energy [kJ]'),
                ('Friction Energy [kJ]', '#FB8C00', 'Friction_Energy [kJ]'),
            ]
            lines = []
            if time is not None:
                for label, color, col in pairs:
                    if col in df.columns:
                        l, = ax.plot(time, df[col].values, label=label, color=color, linewidth=2)
                        lines.append((l, label, color))
            _styled_ax(ax, 'Energy Balance', 'Time [ms]', 'Energy [kJ]')
            self.canvas_energy.fig.tight_layout()
            self.canvas_energy.set_checkable_legend(ax, lines)
            self.canvas_energy.canvas.draw()
        except Exception:
            pass

        try:
            self.canvas_time_force.fig.clear()
            ax = self.canvas_time_force.fig.add_subplot(111)
            lines = []
            if time is not None:
                for i in range(n_coupler):
                    tag = str(i).zfill(2)
                    col = f'Coupler_Force_{tag} [kN]'
                    if col in df.columns:
                        c = _LINE_COLORS[i % len(_LINE_COLORS)]
                        lbl = f'Coupler_{tag}'
                        l, = ax.plot(time, df[col].values,
                                label=lbl,
                                color=c,
                                linewidth=2)
                        lines.append((l, lbl, c))
            _styled_ax(ax, 'Time - Coupler Force', 'Time [ms]', 'Force [kN]')
            self.canvas_time_force.fig.tight_layout()
            self.canvas_time_force.set_checkable_legend(ax, lines)
            self.canvas_time_force.canvas.draw()
        except Exception:
            pass

        try:
            self.canvas_fd.fig.clear()
            ax = self.canvas_fd.fig.add_subplot(111)
            lines = []
            for i in range(n_coupler):
                tag = str(i).zfill(2)
                d_col = f'Coupler_Displacement_{tag} [mm]'
                f_col = f'Coupler_Force_{tag} [kN]'
                if d_col in df.columns and f_col in df.columns:
                    c = _LINE_COLORS[i % len(_LINE_COLORS)]
                    lbl = f'Coupler_{tag}'
                    l, = ax.plot(df[d_col].values, df[f_col].values,
                            label=lbl,
                            color=c,
                            linewidth=2)
                    lines.append((l, lbl, c))
            _styled_ax(ax, 'Force - Displacement (F-D Curve)',
                       'Displacement [mm]', 'Force [kN]')
            self.canvas_fd.fig.tight_layout()
            self.canvas_fd.set_checkable_legend(ax, lines)
            self.canvas_fd.canvas.draw()
        except Exception:
            pass

        try:
            self.canvas_time_acc.fig.clear()
            ax = self.canvas_time_acc.fig.add_subplot(111)
            lines = []
            if time is not None:
                for i in range(n_car):
                    tag = str(i).zfill(2)
                    col = f'Car_Acceleration_{tag} [g]'
                    if col in df.columns:
                        c = _LINE_COLORS[i % len(_LINE_COLORS)]
                        lbl = f'Car_{tag}'
                        l, = ax.plot(time, df[col].values,
                                label=lbl,
                                color=c,
                                linewidth=1.5)
                        lines.append((l, lbl, c))
            _styled_ax(ax, 'Time - Car Acceleration', 'Time [ms]', 'Acceleration [g]')
            self.canvas_time_acc.fig.tight_layout()
            self.canvas_time_acc.set_checkable_legend(ax, lines)
            self.canvas_time_acc.canvas.draw()
        except Exception:
            pass

        # 시뮬 결과 개별 탭 4개 갱신
        self._refresh_overview(df)

    def _refresh_overview(self, df):
        """구 '결과 확인' 탭 콘텐츠 — 4개 개별 탭으로 분리."""
        time = df['Time [ms]'].values if 'Time [ms]' in df.columns else None
        if time is None:
            return

        def _draw(canvas, title, xlabel, ylabel, plot_fn):
            try:
                canvas.fig.clear()
                ax = canvas.fig.add_subplot(111)
                lines = plot_fn(ax)
                ax.set_title(title, fontweight='bold', fontsize=13, pad=8)
                ax.set_xlabel(xlabel, fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.tick_params(axis='both', labelsize=9)
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(loc='best', ncol=2, fontsize=9,
                              frameon=True, framealpha=0.85, shadow=True)
                canvas.fig.tight_layout()
                if lines:
                    canvas.set_checkable_legend(ax, lines)
                canvas.canvas.draw()
            except Exception:
                pass

        inner = self._inner_plot_tabs
        base = self._ov_tab_start   # 4

        if self._mode == 0:   # ── Rigid Wall ──
            def p0(ax):
                lines = []
                if 'Buffer_Displacement [mm]' in df.columns:
                    l, = ax.plot(time, df['Buffer_Displacement [mm]'].values,
                            label='Buffer Displacement', color='steelblue', linewidth=2)
                    lines.append((l, 'Buffer Displacement', 'steelblue'))
                return lines
            def p1(ax):
                lines = []
                if 'Buffer_Velocity [mm/ms]' in df.columns:
                    l, = ax.plot(time, df['Buffer_Velocity [mm/ms]'].values,
                            label='Buffer Velocity', color='darkorange', linewidth=2)
                    lines.append((l, 'Buffer Velocity', 'darkorange'))
                return lines
            def p2(ax):
                lines = []
                if 'Buffer_Force [kN]' in df.columns:
                    l, = ax.plot(time, df['Buffer_Force [kN]'].values,
                            label='Impact Force', color='tomato', linewidth=2)
                    lines.append((l, 'Impact Force', 'tomato'))
                return lines
            def p3(ax):
                lines = []
                if 'Buffer_Displacement [mm]' in df.columns and 'Buffer_Force [kN]' in df.columns:
                    l, = ax.plot(df['Buffer_Displacement [mm]'].values,
                            df['Buffer_Force [kN]'].values,
                            label='F-D Curve', color='mediumvioletred', linewidth=2)
                    lines.append((l, 'F-D Curve', 'mediumvioletred'))
                return lines

            _draw(self.canvas_ov0, 'Buffer Displacement', 'Time [ms]', 'Displacement [mm]', p0)
            _draw(self.canvas_ov1, 'Buffer Velocity',     'Time [ms]', 'Velocity [mm/ms]',  p1)
            _draw(self.canvas_ov2, 'Impact Force',        'Time [ms]', 'Force [kN]',        p2)
            _draw(self.canvas_ov3, 'F-D Curve',  'Displacement [mm]', 'Force [kN]',        p3)

            inner.setTabText(base + 0, 'Buffer 변위')
            inner.setTabText(base + 1, 'Buffer 속도')
            inner.setTabText(base + 2, '충격 하중')
            inner.setTabText(base + 3, 'F-D 곡선')

        else:   # ── Coupling ──
            disp_cols   = [c for c in df.columns if 'Car_Displacement_Difference' in c]
            vel_cols    = [c for c in df.columns if c.startswith('Car_Velocity_')]
            force_cols  = [c for c in df.columns if c.startswith('Coupler_Force_')]
            energy_cols = [c for c in df.columns if c.startswith('Coupler_Internal_Energy_')]

            def p0(ax):
                lines = []
                for i, col in enumerate(disp_cols):
                    c = _LINE_COLORS[i % len(_LINE_COLORS)]
                    lbl = f'Car {i+1}'
                    l, = ax.plot(time, df[col].values, label=lbl,
                            color=c, linewidth=2)
                    lines.append((l, lbl, c))
                return lines
            def p1(ax):
                lines = []
                for i, col in enumerate(vel_cols):
                    c = _LINE_COLORS[i % len(_LINE_COLORS)]
                    lbl = f'Car {i+1}'
                    l, = ax.plot(time, df[col].values, label=lbl,
                            color=c, linewidth=2)
                    lines.append((l, lbl, c))
                return lines
            def p2(ax):
                lines = []
                for i, col in enumerate(force_cols):
                    c = _LINE_COLORS[i % len(_LINE_COLORS)]
                    lbl = f'Coupler {i+1}'
                    l, = ax.plot(time, df[col].values, label=lbl,
                            color=c, linewidth=2)
                    lines.append((l, lbl, c))
                return lines
            def p3(ax):
                lines = []
                for i, col in enumerate(energy_cols):
                    c = _LINE_COLORS[i % len(_LINE_COLORS)]
                    lbl = f'Coupler {i+1}'
                    l, = ax.plot(time, df[col].values, label=lbl,
                            color=c, linewidth=2)
                    lines.append((l, lbl, c))
                return lines

            _draw(self.canvas_ov0, 'Relative Displacement by Car',
                  'Time [ms]', 'Displacement [mm]', p0)
            _draw(self.canvas_ov1, 'Velocity by Car',
                  'Time [ms]', 'Velocity [km/h]',   p1)
            _draw(self.canvas_ov2, 'Impact Force by Coupler',
                  'Time [ms]', 'Force [kN]',         p2)
            _draw(self.canvas_ov3, 'Absorbed Energy by Coupler',
                  'Time [ms]', 'Energy [kJ]',        p3)

            inner.setTabText(base + 0, '차량 상대변위')
            inner.setTabText(base + 1, '차량 속도')
            inner.setTabText(base + 2, '연결기 하중')
            inner.setTabText(base + 3, '연결기 에너지')

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
        btn_plot = QPushButton("  그래프 표시")
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

    def _detect_topology(self, result):
        try:
            import pandas as pd
            if not isinstance(result, pd.DataFrame):
                return
            self._n_car     = len([c for c in result.columns if c.startswith('Car_Acceleration_')])
            self._n_coupler = len([c for c in result.columns if c.startswith('Coupler_Force_')])
        except Exception:
            pass

    def _browse(self, edit):
        p, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", "CSV / Excel (*.csv *.xlsx)")
        if p:
            edit.setText(p)

    def _browse_coupler_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", "CSV / Excel (*.csv *.xlsx)")
        if p:
            self.edit_coupler_csv.setText(p)
            self._load_columns()

    def _run_validation(self):
        if self._sim_result is None:
            QMessageBox.warning(self, "경고", "먼저 시뮬레이션 탭에서 시뮬레이션을 실행하세요.")
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
            sim_t = np.array(self._sim_result.get('time', []) if isinstance(self._sim_result, dict) else [])
            sim_d = np.array(self._sim_result.get('displacement', []) if isinstance(self._sim_result, dict) else [])
            if len(sim_d) == 0 and isinstance(self._sim_result, dict):
                disp_list = self._sim_result.get('displacement_list', [])
                if disp_list:
                    sim_d = np.array(disp_list[0])
            from scipy.interpolate import interp1d
            if len(sim_t) and len(exp_t):
                t_min = max(sim_t[0], exp_t[0])
                t_max = min(sim_t[-1], exp_t[-1])
                t_common = np.linspace(t_min, t_max, 500)
                s = interp1d(sim_t, sim_d, fill_value='extrapolate')(t_common)
                e = interp1d(exp_t, exp_d, fill_value='extrapolate')(t_common)
            else:
                s, e = sim_d, exp_d
            from core.simulation import f_test_simulation_numerical_validation
            metrics = f_test_simulation_numerical_validation(s, e)
            _fill_table(self.metric_table, [[k, f"{v:.4f}"] for k, v in metrics.items()])
            self.cmp_canvas.fig.clear()
            ax = self.cmp_canvas.fig.add_subplot(111)
            if len(sim_t):
                ax.plot(sim_t, sim_d, label='Simulation', color='royalblue', linewidth=2)
            if len(exp_t):
                ax.plot(exp_t, exp_d, label='Experiment', color='tomato', linestyle='--', linewidth=2)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Displacement (mm)")
            ax.set_title("시뮬레이션 vs 실험 비교", fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.cmp_canvas.fig.tight_layout()
            self.cmp_canvas.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "분석 오류", str(e))

    def _load_columns(self):
        path = self.edit_coupler_csv.text().strip()
        if not path:
            return
        try:
            import pandas as pd
            cols = list(pd.read_csv(path, nrows=0).columns)
            for cb in (self.combo_x, self.combo_y):
                cb.clear()
                cb.addItems(cols)
        except Exception as e:
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
            self.couple_canvas.fig.clear()
            ax = self.couple_canvas.fig.add_subplot(111)
            ax.plot(df[x_col].values, df[y_col].values,
                    marker='o', markersize=3, linewidth=1.5, color='steelblue')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} vs {x_col}", fontweight='bold')
            ax.grid(True, alpha=0.3)
            self.couple_canvas.fig.tight_layout()
            self.couple_canvas.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "플롯 오류", str(e))
