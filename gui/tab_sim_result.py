"""
gui/tab_sim_result.py
시뮬레이션 결과 확인 탭 (탭 ③).

tab_simulation에서 시뮬레이션이 완료되면 main_window가
set_result(result, mode)를 호출하여 결과를 표시합니다.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QLabel,
    QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt


class SimResultTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._mode = 1   # 0=Rigid Wall, 1=Coupling
        self._init_ui()

    def _init_ui(self):
        vl = QVBoxLayout(self)
        vl.setContentsMargins(10, 10, 10, 10)
        vl.setSpacing(6)

        # 상태 레이블
        self.lbl_info = QLabel("시뮬레이션을 실행하면 결과가 여기에 표시됩니다.")
        self.lbl_info.setStyleSheet("color: #777; font-size: 12px;")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(self.lbl_info)

        # 결과 플롯
        from gui.widgets.plot_canvas import PlotCanvas
        result_box = QGroupBox("시뮬레이션 결과")
        result_box.setStyleSheet("QGroupBox { font-weight: bold; font-size: 13px; }")
        rb = QVBoxLayout(result_box)
        self.result_canvas = PlotCanvas(nrows=2, ncols=2, figsize=(10, 6))
        rb.addWidget(self.result_canvas)
        vl.addWidget(result_box, stretch=1)

        # 저장 버튼
        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("💾  결과 저장 (CSV)")
        self.btn_save.setFixedHeight(30)
        self.btn_save.setEnabled(False)
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.setStyleSheet(
            "background:#455A64; color:white; font-weight:bold; border-radius:4px;")
        self.btn_save.clicked.connect(self._save_result)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_save)
        vl.addLayout(btn_row)

    # ── 결과 수신 ─────────────────────────────────────────────────────
    def set_result(self, result, mode: int):
        """main_window에서 시뮬레이션 완료 시 호출."""
        self._result = result
        self._mode = mode
        self.btn_save.setEnabled(True)
        self.lbl_info.setText("시뮬레이션 결과")
        self._plot_result(result)

    # ── 결과 플롯 ─────────────────────────────────────────────────────
    def _plot_result(self, result):
        import pandas as pd
        if result is None or (isinstance(result, pd.DataFrame) and result.empty):
            return

        self.result_canvas.clear()
        axes = self.result_canvas.axes   # shape (2, 2)
        time = result['Time [ms]'].values

        if self._mode == 0:     # Rigid Wall
            axes[0, 0].plot(time, result['Buffer_Displacement [mm]'].values,
                            color='steelblue')
            axes[0, 0].set_title('Buffer Displacement (mm)')
            axes[0, 0].set_xlabel('Time [ms]')
            axes[0, 0].set_ylabel('Displacement [mm]')
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(time, result['Buffer_Velocity [mm/ms]'].values,
                            color='darkorange')
            axes[0, 1].set_title('Buffer Velocity (mm/ms)')
            axes[0, 1].set_xlabel('Time [ms]')
            axes[0, 1].set_ylabel('Velocity [mm/ms]')
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(time, result['Buffer_Force [kN]'].values,
                            color='tomato')
            axes[1, 0].set_title('Impact Force (kN)')
            axes[1, 0].set_xlabel('Time [ms]')
            axes[1, 0].set_ylabel('Force [kN]')
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(result['Buffer_Displacement [mm]'].values,
                            result['Buffer_Force [kN]'].values,
                            color='mediumvioletred')
            axes[1, 1].set_title('F-D Curve')
            axes[1, 1].set_xlabel('Displacement [mm]')
            axes[1, 1].set_ylabel('Force [kN]')
            axes[1, 1].grid(True, alpha=0.3)

        else:                   # Coupling
            disp_cols   = [c for c in result.columns if 'Car_Displacement_Difference' in c]
            vel_cols    = [c for c in result.columns if c.startswith('Car_Velocity_')]
            force_cols  = [c for c in result.columns if c.startswith('Coupler_Force_')]
            energy_cols = [c for c in result.columns if c.startswith('Coupler_Internal_Energy_')]

            for i, col in enumerate(disp_cols):
                axes[0, 0].plot(time, result[col].values, label=f'Car {i+1}')
            axes[0, 0].set_title('Relative Displacement by Car (mm)')
            axes[0, 0].set_xlabel('Time [ms]')
            axes[0, 0].set_ylabel('Displacement [mm]')
            axes[0, 0].grid(True, alpha=0.3)
            if disp_cols: axes[0, 0].legend(fontsize=7)

            for i, col in enumerate(vel_cols):
                axes[0, 1].plot(time, result[col].values, label=f'Car {i+1}')
            axes[0, 1].set_title('Velocity by Car (km/h)')
            axes[0, 1].set_xlabel('Time [ms]')
            axes[0, 1].set_ylabel('Velocity [km/h]')
            axes[0, 1].grid(True, alpha=0.3)
            if vel_cols: axes[0, 1].legend(fontsize=7)

            for i, col in enumerate(force_cols):
                axes[1, 0].plot(time, result[col].values, label=f'Coupler {i+1}')
            axes[1, 0].set_title('Impact Force by Coupler (kN)')
            axes[1, 0].set_xlabel('Time [ms]')
            axes[1, 0].set_ylabel('Force [kN]')
            axes[1, 0].grid(True, alpha=0.3)
            if force_cols: axes[1, 0].legend(fontsize=7)

            for i, col in enumerate(energy_cols):
                axes[1, 1].plot(time, result[col].values, label=f'Coupler {i+1}')
            axes[1, 1].set_title('Absorbed Energy by Coupler (kJ)')
            axes[1, 1].set_xlabel('Time [ms]')
            axes[1, 1].set_ylabel('Energy [kJ]')
            axes[1, 1].grid(True, alpha=0.3)
            if energy_cols: axes[1, 1].legend(fontsize=7)

        self.result_canvas.fig.tight_layout()
        self.result_canvas.canvas.draw()

    # ── 결과 저장 ─────────────────────────────────────────────────────
    def _save_result(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", "simulation_result.csv", "CSV (*.csv)")
        if path:
            try:
                self._result.to_csv(path, index=False, encoding='utf-8-sig')
            except Exception as e:
                QMessageBox.critical(self, "저장 오류", str(e))
