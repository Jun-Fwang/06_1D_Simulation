"""
gui/tab_simulation.py
시뮬레이션 탭.

레이아웃: 수평 QSplitter
  왼쪽  – 모드 선택(세그먼트 버튼) + 파라미터 + 공통 설정 + 실행/중지
  오른쪽 – 결과 그래프(2×2) + 결과 저장 버튼

기본 모드: Coupling
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QDoubleSpinBox, QComboBox, QProgressBar, QMessageBox,
    QStackedWidget, QSplitter, QScrollArea, QFrame, QDialog,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
import numpy as np

_BTN_MODE_ON = """
QPushButton {
    background: #1565C0; color: white;
    font-weight: bold; font-size: 12px;
    border: none; border-radius: 4px; padding: 5px 14px;
}
"""
_BTN_MODE_OFF = """
QPushButton {
    background: #E0E0E0; color: #555;
    font-size: 12px;
    border: none; border-radius: 4px; padding: 5px 14px;
}
QPushButton:hover { background: #BDBDBD; }
"""


class _RigidWallPreviewWidget(QWidget):
    """Rigid Wall 모드의 차량-연결기-고정벽 구성 미리보기."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        cy = h // 2 - 8

        # 차량
        car_x, car_w, car_h = 24, 90, 42
        car_y = cy - car_h // 2
        p.setPen(QPen(QColor("#1E3A8A"), 1))
        p.setBrush(QBrush(QColor("#1D4ED8")))
        p.drawRoundedRect(car_x, car_y, car_w, car_h, 6, 6)

        # 차량 바퀴
        p.setBrush(QBrush(QColor("#1F2937")))
        p.setPen(QPen(QColor("#111827"), 1))
        p.drawEllipse(car_x + 14, car_y + car_h - 6, 14, 14)
        p.drawEllipse(car_x + car_w - 28, car_y + car_h - 6, 14, 14)

        # 연결기
        c_start = car_x + car_w + 6
        c_end = w - 90
        p.setPen(QPen(QColor("#555"), 3))
        p.drawLine(c_start, cy, c_end, cy)
        p.setBrush(QBrush(QColor("#9CA3AF")))
        p.setPen(QPen(QColor("#374151"), 1))
        p.drawEllipse((c_start + c_end) // 2 - 7, cy - 7, 14, 14)

        # 고정벽
        wall_x, wall_w, wall_h = w - 80, 32, 74
        wall_y = cy - wall_h // 2
        p.setPen(QPen(QColor("#1E40AF"), 1))
        p.setBrush(QBrush(QColor("#2563EB")))
        p.drawRect(wall_x, wall_y, wall_w, wall_h)

        # 라벨
        p.setPen(QColor("#333"))
        p.drawText(car_x + 16, wall_y + wall_h + 20, "차량")
        p.drawText((c_start + c_end) // 2 - 20, wall_y + wall_h + 20, "연결기")
        p.drawText(wall_x - 8, wall_y + wall_h + 20, "고정벽")


class SimulationTab(QWidget):
    def __init__(self, model_registry: dict, parent=None):
        super().__init__(parent)
        self.model_registry = model_registry
        self._worker = None
        self._result = None
        self._progress_dialog = None
        self._progress_dialog_label = None
        self._progress_dialog_bar = None
        self._mode = 1          # 0=Rigid Wall, 1=Coupling (기본값)
        self._init_ui()

    # ── UI 구성 ───────────────────────────────────────────────────────
    def _init_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ── 왼쪽: 제어 패널 (모드 선택 + 공통 설정 + 실행) ──────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(290)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 10, 10)
        lv.setSpacing(8)

        # 모드 선택 세그먼트 버튼
        mode_box = QGroupBox("시뮬레이션 모드")
        mode_bl = QHBoxLayout(mode_box)
        mode_bl.setSpacing(6)
        self.btn_rw = QPushButton("Rigid Wall")
        self.btn_cp = QPushButton("Coupling")
        self.btn_rw.setFixedHeight(30)
        self.btn_cp.setFixedHeight(30)
        self.btn_rw.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_cp.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rw.clicked.connect(lambda: self._set_mode(0))
        self.btn_cp.clicked.connect(lambda: self._set_mode(1))
        mode_bl.addWidget(self.btn_rw)
        mode_bl.addWidget(self.btn_cp)
        lv.addWidget(mode_box)

        # 공통 시뮬레이션 파라미터
        common_box = QGroupBox("시뮬레이션 설정")
        cv = QVBoxLayout(common_box)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("종료 시간 (ms):"))
        self.spin_time = QDoubleSpinBox()
        self.spin_time.setRange(1, 100000)
        self.spin_time.setValue(2000)
        self.spin_time.setDecimals(0)
        r1.addWidget(self.spin_time)
        r1.addStretch()
        cv.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("시간 간격 dt (ms):"))
        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setRange(0.001, 10)
        self.spin_dt.setValue(0.1)
        self.spin_dt.setDecimals(3)
        r2.addWidget(self.spin_dt)
        r2.addStretch()
        cv.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("에너지 증가 한계 (%):"))
        self.spin_elimit = QDoubleSpinBox()
        self.spin_elimit.setRange(0.1, 100.0)
        self.spin_elimit.setValue(10.0)
        self.spin_elimit.setDecimals(1)
        self.spin_elimit.setSingleStep(0.1)
        self.spin_elimit.setToolTip("허용 에너지 증가 한계(%)")
        r3.addWidget(self.spin_elimit)
        r3.addStretch()
        cv.addLayout(r3)
        lv.addWidget(common_box)

        # 실행 / 중지 버튼
        self.btn_run = QPushButton("▶  시뮬레이션 실행")
        self.btn_run.setStyleSheet(
            "background:#2E7D32; color:white; font-weight:bold;"
            "font-size:14px; border-radius:5px;")
        self.btn_run.setFixedHeight(42)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.clicked.connect(self._run_simulation)

        self.btn_stop = QPushButton("■  중지")
        self.btn_stop.setStyleSheet(
            "background:#C62828; color:white; font-weight:bold;"
            "font-size:13px; border-radius:5px;")
        self.btn_stop.setFixedHeight(42)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop.clicked.connect(self._stop_simulation)

        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        run_row.addWidget(self.btn_run, stretch=3)
        run_row.addWidget(self.btn_stop, stretch=1)
        lv.addLayout(run_row)

        # 진행바 + 상태 레이블
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(18)
        lv.addWidget(self.progress_bar)

        self.lbl_status = QLabel("대기 중")
        self.lbl_status.setStyleSheet("color: #555; font-size: 11px;")
        lv.addWidget(self.lbl_status)

        lv.addStretch()
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        # ── 오른쪽: 모드별 파라미터 패널 (풀 너비) ──────────────────
        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(self._build_rigidwall_panel())  # 0: Rigid Wall
        self.right_stack.addWidget(self._build_coupling_panel())   # 1: Coupling
        splitter.addWidget(self.right_stack)

        splitter.setSizes([290, 1110])
        root.addWidget(splitter)

        # 기본 모드: Coupling
        self._set_mode(1)

    # ── 모드 전환 ─────────────────────────────────────────────────────
    def _set_mode(self, idx: int):
        self._mode = idx
        self.right_stack.setCurrentIndex(idx)
        self.btn_rw.setStyleSheet(_BTN_MODE_ON if idx == 0 else _BTN_MODE_OFF)
        self.btn_cp.setStyleSheet(_BTN_MODE_ON if idx == 1 else _BTN_MODE_OFF)

    # ── Rigid Wall 파라미터 패널 ──────────────────────────────────────
    def _build_rigidwall_panel(self):
        w = QGroupBox("Rigid Wall 파라미터")
        vl = QVBoxLayout(w)

        preview_box = QGroupBox("구성 미리보기")
        pv = QVBoxLayout(preview_box)
        pv.addWidget(_RigidWallPreviewWidget())
        vl.addWidget(preview_box)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("차량 질량 (kg):"))
        self.spin_rw_mass = QDoubleSpinBox()
        self.spin_rw_mass.setRange(1000, 500000)
        self.spin_rw_mass.setValue(33000)
        self.spin_rw_mass.setSingleStep(1000)
        self.spin_rw_mass.setDecimals(0)
        r1.addWidget(self.spin_rw_mass)
        r1.addStretch()
        vl.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("충돌 속도 (km/h):"))
        self.spin_rw_vel = QDoubleSpinBox()
        self.spin_rw_vel.setRange(0.1, 200)
        self.spin_rw_vel.setValue(5.0)
        r2.addWidget(self.spin_rw_vel)
        r2.addStretch()
        vl.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("완충기 모델:"))
        self.combo_rw_model = QComboBox()
        r3.addWidget(self.combo_rw_model)
        r3.addStretch()
        vl.addLayout(r3)
        return w

    # ── Coupling 파라미터 패널 ────────────────────────────────────────
    def _build_coupling_panel(self):
        from gui.widgets.train_config_widget import TrainConfigWidget
        w = QGroupBox("Coupling – 열차 편성")
        vl = QVBoxLayout(w)
        self.train_config = TrainConfigWidget(self.model_registry)
        vl.addWidget(self.train_config)
        return w

    # ── 레지스트리 갱신 ───────────────────────────────────────────────
    def refresh_registry(self, registry: dict):
        self.model_registry = registry
        cur = self.combo_rw_model.currentText()
        self.combo_rw_model.blockSignals(True)
        self.combo_rw_model.clear()
        self.combo_rw_model.addItems(list(registry.keys()))
        if cur in registry:
            self.combo_rw_model.setCurrentText(cur)
        self.combo_rw_model.blockSignals(False)
        self.train_config.refresh_registry(registry)

    # ── 시뮬레이션 실행 ──────────────────────────────────────────────
    def _run_simulation(self):
        from core.workers import RigidWallWorker, CouplingWorker
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "실행 중", "시뮬레이션이 이미 실행 중입니다. 먼저 중지해 주세요.")
            return

        t_end = self.spin_time.value()
        dt = self.spin_dt.value()
        energy_limit = self.spin_elimit.value() / 100.0

        if self._mode == 0:     # Rigid Wall
            model_name = self.combo_rw_model.currentText()
            if not model_name or model_name not in self.model_registry:
                QMessageBox.warning(self, "경고", "완충기 모델을 선택하세요.")
                return
            buffer_model = self.model_registry[model_name]
            mass = self.spin_rw_mass.value()
            vel_mms = self.spin_rw_vel.value() / 3.6   # km/h → mm/ms
            self._worker = RigidWallWorker(mass, buffer_model, vel_mms,
                                           dt, t_end, energy_limit)

        else:                   # Coupling
            cfg = self.train_config.get_train_config()
            coupler_models, missing = [], []
            for n in cfg['coupler_name_list']:
                if n in self.model_registry:
                    coupler_models.append(self.model_registry[n])
                else:
                    missing.append(n)
            if missing:
                QMessageBox.warning(self, "모델 누락",
                                    f"다음 연결기 모델을 먼저 등록하세요:\n{missing}")
                return
            self._worker = CouplingWorker(
                car_mass_list=cfg['car_mass_list'],
                coupler_model_list=coupler_models,
                mu_kinetic_list=cfg['mu_kinetic_list'],
                mu_static_list=cfg['mu_static_list'],
                car_velocity_list=cfg['car_velocity_list'],
                termination_time=t_end,
                dt=dt,
                energy_increase_limit=energy_limit,
                car_spring_enabled=cfg.get('car_spring_enabled', False),
                car_stiffness_list=cfg.get('car_stiffness_list', []),
            )

        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_simulation_done)
        self._worker.error.connect(self._on_simulation_error)
        self._worker.cancelled.connect(self._on_simulation_cancelled)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._on_progress(0)
        self._set_status("시뮬레이션 실행 중…")
        self._show_progress_popup()
        self._worker.start()

    def _stop_simulation(self):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self.btn_stop.setEnabled(False)
            self._set_status("중지 요청 중…")

    def _on_simulation_done(self, result):
        if self.sender() is not self._worker:
            return
        self._result = result
        self._worker = None
        self._close_progress_popup()
        self._reset_buttons()
        self._on_progress(100)
        self._set_status("시뮬레이션 완료 ✔")
        QMessageBox.information(self, "시뮬레이션 완료", "시뮬레이션이 완료되었습니다.")

    def _on_simulation_error(self, msg):
        if self.sender() is not self._worker:
            return
        self._worker = None
        self._close_progress_popup()
        QMessageBox.critical(self, "시뮬레이션 오류", msg)
        self._set_status("오류 발생")
        self._reset_buttons()

    def _on_simulation_cancelled(self):
        if self.sender() is not self._worker:
            return
        self._worker = None
        self._close_progress_popup()
        self._set_status("중지됨")
        self._reset_buttons()

    def _on_progress(self, value: int):
        pct = int(max(0, min(100, value)))
        self.progress_bar.setValue(pct)
        if self._progress_dialog_bar is not None:
            self._progress_dialog_bar.setValue(pct)

    def _set_status(self, text: str):
        self.lbl_status.setText(text)
        if self._progress_dialog_label is not None:
            self._progress_dialog_label.setText(text)

    def _ensure_progress_popup(self):
        if self._progress_dialog is not None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("시뮬레이션 상태")
        dlg.setModal(True)
        dlg.setFixedWidth(360)

        vl = QVBoxLayout(dlg)
        vl.setContentsMargins(14, 12, 14, 12)
        vl.setSpacing(10)

        lbl = QLabel("시뮬레이션 실행 중…")
        bar = QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)

        vl.addWidget(lbl)
        vl.addWidget(bar)

        self._progress_dialog = dlg
        self._progress_dialog_label = lbl
        self._progress_dialog_bar = bar

    def _show_progress_popup(self):
        self._ensure_progress_popup()
        self._progress_dialog_bar.setValue(0)
        self._progress_dialog_label.setText("시뮬레이션 실행 중…")
        self._progress_dialog.show()
        self._progress_dialog.raise_()
        self._progress_dialog.activateWindow()

    def _close_progress_popup(self):
        if self._progress_dialog is not None:
            self._progress_dialog.hide()

    def _reset_buttons(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def get_last_result(self):
        return self._result
