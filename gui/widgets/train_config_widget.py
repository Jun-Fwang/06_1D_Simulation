"""
gui/widgets/train_config_widget.py
열차 편성 시각화 위젯.
- 차량을 QPainter로 직접 그림 (차체 + 캐빈 + 바퀴)
- 차량 추가/제거 시 그림 실시간 반영
- 연결기(coupler) 클릭 → 등록된 완충기 모델 선택 다이얼로그
- Moving(파란색) / Stationary(회색) 색상 구분
- get_train_config() → 시뮬레이션 함수에 전달할 파라미터 dict 반환
"""
from PyQt6.QtWidgets import (
    QWidget, QScrollArea, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QDoubleSpinBox, QComboBox,
    QDialog, QDialogButtonBox, QInputDialog, QSizePolicy,
    QFrame, QSpinBox,
)
from PyQt6.QtCore import Qt, QRect, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath


# ─── 연결기 위젯 ─────────────────────────────────────────────────────────────

class CouplerWidget(QWidget):
    """두 차량 사이의 연결기. 클릭하면 모델 선택 다이얼로그 열림."""
    model_assigned = pyqtSignal(int, str)   # (coupler_index, model_name)

    _W, _H = 52, 80

    def __init__(self, coupler_index: int, model_registry: dict, parent=None):
        super().__init__(parent)
        self.coupler_index = coupler_index
        self.model_registry = model_registry
        self.assigned_model = "미지정"
        self.setFixedSize(self._W, self._H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("클릭하여 완충기 모델 선택")

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cy = 24   # 차량 그림(_CarDrawWidget) 중심 높이에 맞춤

        # 연결 라인
        pen = QPen(QColor("#555"), 2)
        p.setPen(pen)
        p.drawLine(0, cy, w, cy)

        # 중앙 원 (클릭 영역 강조)
        assigned = self.assigned_model != "미지정"
        brush_color = QColor("#2196F3") if assigned else QColor("#bbb")
        p.setBrush(QBrush(brush_color))
        p.setPen(QPen(QColor("#333"), 1))
        r = 7
        p.drawEllipse(w // 2 - r, cy - r, r * 2, r * 2)

        # 모델 이름 (짧게)
        p.setPen(QColor("#333"))
        p.setFont(QFont("Arial", 6))
        name = self.assigned_model[:4] if len(self.assigned_model) > 4 else self.assigned_model
        p.drawText(QRect(0, cy + r + 2, w, 16), Qt.AlignmentFlag.AlignHCenter, name)

    def mousePressEvent(self, _event):
        if not self.model_registry:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "알림", "먼저 탭 4에서 완충기 모델을 등록해 주세요.")
            return
        names = list(self.model_registry.keys())
        dlg = QDialog(self)
        dlg.setWindowTitle(f"연결기 {self.coupler_index} 모델 선택")
        dlg.setMinimumWidth(300)
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel("등록된 완충기 모델:"))
        combo = QComboBox()
        combo.addItems(names)
        if self.assigned_model in names:
            combo.setCurrentText(self.assigned_model)
        vl.addWidget(combo)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vl.addWidget(btns)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.assigned_model = combo.currentText()
            self.update()
            self.model_assigned.emit(self.coupler_index, self.assigned_model)


# ─── 단일 차량 위젯 ──────────────────────────────────────────────────────────

class TrainCarWidget(QWidget):
    """차량 한 량. QPainter로 차체 그림 + 하단에 파라미터 입력 폼."""

    _CAR_W, _CAR_H = 80, 48   # 차체 그림 영역 크기

    def __init__(self, car_index: int, parent=None):
        super().__init__(parent)
        self.car_index = car_index
        self._is_moving = True
        self._spring_visible = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(4)
        self.setFixedWidth(160)

        # 차량 그림 영역
        self.car_draw = _CarDrawWidget(self)
        self.car_draw.setFixedSize(self._CAR_W, self._CAR_H)
        layout.addWidget(self.car_draw, alignment=Qt.AlignmentFlag.AlignHCenter)

        # 번호 레이블
        lbl = QLabel(f"Car {self.car_index + 1}")
        lbl.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lbl.setStyleSheet("font-weight: bold; font-size: 10px;")
        layout.addWidget(lbl)

        # 질량 (라벨-입력 좌우 배치)
        self.spin_mass = QDoubleSpinBox()
        self.spin_mass.setRange(1000, 500000)
        self.spin_mass.setValue(33000)
        self.spin_mass.setSingleStep(1000)
        self.spin_mass.setDecimals(0)
        self.spin_mass.setToolTip("질량 (kg)")
        self.spin_mass.setFixedWidth(98)
        row_mass = QHBoxLayout()
        row_mass.setContentsMargins(0, 0, 0, 0)
        row_mass.setSpacing(6)
        lbl_mass = QLabel("질량 (kg)")
        lbl_mass.setStyleSheet("font-size: 9px; color: #444;")
        lbl_mass.setFixedWidth(54)
        row_mass.addWidget(lbl_mass)
        row_mass.addWidget(self.spin_mass)
        row_mass.addStretch()
        layout.addLayout(row_mass)

        # 차량 강성 입력 행 (기본은 숨김)
        self.spin_stiffness = QDoubleSpinBox()
        self.spin_stiffness.setRange(0.0, 10000.0)
        self.spin_stiffness.setValue(10.0)
        self.spin_stiffness.setSingleStep(1.0)
        self.spin_stiffness.setDecimals(1)
        self.spin_stiffness.setToolTip("차량강성 (kN/mm)")
        self.spin_stiffness.setFixedWidth(98)

        self.row_stiffness_widget = QWidget()
        self.row_stiffness_widget.setVisible(False)
        self.row_stiffness = QHBoxLayout(self.row_stiffness_widget)
        self.row_stiffness.setContentsMargins(0, 0, 0, 0)
        self.row_stiffness.setSpacing(6)
        lbl_stiff = QLabel("차량강성(kN/mm)")
        lbl_stiff.setStyleSheet("font-size: 9px; color: #444;")
        lbl_stiff.setFixedWidth(54)
        self.row_stiffness.addWidget(lbl_stiff)
        self.row_stiffness.addWidget(self.spin_stiffness)
        self.row_stiffness.addStretch()
        layout.addWidget(self.row_stiffness_widget)

        # 상태 (라벨-입력 좌우 배치)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Moving", "Stationary"])
        self.combo_mode.setToolTip("차량 상태")
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        self.combo_mode.setFixedWidth(98)
        row_mode = QHBoxLayout()
        row_mode.setContentsMargins(0, 0, 0, 0)
        row_mode.setSpacing(6)
        lbl_mode = QLabel("상태")
        lbl_mode.setStyleSheet("font-size: 9px; color: #444;")
        lbl_mode.setFixedWidth(54)
        row_mode.addWidget(lbl_mode)
        row_mode.addWidget(self.combo_mode)
        row_mode.addStretch()
        layout.addLayout(row_mode)

        # 속도 (라벨-입력 좌우 배치)
        self.spin_vel = QDoubleSpinBox()
        self.spin_vel.setRange(0, 200)
        self.spin_vel.setValue(10.0)
        self.spin_vel.setSingleStep(0.5)
        self.spin_vel.setToolTip("초기 속도 (km/h)")
        self.spin_vel.setFixedWidth(98)
        row_vel = QHBoxLayout()
        row_vel.setContentsMargins(0, 0, 0, 0)
        row_vel.setSpacing(6)
        lbl_v = QLabel("v(km/h)")
        lbl_v.setStyleSheet("font-size: 9px; color: #444;")
        lbl_v.setFixedWidth(54)
        row_vel.addWidget(lbl_v)
        row_vel.addWidget(self.spin_vel)
        row_vel.addStretch()
        layout.addLayout(row_vel)

        # 마찰계수 (정지/동 마찰계수 각각 직접 입력)
        self.spin_mu_static = QDoubleSpinBox()
        self.spin_mu_static.setRange(0, 1)
        self.spin_mu_static.setValue(0.0)
        self.spin_mu_static.setSingleStep(0.01)
        self.spin_mu_static.setDecimals(3)
        self.spin_mu_static.setToolTip("정지 마찰계수 (static)")
        self.spin_mu_static.setFixedWidth(98)
        row_mu_static = QHBoxLayout()
        row_mu_static.setContentsMargins(0, 0, 0, 0)
        row_mu_static.setSpacing(6)
        lbl_mu_static = QLabel("μ_s")
        lbl_mu_static.setStyleSheet("font-size: 9px; color: #444;")
        lbl_mu_static.setFixedWidth(54)
        row_mu_static.addWidget(lbl_mu_static)
        row_mu_static.addWidget(self.spin_mu_static)
        row_mu_static.addStretch()
        layout.addLayout(row_mu_static)

        self.spin_mu_kinetic = QDoubleSpinBox()
        self.spin_mu_kinetic.setRange(0, 1)
        self.spin_mu_kinetic.setValue(0.0)
        self.spin_mu_kinetic.setSingleStep(0.01)
        self.spin_mu_kinetic.setDecimals(3)
        self.spin_mu_kinetic.setToolTip("동 마찰계수 (kinetic)")
        self.spin_mu_kinetic.setFixedWidth(98)
        row_mu_kinetic = QHBoxLayout()
        row_mu_kinetic.setContentsMargins(0, 0, 0, 0)
        row_mu_kinetic.setSpacing(6)
        lbl_mu_kinetic = QLabel("μ_k")
        lbl_mu_kinetic.setStyleSheet("font-size: 9px; color: #444;")
        lbl_mu_kinetic.setFixedWidth(54)
        row_mu_kinetic.addWidget(lbl_mu_kinetic)
        row_mu_kinetic.addWidget(self.spin_mu_kinetic)
        row_mu_kinetic.addStretch()
        layout.addLayout(row_mu_kinetic)

        # 신규 차량 기본 상태는 Stationary
        self.combo_mode.setCurrentText("Stationary")

    def _on_mode_changed(self, mode: str):
        self._is_moving = (mode == "Moving")
        self.car_draw.set_moving(self._is_moving)
        if mode == "Stationary":
            self.spin_vel.setValue(0.0)
            self.spin_mu_static.setValue(0.12)
            self.spin_mu_kinetic.setValue(0.1)
        else:
            self.spin_mu_static.setValue(0.0)
            self.spin_mu_kinetic.setValue(0.0)

    def set_spring_visible(self, visible: bool):
        self._spring_visible = visible
        self.row_stiffness_widget.setVisible(visible)

    def get_params(self) -> dict:
        return {
            'mass': self.spin_mass.value(),
            'velocity_kmh': self.spin_vel.value(),
            'mu_kinetic': self.spin_mu_kinetic.value(),
            'mu_static': self.spin_mu_static.value(),
            'is_moving': self._is_moving,
            'spring_stiffness': self.spin_stiffness.value(),
        }


class _CarDrawWidget(QWidget):
    """차량 형상을 QPainter로 그리는 내부 위젯."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_moving = True

    def set_moving(self, val: bool):
        self._is_moving = val
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # 차체 색
        body_color = QColor("#1976D2") if self._is_moving else QColor("#78909C")
        window_color = QColor("#B3E5FC") if self._is_moving else QColor("#CFD8DC")

        wheel_r = 7
        body_y = 6
        body_h = h - wheel_r * 2 - 6

        # 차체 사각형
        p.setBrush(QBrush(body_color))
        p.setPen(QPen(QColor("#222"), 1))
        p.drawRoundedRect(4, body_y, w - 8, body_h, 5, 5)

        # 창문 (2개)
        p.setBrush(QBrush(window_color))
        p.setPen(QPen(QColor("#aaa"), 1))
        win_w = (w - 8) // 3
        win_h = body_h // 2 - 4
        p.drawRoundedRect(8, body_y + 4, win_w, win_h, 3, 3)
        p.drawRoundedRect(w - 8 - win_w, body_y + 4, win_w, win_h, 3, 3)

        # 캐빈 라인 (상단)
        p.setPen(QPen(QColor("#fff"), 2))
        p.drawLine(4, body_y + body_h // 2 + 2, w - 4, body_y + body_h // 2 + 2)

        # 바퀴 (4개)
        wheel_y = h - wheel_r * 2
        p.setBrush(QBrush(QColor("#333")))
        p.setPen(QPen(QColor("#111"), 1))
        for wx in (10, w - 10 - wheel_r * 2):
            p.drawEllipse(wx, wheel_y, wheel_r * 2, wheel_r * 2)
            # 바퀴 허브
            p.setBrush(QBrush(QColor("#aaa")))
            p.drawEllipse(wx + 3, wheel_y + 3, wheel_r * 2 - 6, wheel_r * 2 - 6)
            p.setBrush(QBrush(QColor("#333")))


# ─── 메인 열차 편성 위젯 ─────────────────────────────────────────────────────

class TrainConfigWidget(QWidget):
    """
    열차 편성 시각화 위젯.
    차량 추가/제거 → 그림 실시간 반영.
    연결기 클릭 → 완충기 모델 선택.
    """

    def __init__(self, model_registry: dict, parent=None):
        super().__init__(parent)
        self.model_registry = model_registry
        self._car_widgets: list[TrainCarWidget] = []
        self._coupler_widgets: list[CouplerWidget] = []
        self.car_spring_enabled = False
        self._shown_once = False
        self._init_ui()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # 컨트롤 버튼 행
        ctrl = QHBoxLayout()
        btn_add = QPushButton("＋ 차량 추가")
        btn_del = QPushButton("－ 차량 제거")
        self.btn_car_spring = QPushButton("차량 강성 On/Off")
        btn_add.setFixedHeight(28)
        btn_del.setFixedHeight(28)
        self.btn_car_spring.setFixedHeight(28)
        self.btn_car_spring.setStyleSheet(
            "QPushButton { background: white; color: #333; border: 1px solid #BDBDBD; border-radius: 4px; }"
        )
        btn_add.clicked.connect(self.add_car)
        btn_del.clicked.connect(self.remove_car)
        self.btn_car_spring.clicked.connect(self.toggle_spring_mode)
        self.lbl_count = QLabel("차량 수: 0")
        ctrl.addWidget(btn_add)
        ctrl.addWidget(btn_del)
        ctrl.addWidget(self.btn_car_spring)
        ctrl.addSpacing(20)
        ctrl.addWidget(self.lbl_count)
        ctrl.addStretch()
        outer.addLayout(ctrl)

        # 안내 레이블
        hint = QLabel("ℹ  질량/속도/마찰계수는 각 칸에서 직접 입력.  "
                      "v = 속도(km/h),  μ_s = 정지 마찰계수,  μ_k = 동 마찰계수  |  "
                      "연결기(●)를 클릭하면 완충기 모델을 지정할 수 있습니다.")
        hint.setStyleSheet("color: #666; font-size: 11px; padding: 0 2px;")
        outer.addWidget(hint)

        # 스크롤 영역 (수직·수평 모두 필요 시 스크롤)
        self.train_scroll = QScrollArea()
        self.train_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.train_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.train_scroll.setWidgetResizable(True)
        self.train_scroll.setMinimumHeight(180)
        outer.addWidget(self.train_scroll, stretch=1)

        self._rebuild_rows()   # 초기 빈 레이아웃 세팅

        # 10량 기본 세팅: Car1~5 Moving(10km/h), Car6~10 Stationary
        for _ in range(10):
            self.add_car()
        for i, car in enumerate(self._car_widgets):
            if i < 5:
                car.combo_mode.setCurrentText("Moving")
                car.spin_vel.setValue(10.0)
            else:
                car.combo_mode.setCurrentText("Stationary")

    # ── 차량 추가 / 제거 ──────────────────────────────────────────────
    def add_car(self):
        idx = len(self._car_widgets)

        if idx > 0:
            c = CouplerWidget(idx - 1, self.model_registry)
            c.model_assigned.connect(self._on_model_assigned)
            self._coupler_widgets.append(c)

        car = TrainCarWidget(idx)
        self._car_widgets.append(car)
        self._rebuild_rows()

    def remove_car(self):
        if len(self._car_widgets) <= 1:
            return

        car = self._car_widgets.pop()
        car.setParent(None)
        car.deleteLater()

        if self._coupler_widgets:
            c = self._coupler_widgets.pop()
            c.setParent(None)
            c.deleteLater()

        self._rebuild_rows()

    def _update_count(self):
        n = len(self._car_widgets)
        self.lbl_count.setText(f"차량 수: {n}  /  연결기 수: {len(self._coupler_widgets)}")

    def _cars_per_row_from_width(self) -> int:
        """스크롤 영역 가시 너비를 기준으로 한 줄당 차량 개수를 계산한다."""
        viewport = self.train_scroll.viewport()
        viewport_width = viewport.width() if viewport else self.train_scroll.width()
        if viewport_width <= 0:
            return 8

        # 차량 위젯(160px) + 연결기 위젯(52px) 수준의 대략적 폭을 기준으로 계산.
        per_car_group_width = 160 + 52
        cars_per_row = max(1, int(viewport_width // per_car_group_width))
        return min(cars_per_row, max(1, len(self._car_widgets)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 창 크기 변화 후 레이아웃 재배치
        self._rebuild_rows()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._shown_once:
            self._shown_once = True
            # 최초 표시 시점에는 실제 뷰포트 폭이 아직 확정되지 않으므로 지연 재계산
            QTimer.singleShot(0, self._rebuild_rows)

    def _rebuild_rows(self):
        """가시 스크롤 폭에 맞춰 한 줄에 들어가는 차량 수를 적응적으로 재구성한다."""
        cars_per_row = self._cars_per_row_from_width()

        new_inner = QWidget()
        vl = QVBoxLayout(new_inner)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.setSpacing(6)

        cars = self._car_widgets
        couplers = self._coupler_widgets
        n = len(cars)

        for row_start in range(0, max(n, 1), cars_per_row):
            if row_start >= n:
                break
            row_end = min(row_start + cars_per_row, n)

            row_w = QWidget()
            row_hl = QHBoxLayout(row_w)
            row_hl.setContentsMargins(0, 0, 0, 0)
            row_hl.setSpacing(0)

            for i in range(row_start, row_end):
                row_hl.addWidget(cars[i])
                if i < n - 1:
                    row_hl.addWidget(couplers[i], alignment=Qt.AlignmentFlag.AlignTop)
            row_hl.addStretch()
            vl.addWidget(row_w)

            # 행 범위 레이블
            row_lbl = QLabel(f"  ▶ Car {row_start + 1}  ~  Car {row_end}")
            row_lbl.setStyleSheet("font-size: 10px; color: #888;")
            vl.addWidget(row_lbl)

            if row_end < n:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("QFrame { background-color: #BDBDBD; }")
                sep.setFixedHeight(2)
                vl.addWidget(sep)

        vl.addStretch()
        self.train_scroll.setWidget(new_inner)   # 이전 inner는 Qt가 자동 삭제
        self._update_count()

    def toggle_spring_mode(self):
        self.car_spring_enabled = not self.car_spring_enabled
        for car in self._car_widgets:
            car.set_spring_visible(self.car_spring_enabled)
        if self.car_spring_enabled:
            self.btn_car_spring.setStyleSheet(
                "QPushButton { background: #1565C0; color: white; font-weight: bold; border: 1px solid #1565C0; border-radius: 4px; }"
            )
        else:
            self.btn_car_spring.setStyleSheet(
                "QPushButton { background: white; color: #333; border: 1px solid #BDBDBD; border-radius: 4px; }"
            )

    def _on_model_assigned(self, coupler_idx: int, model_name: str):
        pass  # 필요 시 상위 위젯에 시그널 전파

    # ── 레지스트리 갱신 ───────────────────────────────────────────────
    def refresh_registry(self, registry: dict):
        """탭 4에서 모델 등록 시 호출하여 레지스트리 공유 갱신."""
        self.model_registry = registry
        for c in self._coupler_widgets:
            c.model_registry = registry
        # 등록된 모델이 있고 연결기가 미지정이면 첫 번째 모델 자동 할당
        if registry:
            first_model = next(iter(registry))
            for c in self._coupler_widgets:
                if c.assigned_model == "미지정":
                    c.assigned_model = first_model
                    c.update()

    # ── 시뮬레이션 파라미터 반환 ──────────────────────────────────────
    def get_train_config(self) -> dict:
        """
        Returns:
            {
                'car_mass_list': [...],
                'car_velocity_list': [...],  # mm/ms
                'mu_kinetic_list': [...],
                'mu_static_list': [...],
                'coupler_name_list': [...],  # str, 길이 = 차량수-1
                'car_spring_enabled': bool,
                'car_stiffness_list': [...],  # kN/mm
            }
        """
        car_mass_list, car_velocity_list = [], []
        mu_kinetic_list, mu_static_list = [], []
        car_stiffness_list = []

        for car in self._car_widgets:
            params = car.get_params()
            car_mass_list.append(params['mass'])
            # km/h → mm/ms  (1 km/h = 1000/3600 m/s = 1/3.6 m/s = 1000/3600 mm/ms)
            car_velocity_list.append(params['velocity_kmh'] / 3.6)
            mu_kinetic_list.append(params['mu_kinetic'])
            mu_static_list.append(params['mu_static'])
            car_stiffness_list.append(params['spring_stiffness'])

        coupler_name_list = [c.assigned_model for c in self._coupler_widgets]

        return {
            'car_mass_list': car_mass_list,
            'car_velocity_list': car_velocity_list,
            'mu_kinetic_list': mu_kinetic_list,
            'mu_static_list': mu_static_list,
            'coupler_name_list': coupler_name_list,
            'car_spring_enabled': self.car_spring_enabled,
            'car_stiffness_list': car_stiffness_list,
        }
