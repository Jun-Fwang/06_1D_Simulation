"""
gui/main_window.py
메인 윈도우.
두 모드(시뮬레이션 및 결과분석 / 완충기 모델 학습)로 6개 탭을 분리.
model_registry를 전역 공유.
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QStackedWidget, QPushButton, QStatusBar,
    QLabel,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from gui.tab_data_process import DataProcessTab
from gui.tab_sampling import SamplingTab
from gui.tab_model_training import ModelTrainingTab
from gui.tab_buffer_model import BufferModelTab
from gui.tab_simulation import SimulationTab
from gui.tab_result_analysis import ResultAnalysisTab

_STYLE_ACTIVE = """
QPushButton {
    background: #1565C0;
    color: white;
    font-weight: bold;
    font-size: 13px;
    border: none;
    border-radius: 5px;
    padding: 6px 20px;
}
"""
_STYLE_INACTIVE = """
QPushButton {
    background: #E0E0E0;
    color: #555555;
    font-size: 13px;
    border: none;
    border-radius: 5px;
    padding: 6px 20px;
}
QPushButton:hover { background: #BDBDBD; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("1D 충돌 시뮬레이션 프로세서")
        self.resize(1400, 860)

        # 앱 전역 완충기 모델 레지스트리
        self.model_registry: dict = {}
        self.ai_modeling_enabled = False

        self._build_menu()
        self._build_ui()
        self._build_statusbar()
        self._set_ai_modeling_state(False)

    def _build_menu(self):
        """View 메뉴에 AI 모델링 체크 액션을 추가한다."""
        self.view_menu = self.menuBar().addMenu("View")
        self.ai_modeling_action = QAction("AI 모델링", self)
        self.ai_modeling_action.setCheckable(True)
        self.ai_modeling_action.setChecked(False)
        self.ai_modeling_action.toggled.connect(self._set_ai_modeling_state)
        self.view_menu.addAction(self.ai_modeling_action)

    def _set_ai_modeling_state(self, enabled: bool):
        """AI 모델링 토글 상태에 따라 관련 UI를 보이거나 숨긴다."""
        self.ai_modeling_enabled = enabled
        self.ai_modeling_action.setChecked(enabled)

        # 학습 모드 버튼 및 학습 모드 스택 숨김/표시
        if hasattr(self, 'btn_mode_train'):
            self.btn_mode_train.setVisible(enabled)

        # 학습 모드의 내부 탭 위젯 전체 비활성화/활성화
        if hasattr(self, 'train_tabs'):
            self.train_tabs.setVisible(enabled)

        # 완충기 모델 생성 탭의 AI 빔 모델 생성 소스탭 숨김/표시
        if hasattr(self, 'tab_buffer') and hasattr(self.tab_buffer, 'src_tabs'):
            self.tab_buffer.src_tabs.setTabVisible(1, enabled)

        # AI 기능이 끄기면 현재 학습 모드 스택이 보이는 경우 시뮬레이션 모드로 복귀
        if not enabled and hasattr(self, 'mode_stack'):
            if self.mode_stack.currentIndex() == 1:
                self.mode_stack.setCurrentIndex(0)
                self._switch_mode(0)

        # 결과 분석 탭의 AI 관련 서브탭(수치 검증, 연결기 데이터 분석) 숨김/표시
        if hasattr(self, 'tab_result') and hasattr(self.tab_result, 'set_ai_tabs_visible'):
            self.tab_result.set_ai_tabs_visible(enabled)

        # 상태 표시
        if hasattr(self, 'lbl_status'):
            self.lbl_status.setText("AI 모델링: " + ("활성" if enabled else "비활성"))

    # ── 전체 UI 구성 ──────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        vl = QVBoxLayout(central)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        # ── 상단 모드 선택 툴바 ──────────────────────────────────────
        toolbar = QWidget()
        toolbar.setFixedHeight(54)
        toolbar.setStyleSheet(
            "background: #F5F5F5; border-bottom: 2px solid #BDBDBD;")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(16, 8, 16, 8)
        tb_layout.setSpacing(10)

        lbl = QLabel("모드 선택")
        lbl.setStyleSheet("font-weight: bold; color: #333; font-size: 12px;")
        tb_layout.addWidget(lbl)

        self.btn_mode_sim = QPushButton("🚄  시뮬레이션 및 결과분석")
        self.btn_mode_train = QPushButton("🎓  완충기 모델 학습")
        self.btn_mode_sim.setFixedHeight(36)
        self.btn_mode_train.setFixedHeight(36)
        self.btn_mode_sim.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_train.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mode_sim.clicked.connect(lambda: self._switch_mode(0))
        self.btn_mode_train.clicked.connect(lambda: self._switch_mode(1))
        tb_layout.addWidget(self.btn_mode_sim)
        tb_layout.addWidget(self.btn_mode_train)
        tb_layout.addStretch()
        vl.addWidget(toolbar)

        # ── 모드별 탭 스택 ───────────────────────────────────────────
        self.mode_stack = QStackedWidget()

        # 모드 0: 시뮬레이션 및 결과분석
        sim_tabs = QTabWidget()
        sim_tabs.setDocumentMode(True)
        self.tab_buffer    = BufferModelTab(self.model_registry)
        self.tab_sim       = SimulationTab(self.model_registry)
        self.tab_result    = ResultAnalysisTab()
        sim_tabs.addTab(self.tab_buffer,  "① 완충기 모델 생성")
        sim_tabs.addTab(self.tab_sim,     "② 시뮬레이션")
        sim_tabs.addTab(self.tab_result,  "③ 결과 분석")
        self.sim_tabs = sim_tabs

        # 모드 1: 완충기 모델 학습
        train_tabs = QTabWidget()
        train_tabs.setDocumentMode(True)
        self.tab_data   = DataProcessTab()
        self.tab_sample = SamplingTab()
        self.tab_train  = ModelTrainingTab()
        train_tabs.addTab(self.tab_data,   "① 데이터 필터링")
        train_tabs.addTab(self.tab_sample, "② 데이터 샘플링")
        train_tabs.addTab(self.tab_train,  "③ DNN 훈련")
        self.train_tabs = train_tabs

        self.mode_stack.addWidget(sim_tabs)
        self.mode_stack.addWidget(train_tabs)
        vl.addWidget(self.mode_stack, stretch=1)

        self.setCentralWidget(central)
        self._connect_sim_to_result()

        # 기본 모드: 시뮬레이션 및 결과분석
        self._switch_mode(0)

    def _switch_mode(self, idx: int):
        self.mode_stack.setCurrentIndex(idx)
        self.btn_mode_sim.setStyleSheet(
            _STYLE_ACTIVE if idx == 0 else _STYLE_INACTIVE)
        self.btn_mode_train.setStyleSheet(
            _STYLE_ACTIVE if idx == 1 else _STYLE_INACTIVE)

    def _connect_sim_to_result(self):
        """tab_simulation 워커 완료 시 결과를 tab_result로 전달."""
        orig = self.tab_sim._on_simulation_done

        def _wrapped(result):
            orig(result)
            self.tab_result.set_sim_result(result, self.tab_sim._mode)

        self.tab_sim._on_simulation_done = _wrapped

    def _build_statusbar(self):
        bar = QStatusBar()
        self.lbl_status = QLabel("준비")
        bar.addPermanentWidget(self.lbl_status)
        self.setStatusBar(bar)

    # ── 레지스트리 갱신 알림 (tab_buffer → tab_sim) ───────────────────
    def on_registry_updated(self):
        self.tab_sim.refresh_registry(self.model_registry)
        n = len(self.model_registry)
        self.lbl_status.setText(f"완충기 모델 등록: {n}개")
