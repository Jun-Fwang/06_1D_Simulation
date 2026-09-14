"""
gui/tab_data_process.py
데이터 처리 탭 (탭 1).
.bin 파일 경로 설정 → 데이터 불러오기/필터링/다항식 피팅 → 결과 플롯.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QDoubleSpinBox, QFileDialog, QProgressBar,
    QTabWidget, QSplitter, QTextEdit,
)
from PyQt6.QtCore import Qt
import os


class DataProcessTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._result = None
        self._init_ui()

    def _init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)

        # ─ 경로 설정 ─
        path_box = QGroupBox("데이터 경로 설정")
        pb = QVBoxLayout(path_box)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("데이터 루트 폴더:"))
        self.edit_root = QLineEdit()
        self.edit_root.setPlaceholderText("예) D:/data/experiment")
        btn_root = QPushButton("찾기")
        btn_root.setFixedWidth(60)
        btn_root.clicked.connect(lambda: self._browse_dir(self.edit_root))
        r1.addWidget(self.edit_root)
        r1.addWidget(btn_root)
        pb.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("저장 폴더:"))
        self.edit_save = QLineEdit()
        self.edit_save.setPlaceholderText("예) D:/data/processed")
        btn_save = QPushButton("찾기")
        btn_save.setFixedWidth(60)
        btn_save.clicked.connect(lambda: self._browse_dir(self.edit_save))
        r2.addWidget(self.edit_save)
        r2.addWidget(btn_save)
        pb.addLayout(r2)
        main.addWidget(path_box)

        # ─ 필터 파라미터 ─
        filt_box = QGroupBox("버터워스 로우패스 필터")
        fb = QHBoxLayout(filt_box)
        fb.addWidget(QLabel("차단 주파수 (Hz):"))
        self.spin_cutoff = QDoubleSpinBox()
        self.spin_cutoff.setRange(1, 10000)
        self.spin_cutoff.setValue(100)
        fb.addWidget(self.spin_cutoff)
        fb.addSpacing(20)
        fb.addWidget(QLabel("필터 차수:"))
        self.spin_order = QDoubleSpinBox()
        self.spin_order.setRange(1, 10)
        self.spin_order.setValue(4)
        self.spin_order.setDecimals(0)
        fb.addWidget(self.spin_order)
        fb.addStretch()
        main.addWidget(filt_box)

        # ─ 실행 버튼 ─
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ 데이터 처리 실행")
        self.btn_run.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold;")
        self.btn_run.setFixedHeight(32)
        self.btn_run.clicked.connect(self._run)
        btn_row.addWidget(self.btn_run)
        btn_row.addStretch()
        main.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)   # 진행 불명 (busy)
        self.progress_bar.setVisible(False)
        main.addWidget(self.progress_bar)

        # ─ 결과 영역 ─
        from gui.widgets.plot_canvas import PlotCanvas
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumWidth(300)
        self.log_text.setPlaceholderText("처리 결과 로그...")
        self.result_canvas = PlotCanvas(figsize=(8, 5))
        splitter.addWidget(self.log_text)
        splitter.addWidget(self.result_canvas)
        splitter.setSizes([280, 600])
        main.addWidget(splitter, stretch=1)

    def _browse_dir(self, edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if path:
            edit.setText(path)

    def _run(self):
        from core.workers import DataProcessWorker
        root = self.edit_root.text().strip()
        save = self.edit_save.text().strip()
        if not root:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "경고", "데이터 루트 폴더를 선택하세요.")
            return
        params = {
            'root_path': root,
            'save_path': save or root,
            'cutoff_freq': self.spin_cutoff.value(),
            'filter_order': int(self.spin_order.value()),
        }
        self._worker = DataProcessWorker(params)
        self._worker.progress.connect(lambda _: None)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._worker.start()

    def _on_done(self, result):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._result = result
        if isinstance(result, dict):
            summary = result.get('summary', '')
            self.log_text.setPlainText(str(summary))
            fig = result.get('fig')
            if fig is not None:
                self._show_fig(fig)

    def _on_error(self, msg):
        from PyQt6.QtWidgets import QMessageBox
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "처리 오류", msg)

    def _show_fig(self, fig):
        import matplotlib
        self.result_canvas.fig.clear()
        # fig 객체의 axes를 result_canvas.fig에 복제
        for ax in fig.get_axes():
            self.result_canvas.fig.add_axes(ax)
        self.result_canvas.canvas.draw()
