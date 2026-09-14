"""
gui/tab_sampling.py
데이터 샘플링 탭 (탭 2).
처리된 데이터 CSV 로드 → KNN 클러스터링 샘플링 → 다항식 피팅 → 저장.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox,
    QFileDialog, QProgressBar, QSplitter, QTextEdit,
)
from PyQt6.QtCore import Qt


class SamplingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._init_ui()

    def _init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)

        # ─ 파일 입력 ─
        file_box = QGroupBox("데이터 파일")
        fb = QVBoxLayout(file_box)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("입력 CSV/Excel:"))
        self.edit_input = QLineEdit()
        self.edit_input.setReadOnly(True)
        btn_in = QPushButton("찾기")
        btn_in.setFixedWidth(60)
        btn_in.clicked.connect(lambda: self._browse_file(
            self.edit_input, "CSV / Excel (*.csv *.xlsx *.xls)"))
        r1.addWidget(self.edit_input)
        r1.addWidget(btn_in)
        fb.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("저장 경로:"))
        self.edit_output = QLineEdit()
        btn_out = QPushButton("찾기")
        btn_out.setFixedWidth(60)
        btn_out.clicked.connect(lambda: self._browse_save(self.edit_output))
        r2.addWidget(self.edit_output)
        r2.addWidget(btn_out)
        fb.addLayout(r2)
        main.addWidget(file_box)

        # ─ 샘플링 파라미터 ─
        param_box = QGroupBox("샘플링 / 피팅 파라미터")
        pg = QHBoxLayout(param_box)
        pg.addWidget(QLabel("클러스터 수 (k):"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(2, 1000)
        self.spin_k.setValue(50)
        pg.addWidget(self.spin_k)
        pg.addSpacing(20)
        pg.addWidget(QLabel("다항식 차수:"))
        self.spin_deg = QSpinBox()
        self.spin_deg.setRange(1, 10)
        self.spin_deg.setValue(3)
        pg.addWidget(self.spin_deg)
        pg.addSpacing(20)
        pg.addWidget(QLabel("Random seed:"))
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999)
        self.spin_seed.setValue(42)
        pg.addWidget(self.spin_seed)
        pg.addStretch()
        main.addWidget(param_box)

        # ─ 실행 ─
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ 샘플링 실행")
        self.btn_run.setStyleSheet(
            "background:#1976D2; color:white; font-weight:bold;")
        self.btn_run.setFixedHeight(32)
        self.btn_run.clicked.connect(self._run)
        btn_row.addWidget(self.btn_run)
        btn_row.addStretch()
        main.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        main.addWidget(self.progress_bar)

        # ─ 결과 ─
        from gui.widgets.plot_canvas import PlotCanvas
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumWidth(280)
        self.log_text.setPlaceholderText("샘플링 결과 로그...")
        self.result_canvas = PlotCanvas(figsize=(8, 5))
        splitter.addWidget(self.log_text)
        splitter.addWidget(self.result_canvas)
        splitter.setSizes([260, 600])
        main.addWidget(splitter, stretch=1)

    def _browse_file(self, edit, filt):
        path, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", filt)
        if path:
            edit.setText(path)

    def _browse_save(self, edit):
        path, _ = QFileDialog.getSaveFileName(
            self, "저장 파일", "", "CSV (*.csv)")
        if path:
            edit.setText(path)

    def _run(self):
        from PyQt6.QtCore import QThread, pyqtSignal
        from PyQt6.QtWidgets import QMessageBox
        import threading

        input_path = self.edit_input.text().strip()
        if not input_path:
            QMessageBox.warning(self, "경고", "입력 파일을 선택하세요.")
            return

        # 간단한 QThread 인라인 구현
        class _Worker(QThread):
            finished = pyqtSignal(object)
            error = pyqtSignal(str)

            def __init__(self, input_path, output_path, k, deg, seed):
                super().__init__()
                self.input_path = input_path
                self.output_path = output_path
                self.k = k
                self.deg = deg
                self.seed = seed

            def run(self):
                try:
                    from core.data_sampling import (
                        f_load_df, f_sampling_df, f_save_df)
                    df = f_load_df(self.input_path)
                    sampled_df, fig = f_sampling_df(df, self.k, self.deg, self.seed)
                    if self.output_path:
                        f_save_df(sampled_df, self.output_path)
                    self.finished.emit({'df': sampled_df, 'fig': fig})
                except Exception as e:
                    self.error.emit(str(e))

        self._worker = _Worker(
            input_path,
            self.edit_output.text().strip(),
            self.spin_k.value(),
            self.spin_deg.value(),
            self.spin_seed.value(),
        )
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self.btn_run.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._worker.start()

    def _on_done(self, result):
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        df = result.get('df')
        fig = result.get('fig')
        if df is not None:
            self.log_text.setPlainText(
                f"샘플링 완료\n원본: {len(df)}행\n\n{df.describe().to_string()}")
        if fig is not None:
            self._display_fig(fig)

    def _on_error(self, msg):
        from PyQt6.QtWidgets import QMessageBox
        self.btn_run.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "오류", msg)

    def _display_fig(self, fig):
        import matplotlib.pyplot as plt
        self.result_canvas.fig.clear()
        for ax in fig.get_axes():
            self.result_canvas.fig.add_axes(ax)
        self.result_canvas.canvas.draw()
        plt.close(fig)
