"""
gui/tab_buffer_model.py
완충기 모델 등록 탭 (탭 4).

소스 3가지 선택:
  1) DNN / MPR / RBF 파일 로드 (.h5 / .pth / .pkl)
  2) Piecewise 직접 입력 (BufferTableWidget × 2: loading + unloading)
  3) Combined (2개 직렬)

등록된 모델은 main_window.model_registry dict에 공유.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QComboBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QTabWidget,
)
from PyQt6.QtCore import Qt
import numpy as np


class BufferModelTab(QWidget):
    def __init__(self, model_registry: dict, parent=None):
        super().__init__(parent)
        self.model_registry = model_registry
        self.model_inputs: dict = {}
        self._init_ui()

    def _init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)

        body = QHBoxLayout()
        body.setSpacing(12)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # ─ 중단: 소스 선택 탭 ─
        self.src_tabs = QTabWidget()
        self.src_tabs.addTab(self._build_piecewise_tab(), "piecewise linear 빔 모델 생성(piecewise)")
        self.src_tabs.addTab(self._build_file_tab(), "AI 빔 모델 생성 (DNN/MPR/RBF)")
        self.src_tabs.addTab(self._build_combined_tab(), "빔 직렬 연결(옵션)")
        left_layout.addWidget(self.src_tabs, stretch=1)

        # ─ 등록 버튼 ─
        btn_row = QHBoxLayout()
        btn_row.addWidget(QLabel("모델 이름:"))
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("예) Buffer_TypeA")
        self.edit_name.setFixedWidth(220)
        btn_row.addWidget(self.edit_name)

        btn_register = QPushButton("▶ 모델 등록")
        btn_register.setStyleSheet("background:#1976D2; color:white; font-weight:bold;")
        btn_register.setFixedHeight(32)
        btn_register.clicked.connect(self._register_model)
        btn_row.addStretch()
        btn_row.addWidget(btn_register)
        left_layout.addLayout(btn_row)

        right_panel = QGroupBox("등록된 완충기 모델 목록")
        right_layout = QVBoxLayout(right_panel)

        # ─ 등록 목록 ─
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["모델 이름", "유형", "삭제"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 220)
        self.table.setColumnWidth(1, 160)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.cellClicked.connect(self._on_table_clicked)
        right_layout.addWidget(self.table)

        body.addWidget(left_panel, stretch=4)
        body.addWidget(right_panel, stretch=2)
        main.addLayout(body, stretch=1)

    # ── 파일 로드 탭 ──────────────────────────────────────────────────
    def _build_file_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)

        # 로딩 모델 파일
        vl.addWidget(QLabel("로딩 모델 파일 (.h5 / .pth / .pkl):"))
        row_load = QHBoxLayout()
        self.edit_load_file = QLineEdit()
        self.edit_load_file.setReadOnly(True)
        self.edit_load_file.setPlaceholderText("파일 선택...")
        btn_load = QPushButton("찾기")
        btn_load.setFixedWidth(60)
        btn_load.clicked.connect(lambda: self._browse_file(self.edit_load_file))
        row_load.addWidget(self.edit_load_file)
        row_load.addWidget(btn_load)
        vl.addLayout(row_load)

        vl.addWidget(QLabel("언로딩 모델 파일 (.h5 / .pth / .pkl):"))
        row_unload = QHBoxLayout()
        self.edit_unload_file = QLineEdit()
        self.edit_unload_file.setReadOnly(True)
        self.edit_unload_file.setPlaceholderText("파일 선택...")
        btn_unload = QPushButton("찾기")
        btn_unload.setFixedWidth(60)
        btn_unload.clicked.connect(lambda: self._browse_file(self.edit_unload_file))
        row_unload.addWidget(self.edit_unload_file)
        row_unload.addWidget(btn_unload)
        vl.addLayout(row_unload)

        # 물리 파라미터
        param_box = QGroupBox("시뮬레이션 파라미터")
        pg = QVBoxLayout(param_box)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("최대 스트로크 (mm):"))
        self.spin_max_stroke_file = QDoubleSpinBox()
        self.spin_max_stroke_file.setRange(1, 9999)
        self.spin_max_stroke_file.setValue(200)
        r1.addWidget(self.spin_max_stroke_file)
        r1.addStretch()
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("전환 강성 k (kN/mm):"))
        self.spin_k_file = QDoubleSpinBox()
        self.spin_k_file.setRange(0.1, 9999)
        self.spin_k_file.setValue(1000.0)
        self.spin_k_file.setDecimals(1)
        r2.addWidget(self.spin_k_file)
        r2.addStretch()
        r3 = QHBoxLayout()
        r3.addWidget(QLabel("모드:"))
        self.combo_mode_file = QComboBox()
        self.combo_mode_file.addItems(["normal", "couple"])
        r3.addWidget(self.combo_mode_file)
        r3.addStretch()
        pg.addLayout(r1)
        pg.addLayout(r2)
        pg.addLayout(r3)
        vl.addWidget(param_box)
        vl.addStretch()
        return w

    # ── 커브 직접 입력 탭 ─────────────────────────────────────────────
    def _build_piecewise_tab(self):
        from gui.widgets.buffer_table_widget import BufferTableWidget
        from gui.widgets.plot_canvas import PlotCanvas
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)

        # 상단: 테이블 두 개 나란히 (각 캔버스 숨김)
        tables_row = QHBoxLayout()

        load_grp = QGroupBox("로딩 커브")
        ll = QVBoxLayout(load_grp)
        self.buf_load = BufferTableWidget(
            title="로딩", color="#1565C0", show_canvas=False,
            defaults=[
                (0.0, 0.0), (6.0, 43.0), (20.0, 108.0), (30.5, 165.0),
                (42.6, 220.0), (54.7, 278.0), (63.6, 349.0), (73.7, 419.0),
                (82.9, 495.0), (92.0, 560.0), (100.0, 648.0),
            ],
        )
        ll.addWidget(self.buf_load)
        tables_row.addWidget(load_grp)

        unload_grp = QGroupBox("언로딩 커브")
        ul = QVBoxLayout(unload_grp)
        self.buf_unload = BufferTableWidget(
            title="언로딩", color="#B71C1C", show_canvas=False,
            defaults=[
                (0.0, 0.0), (22.0, 10.0), (31.0, 17.0), (45.0, 27.0),
                (53.0, 30.0), (69.0, 42.0), (80.0, 47.0), (85.0, 71.0),
                (90.0, 93.0), (95.0, 129.0), (100.0, 167.0),
            ],
        )
        ul.addWidget(self.buf_unload)
        tables_row.addWidget(unload_grp)

        vl.addLayout(tables_row)

        param_box = QGroupBox("시뮬레이션 파라미터")
        pp = QHBoxLayout(param_box)
        pp.addWidget(QLabel("최대 스트로크 (mm):"))
        self.spin_max_stroke_csv = QDoubleSpinBox()
        self.spin_max_stroke_csv.setRange(1, 2000)
        self.spin_max_stroke_csv.setValue(100)
        pp.addWidget(self.spin_max_stroke_csv)
        pp.addSpacing(16)
        pp.addWidget(QLabel("전환 강성 k:"))
        self.spin_k_csv = QDoubleSpinBox()
        self.spin_k_csv.setRange(0.1, 9999)
        self.spin_k_csv.setValue(1000.0)
        self.spin_k_csv.setDecimals(1)
        pp.addWidget(self.spin_k_csv)
        pp.addSpacing(16)
        pp.addWidget(QLabel("모드:"))
        self.combo_mode_csv = QComboBox()
        self.combo_mode_csv.addItems(["normal", "couple"])
        pp.addWidget(self.combo_mode_csv)
        pp.addStretch()
        vl.addWidget(param_box)

        # 하단: 공유 캔버스 (로딩 + 언로딩 함께 표시)
        preview_grp = QGroupBox("커브 미리보기")
        pg_vl = QVBoxLayout(preview_grp)
        self.piecewise_canvas = PlotCanvas(figsize=(8, 5))
        pg_vl.addWidget(self.piecewise_canvas)
        vl.addWidget(preview_grp, stretch=3)

        # 두 테이블의 data_changed → 공유 캔버스 갱신
        self.buf_load.data_changed.connect(self._update_piecewise_plot)
        self.buf_unload.data_changed.connect(self._update_piecewise_plot)
        self._update_piecewise_plot()   # 초기 렌더

        return w

    def _update_piecewise_plot(self):
        """로딩/언로딩 커브를 하나의 캔버스에 겹쳐 표시."""
        import numpy as np
        from scipy.interpolate import interp1d
        if not hasattr(self, 'piecewise_canvas'):
            return
        self.piecewise_canvas.fig.clear()
        ax = self.piecewise_canvas.fig.add_subplot(111)

        for widget, label, color in [
            (self.buf_load,   "로딩",   "#1565C0"),
            (self.buf_unload, "언로딩", "#B71C1C"),
        ]:
            df = widget.get_curve_dataframe()
            if len(df) >= 2:
                x = df['Displacement [mm]'].values
                y = df['Force [kN]'].values
                x_fine = np.linspace(x.min(), x.max(), 300)
                fn = interp1d(x, y, bounds_error=False, fill_value='extrapolate')
                ax.plot(x_fine, fn(x_fine), color=color, linewidth=2, label=label)
                ax.scatter(x, y, color=color, zorder=5, s=40)

        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Force [kN]')
        ax.set_title('로딩 / 언로딩 커브')
        ax.legend()
        ax.grid(True)
        self.piecewise_canvas.canvas.draw()

    # ── Combined 직렬 탭 ──────────────────────────────────────────────
    def _build_combined_tab(self):
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.addWidget(QLabel("직렬 연결할 등록된 모델 2개를 선택하세요."))

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("모델 A:"))
        self.combo_comb_a = QComboBox()
        r1.addWidget(self.combo_comb_a)
        r1.addStretch()
        vl.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("모델 B:"))
        self.combo_comb_b = QComboBox()
        r2.addWidget(self.combo_comb_b)
        r2.addStretch()
        vl.addLayout(r2)
        vl.addStretch()
        return w

    def refresh_combined_combos(self):
        """등록 목록이 바뀔 때 Combined 탭 콤보박스 갱신."""
        names = list(self.model_registry.keys())
        for cb in (self.combo_comb_a, self.combo_comb_b):
            cur = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(names)
            if cur in names:
                cb.setCurrentText(cur)
            cb.blockSignals(False)

    # ── 모델 등록 ─────────────────────────────────────────────────────
    def _register_model(self):
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "입력 오류", "모델 이름을 입력하세요.")
            return
        exists = name in self.model_registry
        if name in self.model_registry:
            ans = QMessageBox.question(self, "덮어쓰기",
                                       f"'{name}' 이(가) 이미 존재합니다. 덮어쓰시겠습니까?")
            if ans != QMessageBox.StandardButton.Yes:
                return

        tab_idx = self.src_tabs.currentIndex()
        try:
            if tab_idx == 0:
                model_fn, model_type, model_info = self._build_from_curve()
            elif tab_idx == 1:
                model_fn, model_type, model_info = self._build_from_file()
            else:
                model_fn, model_type, model_info = self._build_combined()
        except Exception as e:
            QMessageBox.critical(self, "모델 생성 실패", str(e))
            return

        self.model_registry[name] = model_fn
        self.model_inputs[name] = model_info

        row = self._find_row_by_name(name)
        if row >= 0:
            self.table.setItem(row, 1, QTableWidgetItem(model_type))
        else:
            self._add_table_row(name, model_type)

        self.edit_name.setText(name)
        self.refresh_combined_combos()

        # 시그널 없이 부모(main_window)에 알리려면 직접 참조
        mw = self._find_main_window()
        if mw is not None:
            mw.on_registry_updated()

        if exists:
            QMessageBox.information(self, "등록 완료", f"'{name}' 모델이 갱신되었습니다.")
        else:
            QMessageBox.information(self, "등록 완료", f"'{name}' 모델이 등록되었습니다.")

    def _build_from_file(self):
        from core.simulation import (
            f_load_dnn, f_load_MPR, f_load_RBF,
            f_energy_absorber_creator,
        )
        load_path = self.edit_load_file.text()
        unload_path = self.edit_unload_file.text()
        if not load_path or not unload_path:
            raise ValueError("로딩 / 언로딩 파일을 모두 선택하세요.")

        def load_model(path):
            ext = path.lower()
            if ext.endswith('.h5') or ext.endswith('.pth'):
                return f_load_dnn(path)
            elif 'mpr' in path.lower():
                return f_load_MPR(path)
            else:
                return f_load_RBF(path)

        loading_model = load_model(load_path)
        unloading_model = load_model(unload_path)
        max_stroke = self.spin_max_stroke_file.value()
        k = self.spin_k_file.value()
        couple_mode = (self.combo_mode_file.currentText() == 'couple')  # bool

        buffer = f_energy_absorber_creator(loading_model, unloading_model,
                                            max_stroke, k, couple_mode)
        return buffer, "DNN/MPR/RBF", {
            "source": "file",
            "load_path": load_path,
            "unload_path": unload_path,
            "max_stroke": max_stroke,
            "k": k,
            "mode": self.combo_mode_file.currentText(),
        }

    def _build_from_curve(self):
        from core.simulation import (
            f_force_calculator_by_curve,
            f_energy_absorber_creator,
        )
        load_df = self.buf_load.get_curve_dataframe()
        unload_df = self.buf_unload.get_curve_dataframe()
        if load_df is None or len(load_df) < 2:
            raise ValueError("로딩 커브 데이터가 부족합니다 (최소 2점 필요).")
        if unload_df is None or len(unload_df) < 2:
            raise ValueError("언로딩 커브 데이터가 부족합니다 (최소 2점 필요).")

        # get_curve_dataframe()이 이미 올바른 형식의 DataFrame을 반환하므로
        # f_create_characteristic_curve 경유 없이 바로 사용
        loading_curve = load_df
        unloading_curve = unload_df
        max_stroke = self.spin_max_stroke_csv.value()
        k = self.spin_k_csv.value()
        mode_text = self.combo_mode_csv.currentText()
        couple_mode = (mode_text == 'couple')  # bool

        loading_model = f_force_calculator_by_curve(loading_curve)
        unloading_model = f_force_calculator_by_curve(unloading_curve)
        buffer = f_energy_absorber_creator(loading_model, unloading_model,
                                            max_stroke, k, couple_mode)
        return buffer, "Piecewise Curve", {
            "source": "csv",
            "loading_curve": loading_curve.copy(),
            "unloading_curve": unloading_curve.copy(),
            "max_stroke": max_stroke,
            "k": k,
            "mode": mode_text,
        }

    def _build_combined(self):
        from core.simulation import c_combined_energy_absorber_creator
        name_a = self.combo_comb_a.currentText()
        name_b = self.combo_comb_b.currentText()
        if not name_a or not name_b:
            raise ValueError("Combined에 사용할 모델 2개를 선택하세요.")
        buf_a = self.model_registry.get(name_a)
        buf_b = self.model_registry.get(name_b)
        if buf_a is None or buf_b is None:
            raise ValueError("선택한 모델이 레지스트리에 없습니다.")
        combined = c_combined_energy_absorber_creator(buf_a, buf_b)
        return combined.force_calculator, f"Combined({name_a}+{name_b})", {
            "source": "combined",
            "model_a": name_a,
            "model_b": name_b,
        }

    def _add_table_row(self, name: str, model_type: str):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(model_type))
        btn_del = QPushButton("삭제")
        btn_del.setFixedHeight(22)
        btn_del.clicked.connect(lambda _, n=name: self._delete_row(n))
        self.table.setCellWidget(r, 2, btn_del)

    def _delete_row(self, name: str):
        row = self._find_row_by_name(name)
        if row < 0:
            return
        self.model_registry.pop(name, None)
        self.model_inputs.pop(name, None)
        self.table.removeRow(row)
        self.refresh_combined_combos()
        mw = self._find_main_window()
        if mw is not None:
            mw.on_registry_updated()

    def _find_row_by_name(self, name: str) -> int:
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None and item.text() == name:
                return r
        return -1

    def _on_table_clicked(self, row: int, _col: int):
        item = self.table.item(row, 0)
        if item is None:
            return

        name = item.text().strip()
        if not name:
            return
        self.edit_name.setText(name)

        spec = self.model_inputs.get(name)
        if not spec or spec.get("source") != "csv":
            return

        self.src_tabs.setCurrentIndex(0)
        self.buf_load.set_curve_dataframe(spec["loading_curve"])
        self.buf_unload.set_curve_dataframe(spec["unloading_curve"])
        self.spin_max_stroke_csv.setValue(spec["max_stroke"])
        self.spin_k_csv.setValue(spec["k"])
        self.combo_mode_csv.setCurrentText(spec["mode"])
        self._update_piecewise_plot()

    def _browse_file(self, edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "모델 파일 선택", "",
            "Model files (*.h5 *.pth *.pkl);;All files (*)")
        if path:
            edit.setText(path)

    def _find_main_window(self):
        w = self.parent()
        while w is not None:
            if hasattr(w, 'on_registry_updated'):
                return w
            w = w.parent()
        return None
