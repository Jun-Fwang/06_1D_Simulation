"""
gui/widgets/buffer_table_widget.py
완충기 특성커브 직접 입력 위젯.
- QTableWidget (Displacement / Force 2열) + 실시간 Matplotlib 그래프
- 행 추가 / 삭제 버튼
- CSV 불러오기 / 저장 버튼
- get_curve_dataframe() → simulation.f_force_calculator_by_curve 에 전달
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QFileDialog, QLabel,
    QMessageBox,
    QSplitter,
)
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from gui.widgets.plot_canvas import PlotCanvas


class BufferTableWidget(QWidget):
    """단일 커브(로딩 또는 언로딩) 입력+시각화 위젯."""

    data_changed = pyqtSignal()   # 데이터가 바뀔 때 방출 (공유 캔버스 연동용)

    def __init__(self, title: str = "커브", color: str = "C0",
                 show_canvas: bool = True, parent=None):
        super().__init__(parent)
        self._color = color
        self._show_canvas = show_canvas
        self._building = False          # cellChanged 재귀 방지

        # ── 레이아웃 ──────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # 왼쪽: 테이블 + 버튼
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(2, 2, 2, 2)

        lv.addWidget(QLabel(f"<b>{title}</b>"))

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Displacement [mm]", "Force [kN]"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumWidth(240)
        self.table.cellChanged.connect(self._on_cell_changed)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ContiguousSelection)
        self.table.installEventFilter(self)
        self.table.viewport().installEventFilter(self)
        lv.addWidget(self.table)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("+ 행 추가")
        btn_del = QPushButton("- 행 삭제")
        btn_normalize = QPushButton("정렬/원점 보정")
        btn_csv_load = QPushButton("CSV 불러오기")
        btn_csv_save = QPushButton("CSV 저장")
        btn_clear = QPushButton("초기화")
        btn_add.clicked.connect(self._add_row)
        btn_del.clicked.connect(self._del_row)
        btn_normalize.clicked.connect(self._normalize_curve_rows)
        btn_csv_load.clicked.connect(self._load_csv)
        btn_csv_save.clicked.connect(self._save_csv)
        btn_clear.clicked.connect(self._clear_all_rows)
        for b in (btn_add, btn_del, btn_normalize, btn_csv_load, btn_csv_save, btn_clear):
            btn_row.addWidget(b)
        lv.addLayout(btn_row)
        splitter.addWidget(left)

        # 오른쪽: 개별 그래프 (show_canvas=False 이면 숨김)
        self.canvas = PlotCanvas(figsize=(5, 4))
        if show_canvas:
            splitter.addWidget(self.canvas)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 2)
        else:
            self.canvas.setVisible(False)

        # 기본 포인트 세팅
        self._set_default_rows()

    # ── 기본값 ────────────────────────────────────────────────────────
    def _set_default_rows(self):
        defaults = [(0.0, 0.0), (50.0, 300.0), (100.0, 600.0), (150.0, 900.0)]
        self._building = True
        self.table.setRowCount(0)
        for d, f in defaults:
            self._insert_row(d, f)
        self._building = False
        self._update_plot()

    def _insert_row(self, d: float = 0.0, f: float = 0.0):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(str(d)))
        self.table.setItem(r, 1, QTableWidgetItem(str(f)))

    # ── 버튼 핸들러 ───────────────────────────────────────────────────
    def _add_row(self):
        self._building = True
        self._insert_row()
        self._building = False
        self._update_plot()

    def _del_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self._update_plot()

    def _load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV 불러오기", "", "CSV (*.csv)")
        if not path:
            return
        df = None
        last_error = None
        for encoding in ('utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin1'):
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
                continue

        if df is None:
            QMessageBox.critical(
                self,
                "CSV 불러오기 실패",
                "CSV 파일 인코딩을 해석하지 못했습니다.\n"
                "UTF-8 또는 CP949 형식으로 다시 저장한 뒤 시도해 주세요.\n\n"
                f"파일: {os.path.basename(path)}\n"
                f"오류: {last_error}"
            )
            return

        disp_col = next((c for c in df.columns if 'Disp' in c or 'disp' in c), df.columns[0])
        force_col = next((c for c in df.columns if 'Force' in c or 'force' in c), df.columns[1])
        self._building = True
        self.table.setRowCount(0)
        for _, row in df.iterrows():
            self._insert_row(float(row[disp_col]), float(row[force_col]))
        self._building = False
        self._update_plot()

    def _paste_from_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text.strip():
            return

        rows = [line for line in text.splitlines() if line.strip()]
        if not rows:
            return

        start_row = max(self.table.currentRow(), 0)
        start_col = max(self.table.currentColumn(), 0)

        parsed_rows = []
        for line in rows:
            delimiter = '\t' if '\t' in line else ','
            cols = [cell.strip().strip('"') for cell in line.split(delimiter)]
            if any(col for col in cols):
                parsed_rows.append(cols)

        if not parsed_rows:
            return

        if len(parsed_rows[0]) >= 2:
            try:
                float(parsed_rows[0][0])
                float(parsed_rows[0][1])
            except ValueError:
                parsed_rows = parsed_rows[1:]

        if not parsed_rows:
            return

        required_rows = start_row + len(parsed_rows)
        while self.table.rowCount() < required_rows:
            self.table.insertRow(self.table.rowCount())

        self._building = True
        for row_offset, cols in enumerate(parsed_rows):
            for col_offset, value in enumerate(cols[:2]):
                target_col = start_col + col_offset
                if target_col >= self.table.columnCount():
                    continue
                self.table.setItem(
                    start_row + row_offset,
                    target_col,
                    QTableWidgetItem(value),
                )
        self._building = False
        self._update_plot()

    def _normalize_curve_rows(self):
        """테이블 데이터를 displacement 기준 정렬하고 필요 시 (0, 0) 점을 추가한다."""
        df = self.get_curve_dataframe()
        if df.empty:
            df = pd.DataFrame([(0.0, 0.0)], columns=['Displacement [mm]', 'Force [kN]'])
        else:
            has_origin = ((df['Displacement [mm]'] == 0.0) & (df['Force [kN]'] == 0.0)).any()
            if not has_origin:
                df = pd.concat(
                    [df, pd.DataFrame([(0.0, 0.0)], columns=['Displacement [mm]', 'Force [kN]'])],
                    ignore_index=True,
                )
            df = df.sort_values('Displacement [mm]').reset_index(drop=True)

        self.set_curve_dataframe(df)

    def eventFilter(self, obj, event):
        if obj in (self.table, self.table.viewport()) and event.type() == QEvent.Type.KeyPress:
            if event.matches(QKeySequence.StandardKey.Paste):
                self._paste_from_clipboard()
                return True
            if event.key() == Qt.Key.Key_Insert and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._paste_from_clipboard()
                return True
        return super().eventFilter(obj, event)

    def _save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "CSV 저장", "", "CSV (*.csv)")
        if path:
            self.get_curve_dataframe().to_csv(path, index=False)

    def _clear_all_rows(self):
        self._building = True
        self.table.setRowCount(0)
        self._building = False
        self._update_plot()

    # ── 셀 변경 → 그래프 갱신 ─────────────────────────────────────────
    def _on_cell_changed(self, _row, _col):
        if not self._building:
            self._update_plot()

    def _update_plot(self):
        self.data_changed.emit()   # 공유 캔버스가 있으면 외부에서 처리
        if not self._show_canvas:
            return
        df = self.get_curve_dataframe()
        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        if len(df) >= 2:
            x_fine = np.linspace(df['Displacement [mm]'].min(),
                                 df['Displacement [mm]'].max(), 300)
            fn = interp1d(df['Displacement [mm]'], df['Force [kN]'],
                          bounds_error=False, fill_value='extrapolate')
            ax.plot(x_fine, fn(x_fine), color=self._color, linewidth=2)
        ax.scatter(df['Displacement [mm]'], df['Force [kN]'],
                   color=self._color, zorder=5, s=40)
        ax.set_xlabel('Displacement [mm]')
        ax.set_ylabel('Force [kN]')
        ax.grid(True)
        ax.set_title('특성 커브')
        self.canvas.draw()

    # ── 공개 API ──────────────────────────────────────────────────────
    def get_curve_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in range(self.table.rowCount()):
            try:
                d = float(self.table.item(r, 0).text())
                f = float(self.table.item(r, 1).text())
                rows.append((d, f))
            except (AttributeError, ValueError):
                pass
        df = pd.DataFrame(rows, columns=['Displacement [mm]', 'Force [kN]'])
        return df.sort_values('Displacement [mm]').reset_index(drop=True)

    def set_curve_dataframe(self, df: pd.DataFrame):
        self._building = True
        self.table.setRowCount(0)
        for _, row in df.iterrows():
            self._insert_row(float(row['Displacement [mm]']), float(row['Force [kN]']))
        self._building = False
        self._update_plot()
