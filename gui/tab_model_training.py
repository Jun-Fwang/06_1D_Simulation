"""
gui/tab_model_training.py
DNN 모델 훈련 탭 (탭 3).
C 노트북 기능: 데이터 로드 → 모델 구성 파라미터 설정 → 훈련 → 저장.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QFileDialog, QProgressBar, QSplitter, QTextEdit, QCheckBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class _TrainWorker(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            import numpy as np
            import pandas as pd

            df = pd.read_csv(self.cfg['data_path'])
            self.log.emit(f"데이터 로드: {df.shape}\n")

            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.reshape(-1, 1)

            # 정규화
            from sklearn.preprocessing import StandardScaler
            sx = StandardScaler(); sy = StandardScaler()
            X_s = sx.fit_transform(X)
            y_s = sy.fit_transform(y).ravel()

            # 훈련/검증 분리
            from sklearn.model_selection import train_test_split
            X_tr, X_va, y_tr, y_va = train_test_split(
                X_s, y_s, test_size=self.cfg['val_ratio'], random_state=42)

            backend = self.cfg['backend']
            epochs = self.cfg['epochs']
            batch  = self.cfg['batch_size']
            lr     = self.cfg['learning_rate']
            layers = self.cfg['hidden_layers']
            save_path = self.cfg['save_path']

            history = None
            if backend == 'TensorFlow':
                import tensorflow as tf
                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Input(shape=(X_tr.shape[1],)))
                for units in layers:
                    model.add(tf.keras.layers.Dense(units, activation='relu'))
                model.add(tf.keras.layers.Dense(1))
                model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                               loss='mse', metrics=['mae'])
                self.log.emit("TF 모델 학습 시작...")
                class _CB(tf.keras.callbacks.Callback):
                    def __init__(cb_self, total, sig):
                        super().__init__()
                        cb_self.total = total; cb_self.sig = sig
                    def on_epoch_end(cb_self, epoch, logs=None):
                        pct = int((epoch + 1) / cb_self.total * 100)
                        cb_self.sig.emit(pct)
                        if logs:
                            cb_self.log.emit(
                                f"Epoch {epoch+1}/{cb_self.total}  "
                                f"loss={logs.get('loss',0):.4f}  "
                                f"val_loss={logs.get('val_loss',0):.4f}\n")
                cb = _CB(epochs, self.progress)
                cb.log = self.log
                history = model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch,
                                    validation_data=(X_va, y_va),
                                    callbacks=[cb], verbose=0)
                model.save(save_path)
                self.log.emit(f"모델 저장: {save_path}")
                self.finished.emit({'history': history.history,
                                    'save_path': save_path})

            elif backend == 'PyTorch':
                import torch
                import torch.nn as nn

                class _Net(nn.Module):
                    def __init__(self, in_d, hiddens):
                        super().__init__()
                        dims = [in_d] + hiddens + [1]
                        layers_list = []
                        for i in range(len(dims)-1):
                            layers_list.append(nn.Linear(dims[i], dims[i+1]))
                            if i < len(dims)-2:
                                layers_list.append(nn.ReLU())
                        self.net = nn.Sequential(*layers_list)
                    def forward(self, x):
                        return self.net(x)

                X_t = torch.FloatTensor(X_tr)
                y_t = torch.FloatTensor(y_tr).unsqueeze(1)
                Xv_t = torch.FloatTensor(X_va)
                yv_t = torch.FloatTensor(y_va).unsqueeze(1)
                model = _Net(X_tr.shape[1], layers)
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()
                dset = torch.utils.data.TensorDataset(X_t, y_t)
                loader = torch.utils.data.DataLoader(dset, batch_size=batch)
                loss_hist = []
                self.log.emit("PyTorch 모델 학습 시작...")
                for ep in range(epochs):
                    model.train()
                    ep_loss = 0.0
                    for xb, yb in loader:
                        opt.zero_grad()
                        out = model(xb)
                        loss = criterion(out, yb)
                        loss.backward()
                        opt.step()
                        ep_loss += loss.item()
                    ep_loss /= len(loader)
                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion(model(Xv_t), yv_t).item()
                    loss_hist.append((ep_loss, val_loss))
                    pct = int((ep + 1) / epochs * 100)
                    self.progress.emit(pct)
                    self.log.emit(
                        f"Epoch {ep+1}/{epochs}  loss={ep_loss:.4f}  "
                        f"val_loss={val_loss:.4f}\n")
                torch.save(model.state_dict(), save_path)
                self.log.emit(f"모델 저장: {save_path}")
                self.finished.emit({'history': loss_hist,
                                    'save_path': save_path})
            else:
                raise ValueError(f"지원하지 않는 백엔드: {backend}")

        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


class ModelTrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._init_ui()

    def _init_ui(self):
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)

        # ─ 데이터 / 저장 경로 ─
        path_box = QGroupBox("파일 경로")
        pb = QVBoxLayout(path_box)
        r1 = QHBoxLayout()
        r1.addWidget(QLabel("훈련 데이터 (CSV):"))
        self.edit_data = QLineEdit()
        self.edit_data.setReadOnly(True)
        btn_d = QPushButton("찾기"); btn_d.setFixedWidth(60)
        btn_d.clicked.connect(lambda: self._browse(self.edit_data,
            "CSV (*.csv);;All (*)"))
        r1.addWidget(self.edit_data); r1.addWidget(btn_d)
        pb.addLayout(r1)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("모델 저장 경로:"))
        self.edit_save = QLineEdit()
        btn_s = QPushButton("찾기"); btn_s.setFixedWidth(60)
        btn_s.clicked.connect(lambda: self._browse_save())
        r2.addWidget(self.edit_save); r2.addWidget(btn_s)
        pb.addLayout(r2)
        main.addWidget(path_box)

        # ─ 모델 구성 ─
        cfg_box = QGroupBox("모델 구성")
        cg = QVBoxLayout(cfg_box)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("백엔드:"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["TensorFlow", "PyTorch"])
        self.combo_backend.currentTextChanged.connect(self._update_save_ext)
        r3.addWidget(self.combo_backend)
        r3.addSpacing(20)
        r3.addWidget(QLabel("히든 레이어 (쉼표 구분):"))
        self.edit_layers = QLineEdit("64,64,32")
        self.edit_layers.setFixedWidth(120)
        r3.addWidget(self.edit_layers)
        r3.addStretch()
        cg.addLayout(r3)

        r4 = QHBoxLayout()
        r4.addWidget(QLabel("에폭:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 10000)
        self.spin_epochs.setValue(200)
        r4.addWidget(self.spin_epochs)
        r4.addSpacing(20)
        r4.addWidget(QLabel("배치 크기:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 4096)
        self.spin_batch.setValue(32)
        r4.addWidget(self.spin_batch)
        r4.addSpacing(20)
        r4.addWidget(QLabel("학습률:"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(1e-6, 1)
        self.spin_lr.setValue(0.001)
        self.spin_lr.setDecimals(6)
        r4.addWidget(self.spin_lr)
        r4.addSpacing(20)
        r4.addWidget(QLabel("검증 비율:"))
        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.05, 0.5)
        self.spin_val.setValue(0.2)
        self.spin_val.setDecimals(2)
        r4.addWidget(self.spin_val)
        r4.addStretch()
        cg.addLayout(r4)
        main.addWidget(cfg_box)

        # ─ 실행 ─
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶ 모델 훈련 시작")
        self.btn_run.setStyleSheet(
            "background:#7B1FA2; color:white; font-weight:bold;")
        self.btn_run.setFixedHeight(32)
        self.btn_run.clicked.connect(self._run)
        self.btn_stop = QPushButton("■ 중지")
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        btn_row.addWidget(self.btn_run); btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        main.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        main.addWidget(self.progress_bar)

        # ─ 결과 ─
        from gui.widgets.plot_canvas import PlotCanvas
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumWidth(320)
        self.log_text.setPlaceholderText("훈련 로그...")
        self.loss_canvas = PlotCanvas(figsize=(6, 4))
        splitter.addWidget(self.log_text)
        splitter.addWidget(self.loss_canvas)
        splitter.setSizes([300, 500])
        main.addWidget(splitter, stretch=1)

    def _browse(self, edit, filt):
        p, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", filt)
        if p: edit.setText(p)

    def _browse_save(self):
        backend = self.combo_backend.currentText()
        ext = "*.h5" if backend == "TensorFlow" else "*.pth"
        p, _ = QFileDialog.getSaveFileName(self, "저장 경로", "", f"Model ({ext})")
        if p: self.edit_save.setText(p)

    def _update_save_ext(self, backend):
        path = self.edit_save.text()
        if path.endswith('.h5') and backend == 'PyTorch':
            self.edit_save.setText(path[:-3] + '.pth')
        elif path.endswith('.pth') and backend == 'TensorFlow':
            self.edit_save.setText(path[:-4] + '.h5')

    def _parse_layers(self):
        try:
            return [int(x.strip()) for x in self.edit_layers.text().split(',') if x.strip()]
        except ValueError:
            return [64, 64, 32]

    def _run(self):
        from PyQt6.QtWidgets import QMessageBox
        data_path = self.edit_data.text().strip()
        save_path = self.edit_save.text().strip()
        if not data_path:
            QMessageBox.warning(self, "경고", "훈련 데이터 파일을 선택하세요."); return
        if not save_path:
            QMessageBox.warning(self, "경고", "모델 저장 경로를 입력하세요."); return
        cfg = {
            'data_path': data_path,
            'save_path': save_path,
            'backend': self.combo_backend.currentText(),
            'epochs': self.spin_epochs.value(),
            'batch_size': self.spin_batch.value(),
            'learning_rate': self.spin_lr.value(),
            'val_ratio': self.spin_val.value(),
            'hidden_layers': self._parse_layers(),
        }
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self._worker = _TrainWorker(cfg)
        self._worker.progress.connect(self.progress_bar.setValue)
        self._worker.log.connect(self.log_text.insertPlainText)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.wait(3000)
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False)

    def _on_done(self, result):
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False)
        history = result.get('history', {})
        self._plot_loss(history)

    def _on_error(self, msg):
        from PyQt6.QtWidgets import QMessageBox
        self.btn_run.setEnabled(True); self.btn_stop.setEnabled(False)
        QMessageBox.critical(self, "훈련 오류", msg)

    def _plot_loss(self, history):
        ax = self.loss_canvas.fig.clear()
        ax = self.loss_canvas.fig.add_subplot(111)
        if isinstance(history, dict):
            if 'loss' in history: ax.plot(history['loss'], label='Train loss')
            if 'val_loss' in history: ax.plot(history['val_loss'], label='Val loss')
        elif isinstance(history, list):  # PyTorch: [(train, val), ...]
            tr = [x[0] for x in history]
            vl = [x[1] for x in history]
            ax.plot(tr, label='Train loss')
            ax.plot(vl, label='Val loss')
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("학습 손실"); ax.legend()
        self.loss_canvas.canvas.draw()
