"""
main.py
GUI 1D 시뮬레이션 프로세서 진입점.
실행: python main.py
"""
#라이브러리 임포트

import sys
import os

# 패키지 루트를 sys.path에 추가 (main.py 가 GUI_1D_Simulation/ 아래에 있을 때)
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from gui.main_window import MainWindow


def main():
    # High-DPI 지원
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("1D 충돌 시뮬레이션 프로세서")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
