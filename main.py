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
from PyQt6.QtGui import QIcon
from gui.main_window import MainWindow


def resource_path(relative_path: str) -> str:
    """개발 환경과 PyInstaller(onefile) 실행 환경 모두에서 리소스 경로를 반환."""
    base_path = getattr(sys, "_MEIPASS", ROOT)
    return os.path.join(base_path, relative_path)


def main():
    # High-DPI 지원
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("1D 충돌 시뮬레이션 프로세서")
    app.setStyle("Fusion")

    # 창/작업표시줄 아이콘: --icon 옵션은 exe 파일 아이콘만 지정하므로 별도 설정 필요
    icon_path = resource_path("train_front_icon.ico")
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)

    window = MainWindow()
    if os.path.exists(icon_path):
        window.setWindowIcon(app_icon)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
