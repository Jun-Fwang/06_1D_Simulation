"""
core/workers.py
QThread 기반 백그라운드 워커 클래스들.
tqdm 대신 progress 시그널로 QProgressBar 연동.
"""
from PyQt6.QtCore import QThread, pyqtSignal
from core.simulation import (
    f_rigidwall_simulation,
    f_coupling_simulation,
)


class RigidWallWorker(QThread):
    """고정벽 시뮬레이션 워커."""
    progress = pyqtSignal(int)          # 0~100
    finished = pyqtSignal(object)       # pd.DataFrame
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, car_mass, buffer_model, impact_velocity,
                 dt, termination_time, energy_increase_limit=0.01):
        super().__init__()
        self.car_mass = car_mass
        self.buffer_model = buffer_model
        self.impact_velocity = impact_velocity
        self.dt = dt
        self.termination_time = termination_time
        self.energy_increase_limit = energy_increase_limit

    def run(self):
        try:
            result = f_rigidwall_simulation(
                car_mass=self.car_mass,
                buffer_model=self.buffer_model,
                impact_velocity=self.impact_velocity,
                dt=self.dt,
                termination_time=self.termination_time,
                energy_increase_limit=self.energy_increase_limit,
                progress_callback=self.progress.emit,
                interrupt_callback=self.isInterruptionRequested,
            )
            self.finished.emit(result)
        except Exception as e:
            if str(e) == "__INTERRUPTED__":
                self.cancelled.emit()
            else:
                self.error.emit(str(e))


class CouplingWorker(QThread):
    """연결 시뮬레이션 워커."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, car_mass_list, coupler_model_list,
                 mu_kinetic_list, mu_static_list, car_velocity_list,
                 termination_time, dt, energy_increase_limit=0.01,
                 car_spring_enabled=False, car_stiffness_list=None):
        super().__init__()
        self.car_mass_list = car_mass_list
        self.coupler_model_list = coupler_model_list
        self.mu_kinetic_list = mu_kinetic_list
        self.mu_static_list = mu_static_list
        self.car_velocity_list = car_velocity_list
        self.termination_time = termination_time
        self.dt = dt
        self.energy_increase_limit = energy_increase_limit
        self.car_spring_enabled = car_spring_enabled
        self.car_stiffness_list = car_stiffness_list or []

    def run(self):
        try:
            result = f_coupling_simulation(
                car_mass_list=self.car_mass_list,
                coupler_model_list=self.coupler_model_list,
                mu_kinetic_friction_list=self.mu_kinetic_list,
                mu_static_friction_list=self.mu_static_list,
                car_velocity_list=self.car_velocity_list,
                termination_time=self.termination_time,
                dt=self.dt,
                energy_increase_limit=self.energy_increase_limit,
                car_spring_enabled=self.car_spring_enabled,
                car_stiffness_list=self.car_stiffness_list,
                progress_callback=self.progress.emit,
                interrupt_callback=self.isInterruptionRequested,
            )
            self.finished.emit(result)
        except Exception as e:
            if str(e) == "__INTERRUPTED__":
                self.cancelled.emit()
            else:
                self.error.emit(str(e))


class DataProcessWorker(QThread):
    """데이터 처리 워커 (A 노트북)."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
