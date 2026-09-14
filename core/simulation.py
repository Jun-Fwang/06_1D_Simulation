"""
core/simulation.py
D_Dynamic_Simulation_Code.ipynb (TF) + D2 (PyTorch) 의 함수 추출
TF / PyTorch 중 설치된 것만 사용. 미설치 시 해당 모델 타입만 비활성화.
"""
import pickle
import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import interp1d, RegularGridInterpolator
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ─── TensorFlow / PyTorch 조건부 import ─────────────────────────────────────

_HAS_TF = False
_HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as tf_load_model
    _HAS_TF = True
except ImportError:
    pass

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

# ─── 모델 로딩 ───────────────────────────────────────────────────────────────

def f_load_dnn(loading_address: str, backend: str = 'auto'):
    """
    DNN 불러오기.
    backend: 'auto' | 'tensorflow' | 'pytorch'
    """
    if backend == 'auto':
        if loading_address.endswith('.h5') or loading_address.endswith('.keras'):
            backend = 'tensorflow'
        elif loading_address.endswith('.pth'):
            backend = 'pytorch'
        else:
            backend = 'tensorflow' if _HAS_TF else 'pytorch'

    if backend == 'tensorflow':
        if not _HAS_TF:
            raise RuntimeError("TensorFlow가 설치되어 있지 않습니다.")
        path = loading_address if loading_address.endswith(('.h5', '.keras')) else loading_address + '.h5'
        return tf_load_model(path)
    else:
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch가 설치되어 있지 않습니다.")
        path = loading_address if loading_address.endswith('.pth') else loading_address + '.pth'
        return torch.load(path, weights_only=False)


def f_load_MPR(loading_address: str):
    path = loading_address if loading_address.endswith('.pkl') else loading_address + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


def f_load_RBF(loading_address: str):
    path = loading_address if loading_address.endswith('.pkl') else loading_address + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─── 피스와이즈 보간 모델 ────────────────────────────────────────────────────

def f_force_calculator_by_piecewise(prediction_model, model_type: str,
                                    max_disp: int, max_vel: int):
    d = np.linspace(0, max_disp, max_disp * 10 + 1)
    v = np.linspace(0, max_vel, max_vel * 100 + 1)
    dd, vv = np.meshgrid(d, v)
    input_data = np.column_stack([dd.ravel(), vv.ravel()])

    if model_type == 'DNN_TF':
        piecewise_ff = prediction_model.predict(input_data, verbose=0)
        ff = piecewise_ff.reshape(max_vel * 100 + 1, max_disp * 10 + 1)
        piecewise_model = RegularGridInterpolator((v, d), ff)
        def _calc(displacement, velocity):
            point = np.array([[velocity, displacement]])
            return float(piecewise_model(point))
        return _calc

    if model_type == 'DNN_PT':
        import torch
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        with torch.no_grad():
            prediction_model.eval()
            piecewise_ff = prediction_model(input_tensor).numpy()
        ff = piecewise_ff.reshape(max_vel * 100 + 1, max_disp * 10 + 1)
        piecewise_model = RegularGridInterpolator((v, d), ff)
        def _calc(displacement, velocity):
            point = np.array([[velocity, displacement]])
            return float(piecewise_model(point))
        return _calc

    if model_type == 'MPR':
        def _calc(displacement, velocity):
            tbl = prediction_model['feature_transform'].fit_transform(
                pd.DataFrame([{'Displacement [mm]': displacement, 'Velocity [mm/ms]': velocity}])
            )
            return float(prediction_model['weights'].predict(tbl))
        return _calc

    if model_type == 'RBF':
        def _calc(displacement, velocity):
            return float(prediction_model(displacement, velocity))
        return _calc

    raise ValueError(f"Unknown model_type: {model_type}")


# ─── 특성 커브 기반 모델 ─────────────────────────────────────────────────────

def f_load_curve(loading_address: str) -> pd.DataFrame:
    path = loading_address if loading_address.endswith('.csv') else loading_address + '.csv'
    return pd.read_csv(path)


def f_create_characteristic_curve(x1, x2, y1, y2) -> pd.DataFrame:
    return pd.DataFrame({'Displacement [mm]': [x1, x2], 'Force [kN]': [y1, y2]})


def f_force_calculator_by_curve(characteristic_curve: pd.DataFrame):
    area = np.trapezoid(characteristic_curve['Force [kN]'].values,
                    characteristic_curve['Displacement [mm]'].values)
    print(f'Max Absorbed Energy: {round(area / 1000, 1)} kJ')
    interp = interp1d(characteristic_curve['Displacement [mm]'],
                      characteristic_curve['Force [kN]'],
                      bounds_error=False, fill_value=0.1)
    def _calc(d, v):
        return float(interp(d))
    return _calc


# ─── 완충기 모델 생성 ────────────────────────────────────────────────────────

def f_energy_absorber_creator(loading_model, unloading_model,
                               max_stroke: float, transition_k: float,
                               couple_mode: bool = False):
    def _calc(d, v, pre_f, dt):
        if couple_mode:
            d, v = d / 2, v / 2
        tension = False
        if d < 0:
            d, v, pre_f = -d, -v, -pre_f
            tension = True
        transition_force = pre_f + transition_k * v * dt
        if v > 0:
            pred = loading_model(d, v)
            if transition_force < pred or d > max_stroke:
                pred = transition_force
        else:
            pred = unloading_model(d, v)
            if transition_force > pred:
                pred = transition_force
        if pred < 0:
            pred = 0.0
        if tension:
            pred = -pred
        return pred
    return _calc


# ─── 완충기 2개 직렬 연동 클래스 ────────────────────────────────────────────

class c_combined_energy_absorber_creator:
    """완충기 2개를 직렬로 연결하여 힘을 최적화로 분배하는 클래스."""

    def __init__(self, buffer_1_model, buffer_2_model):
        self.history_data = []
        self.T = 0
        self.dt = 0
        self.buffer_1_model = buffer_1_model
        self.buffer_2_model = buffer_2_model
        self.total_displacement = 0
        self.buffer_1_displacement = 0
        self.buffer_2_displacement = 0
        self.buffer_1_velocity = 0
        self.buffer_2_velocity = 0
        self.buffer_1_force = 0
        self.buffer_2_force = 0
        self.buffer_1_internal_energy = 0
        self.buffer_2_internal_energy = 0

    def loss_calculator(self, D1):
        V1 = (D1 - self.buffer_1_displacement) / self.dt
        F1 = self.buffer_1_model(D1, V1, self.buffer_1_force, self.dt)
        D2 = self.total_displacement - D1
        V2 = (D2 - self.buffer_2_displacement) / self.dt
        F2 = self.buffer_2_model(D2, V2, self.buffer_2_force, self.dt)
        return abs(F1 - F2)

    def force_calculator(self, d, v, pre_f, dt):
        self.history_data.append({
            'Time [ms]': self.T,
            'Buffer_1_Displacement [mm]': round(self.buffer_1_displacement / 2, 3),
            'Buffer_2_Displacement [mm]': round(self.buffer_2_displacement / 2, 3),
            'Buffer_1_Velocity [mm/ms]': round(self.buffer_1_velocity / 2, 3),
            'Buffer_2_Velocity [mm/ms]': round(self.buffer_2_velocity / 2, 3),
            'Buffer_1_Force [kN]': round(self.buffer_1_force, 3),
            'Buffer_2_Force [kN]': round(self.buffer_2_force, 3),
            'Buffer_1_Internal_Energy [kJ]': round(self.buffer_1_internal_energy / 2000, 3),
            'Buffer_2_Internal_Energy [kJ]': round(self.buffer_2_internal_energy / 2000, 3),
        })
        self.dt = dt
        self.T += dt
        delta_total = d - self.total_displacement
        self.total_displacement = d

        if delta_total > 0:
            D1_bounds = [(self.buffer_1_displacement,
                          self.buffer_1_displacement + delta_total)]
        else:
            D1_bounds = [(self.buffer_1_displacement + delta_total,
                          self.buffer_1_displacement)]

        D1_list = [self.buffer_1_displacement + delta_total / 4 * i for i in range(5)]
        minimum_loss = float('inf')
        best_D1 = self.buffer_1_displacement
        for init in D1_list:
            opt = sp.optimize.minimize(self.loss_calculator, init,
                                       method='Nelder-Mead', bounds=D1_bounds)
            if opt.fun < minimum_loss:
                minimum_loss = opt.fun
                best_D1 = opt.x[0]

        next_d1 = best_D1
        delta_d1 = next_d1 - self.buffer_1_displacement
        self.buffer_1_displacement = next_d1
        self.buffer_1_velocity = delta_d1 / dt
        self.buffer_1_force = self.buffer_1_model(
            self.buffer_1_displacement, self.buffer_1_velocity, self.buffer_1_force, dt
        )
        self.buffer_1_internal_energy += delta_d1 * self.buffer_1_force

        next_d2 = self.total_displacement - best_D1
        delta_d2 = next_d2 - self.buffer_2_displacement
        self.buffer_2_displacement = next_d2
        self.buffer_2_velocity = delta_d2 / dt
        self.buffer_2_force = self.buffer_2_model(
            self.buffer_2_displacement, self.buffer_2_velocity, self.buffer_2_force, dt
        )
        self.buffer_2_internal_energy += delta_d2 * self.buffer_2_force

        return (self.buffer_1_force + self.buffer_2_force) / 2

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_data)


# ─── 고정벽 시뮬레이션 ──────────────────────────────────────────────────────

def f_rigidwall_simulation(car_mass: float, buffer_model,
                            impact_velocity: float, dt: float,
                            termination_time: float,
                            energy_increase_limit: float,
                            progress_callback=None,
                            interrupt_callback=None) -> pd.DataFrame:
    history_list = []
    t = 0.0
    car_displacement = 0.0
    car_velocity = impact_velocity
    car_acceleration = 0.0
    buffer_displacement = 0.0
    buffer_velocity = 0.0
    buffer_force = 0.0
    initial_energy = kinetic_energy = global_energy = 0.5 * car_mass * car_velocity ** 2
    internal_energy = 0.0

    total_steps = int(termination_time / dt)
    for n in range(total_steps):
        if interrupt_callback and interrupt_callback():
            raise RuntimeError("__INTERRUPTED__")
        if progress_callback:
            progress_callback(int(n / total_steps * 100))
        if abs(global_energy / initial_energy - 1) > energy_increase_limit:
            raise RuntimeError("수치 안정성 오류: dt를 줄여주세요.")
        elif buffer_displacement < 0:
            break

        history_list.append([
            t,
            round(global_energy / 1000, 3),
            round(kinetic_energy / 1000, 3),
            round(internal_energy / 1000, 3),
            round(buffer_displacement, 3),
            round(buffer_velocity, 3),
            round(buffer_force, 3),
            round(internal_energy / 1000, 3),
        ])
        t += dt
        car_displacement += car_velocity * dt
        car_velocity += car_acceleration * dt
        buffer_displacement = car_displacement
        buffer_velocity = car_velocity
        buffer_force = buffer_model(buffer_displacement, buffer_velocity, buffer_force, dt)
        car_acceleration = -buffer_force / car_mass
        kinetic_energy = 0.5 * car_mass * car_velocity ** 2
        internal_energy += buffer_force * buffer_velocity * dt
        global_energy = kinetic_energy + internal_energy

    if progress_callback:
        progress_callback(100)

    return pd.DataFrame(history_list, columns=[
        'Time [ms]', 'Global_Energy [kJ]', 'Kinetic_Energy [kJ]',
        'Internal_Energy [kJ]', 'Buffer_Displacement [mm]',
        'Buffer_Velocity [mm/ms]', 'Buffer_Force [kN]',
        'Buffer_Absorbed Energy [kJ]',
    ])


# ─── 연결 시뮬레이션 ─────────────────────────────────────────────────────────

def f_coupling_simulation(car_mass_list: list, coupler_model_list: list,
                           mu_kinetic_friction_list: list,
                           mu_static_friction_list: list,
                           car_velocity_list: list,
                           termination_time: float, dt: float,
                           energy_increase_limit: float,
                           car_spring_enabled: bool = False,
                           car_stiffness_list: list | None = None,
                           progress_callback=None,
                           interrupt_callback=None) -> pd.DataFrame:
    if not car_spring_enabled or not car_stiffness_list:
        # 기존 단일 질점 Coupling 경로 그대로 유지
        t = 0.0
        g = 9.81e-3
        num_cars = len(car_mass_list)
        num_couplers = len(coupler_model_list)

        car_displacement_list = [1000.0 * i for i in range(num_cars)]
        car_acceleration_list = [0.0] * num_cars
        car_kinetic_energy_list = [0.5 * m * v ** 2 for m, v in zip(car_mass_list, car_velocity_list)]
        car_friction_energy_list = [0.0] * num_cars

        coupler_displacement_list = [0.0] * num_couplers
        coupler_velocity_list = [0.0] * num_couplers
        coupler_force_list = [0.0] * num_couplers
        coupler_internal_energy_list = [0.0] * num_couplers

        initial_energy = sum(car_kinetic_energy_list)
        history_list = []
        total_steps = int(termination_time / dt)
        friction_force = 0.0

        for step in range(total_steps):
            if interrupt_callback and interrupt_callback():
                raise RuntimeError("__INTERRUPTED__")
            if progress_callback:
                progress_callback(int(step / total_steps * 100))

            kinetic_energy = sum(car_kinetic_energy_list)
            friction_energy = sum(car_friction_energy_list)
            internal_energy = sum(coupler_internal_energy_list)
            global_energy = kinetic_energy + friction_energy + internal_energy

            if (global_energy / initial_energy - 1) > energy_increase_limit:
                raise RuntimeError("수치 안정성 오류: dt를 줄여주세요.")

            current_history = [t,
                               global_energy / 1000, kinetic_energy / 1000,
                               internal_energy / 1000, friction_energy / 1000]

            for i in range(num_cars):
                travel = car_displacement_list[i] - 1000.0 * i
                current_history.extend([
                    car_displacement_list[i],
                    travel,
                    car_velocity_list[i] * 3.6,
                    car_acceleration_list[i] / g,
                    car_friction_energy_list[i] / 1000,
                ])

            for i in range(num_couplers):
                current_history.extend([
                    coupler_displacement_list[i],
                    coupler_velocity_list[i],
                    coupler_force_list[i],
                    coupler_internal_energy_list[i] / 1000,
                ])

            history_list.append(current_history)

            # 상태 업데이트
            t += dt
            for i in range(num_cars):
                car_displacement_list[i] += car_velocity_list[i] * dt
                car_velocity_list[i] += car_acceleration_list[i] * dt

            for i in range(num_couplers):
                coupler_displacement_list[i] = 1000.0 + (
                    car_displacement_list[i] - car_displacement_list[i + 1]
                )
                coupler_velocity_list[i] = car_velocity_list[i] - car_velocity_list[i + 1]
                coupler_force_list[i] = coupler_model_list[i](
                    coupler_displacement_list[i], coupler_velocity_list[i],
                    coupler_force_list[i], dt
                )
                coupler_internal_energy_list[i] += (
                    coupler_force_list[i] * coupler_velocity_list[i] * dt
                )

            for i in range(num_cars):
                if i == 0:
                    net_force = -coupler_force_list[0]
                elif i == num_cars - 1:
                    net_force = coupler_force_list[-1]
                else:
                    net_force = coupler_force_list[i - 1] - coupler_force_list[i]

                friction_force = 0.0
                if car_velocity_list[i] == 0:
                    if abs(net_force) <= mu_static_friction_list[i] * car_mass_list[i] * g:
                        net_force = 0.0
                else:
                    direction = -1 if car_velocity_list[i] > 0 else 1
                    friction_force = direction * mu_kinetic_friction_list[i] * car_mass_list[i] * g
                    if abs(car_velocity_list[i]) < mu_kinetic_friction_list[i] * g * dt:
                        car_velocity_list[i] = 0.0
                    else:
                        net_force += friction_force

                car_acceleration_list[i] = net_force / car_mass_list[i]
                car_kinetic_energy_list[i] = 0.5 * car_mass_list[i] * car_velocity_list[i] ** 2
                car_friction_energy_list[i] += abs(friction_force * car_velocity_list[i] * dt)

        if progress_callback:
            progress_callback(100)

        column_names = ['Time [ms]', 'Global_Energy [kJ]', 'Kinetic_Energy [kJ]',
                        'Internal_Energy [kJ]', 'Friction_Energy [kJ]']
        for i in range(num_cars):
            column_names += [
                f'Car_Displacement_{str(i).zfill(2)} [mm]',
                f'Car_Displacement_Difference_{str(i).zfill(2)} [mm]',
                f'Car_Velocity_{str(i).zfill(2)} [km/h]',
                f'Car_Acceleration_{str(i).zfill(2)} [g]',
                f'Car_Friction_Energy_{str(i).zfill(2)} [kJ]',
            ]
        for i in range(num_couplers):
            column_names += [
                f'Coupler_Displacement_{str(i).zfill(2)} [mm]',
                f'Coupler_Velocity_{str(i).zfill(2)} [m/s]',
                f'Coupler_Force_{str(i).zfill(2)} [kN]',
                f'Coupler_Internal_Energy_{str(i).zfill(2)} [kJ]',
            ]

        return pd.DataFrame(history_list, columns=column_names)

    # ------------------------------------------------------------------
    # 차량 강성(내부 스프링) 경로: 각 차량을 두 반질량 + 내부 스프링으로
    # 표현하고, 변위/속도/가속도는 두 질점 평균으로 리포팅한다.
    # ------------------------------------------------------------------
    t = 0.0
    g = 9.81e-3
    num_cars = len(car_mass_list)
    num_couplers = len(coupler_model_list)

    # 기존 구조 유지: 위치와 속도는 차량 기준에서 계산하되,
    # 내부 스프링은 두 반질량 포인트 좌/우 상태와 평균값으로 보고한다.
    car_displacement_list = [1000.0 * i for i in range(num_cars)]
    car_acceleration_list = [0.0] * num_cars
    car_kinetic_energy_list = [0.5 * m * v ** 2 for m, v in zip(car_mass_list, car_velocity_list)]
    car_friction_energy_list = [0.0] * num_cars

    # 내부 스프링 상태: 좌/우 질량 포인트의 위치와 속도
    left_disp = [1000.0 * i for i in range(num_cars)]
    right_disp = [1000.0 * i for i in range(num_cars)]
    left_vel = [v for v in car_velocity_list]
    right_vel = [v for v in car_velocity_list]
    left_acc = [0.0] * num_cars
    right_acc = [0.0] * num_cars
    spring_force = [0.0] * num_cars

    coupler_displacement_list = [0.0] * num_couplers
    coupler_velocity_list = [0.0] * num_couplers
    coupler_force_list = [0.0] * num_couplers
    coupler_internal_energy_list = [0.0] * num_couplers

    initial_energy = sum(car_kinetic_energy_list)
    history_list = []
    total_steps = int(termination_time / dt)
    friction_force = 0.0

    for step in range(total_steps):
        if interrupt_callback and interrupt_callback():
            raise RuntimeError("__INTERRUPTED__")
        if progress_callback:
            progress_callback(int(step / total_steps * 100))

        kinetic_energy = sum(car_kinetic_energy_list)
        friction_energy = sum(car_friction_energy_list)
        internal_energy = sum(coupler_internal_energy_list)
        global_energy = kinetic_energy + friction_energy + internal_energy

        if (global_energy / initial_energy - 1) > energy_increase_limit:
            raise RuntimeError("수치 안정성 오류: dt를 줄여주세요.")

        current_history = [t,
                           global_energy / 1000, kinetic_energy / 1000,
                           internal_energy / 1000, friction_energy / 1000]

        # 기존 컬럼 유지. 각 차량의 평균 변위/속도/가속도 계산
        for i in range(num_cars):
            avg_disp = (left_disp[i] + right_disp[i]) / 2.0
            avg_vel = (left_vel[i] + right_vel[i]) / 2.0
            avg_acc = (left_acc[i] + right_acc[i]) / 2.0
            travel = avg_disp - 1000.0 * i
            current_history.extend([
                avg_disp,
                travel,
                avg_vel * 3.6,
                avg_acc / g,
                car_friction_energy_list[i] / 1000,
            ])

        for i in range(num_couplers):
            current_history.extend([
                coupler_displacement_list[i],
                coupler_velocity_list[i],
                coupler_force_list[i],
                coupler_internal_energy_list[i] / 1000,
            ])

        history_list.append(current_history)

        # 상태 업데이트: 기존 연결 상태 + 내부 스프링 반작용
        t += dt
        for i in range(num_cars):
            # 단순히 스프링을 추가한 평균 모델: 좌/우 질량 점의 속도/변위를 동시에 갱신
            left_disp[i] += left_vel[i] * dt
            right_disp[i] += right_vel[i] * dt
            left_vel[i] += left_acc[i] * dt
            right_vel[i] += right_acc[i] * dt
            car_displacement_list[i] = (left_disp[i] + right_disp[i]) / 2.0

        for i in range(num_couplers):
            coupler_displacement_list[i] = 1000.0 + (
                car_displacement_list[i] - car_displacement_list[i + 1]
            )
            coupler_velocity_list[i] = car_velocity_list[i] - car_velocity_list[i + 1]
            coupler_force_list[i] = coupler_model_list[i](
                coupler_displacement_list[i], coupler_velocity_list[i],
                coupler_force_list[i], dt
            )
            coupler_internal_energy_list[i] += (
                coupler_force_list[i] * coupler_velocity_list[i] * dt
            )

        for i in range(num_cars):
            if i == 0:
                net_force = -coupler_force_list[0]
            elif i == num_cars - 1:
                net_force = coupler_force_list[-1]
            else:
                net_force = coupler_force_list[i - 1] - coupler_force_list[i]

            friction_force = 0.0
            if car_velocity_list[i] == 0:
                if abs(net_force) <= mu_static_friction_list[i] * car_mass_list[i] * g:
                    net_force = 0.0
            else:
                direction = -1 if car_velocity_list[i] > 0 else 1
                friction_force = direction * mu_kinetic_friction_list[i] * car_mass_list[i] * g
                if abs(car_velocity_list[i]) < mu_kinetic_friction_list[i] * g * dt:
                    car_velocity_list[i] = 0.0
                else:
                    net_force += friction_force

            # 내부 스프링 반작용을 각 차량에 반대 방향으로 나누어 적용
            spring_k = car_stiffness_list[i] if i < len(car_stiffness_list) else 10.0
            spring_force[i] = spring_k * (right_disp[i] - left_disp[i])
            net_force_left = net_force / 2.0 - spring_force[i]
            net_force_right = net_force / 2.0 + spring_force[i]

            half_mass = car_mass_list[i] / 2.0
            left_acc[i] = net_force_left / half_mass
            right_acc[i] = net_force_right / half_mass

            # 좌/우 질량점 속도는 평균화된 차량 속도와 내부 스프링 분산을 반영
            left_vel[i] += left_acc[i] * dt
            right_vel[i] += right_acc[i] * dt
            car_velocity_list[i] = (left_vel[i] + right_vel[i]) / 2.0

            car_acceleration_list[i] = (left_acc[i] + right_acc[i]) / 2.0
            car_kinetic_energy_list[i] = 0.5 * car_mass_list[i] * car_velocity_list[i] ** 2
            car_friction_energy_list[i] += abs(friction_force * car_velocity_list[i] * dt)

    if progress_callback:
        progress_callback(100)

    column_names = ['Time [ms]', 'Global_Energy [kJ]', 'Kinetic_Energy [kJ]',
                    'Internal_Energy [kJ]', 'Friction_Energy [kJ]']
    for i in range(num_cars):
        column_names += [
            f'Car_Displacement_{str(i).zfill(2)} [mm]',
            f'Car_Displacement_Difference_{str(i).zfill(2)} [mm]',
            f'Car_Velocity_{str(i).zfill(2)} [km/h]',
            f'Car_Acceleration_{str(i).zfill(2)} [g]',
            f'Car_Friction_Energy_{str(i).zfill(2)} [kJ]',
        ]
    for i in range(num_couplers):
        column_names += [
            f'Coupler_Displacement_{str(i).zfill(2)} [mm]',
            f'Coupler_Velocity_{str(i).zfill(2)} [m/s]',
            f'Coupler_Force_{str(i).zfill(2)} [kN]',
            f'Coupler_Internal_Energy_{str(i).zfill(2)} [kJ]',
        ]

    # 내부 스프링 경로는 기존 coupler/car 컬럼 스키마를 그대로 유지한다.
    # 새로 선언한 평균 리포트용 컬럼을 실제 history row에 넣지 않으므로,
    # 여기서는 기존 컬럼 정의만 사용해 DataFrame을 생성한다.
    return pd.DataFrame(history_list, columns=column_names)


# ─── 결과 시각화 ─────────────────────────────────────────────────────────────

def f_rigidwall_data_curve_analysis(simulation_df: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(simulation_df['Time [ms]'], simulation_df['Global_Energy [kJ]'], label='Global')
    ax1.plot(simulation_df['Time [ms]'], simulation_df['Kinetic_Energy [kJ]'], label='Kinetic')
    ax1.plot(simulation_df['Time [ms]'], simulation_df['Internal_Energy [kJ]'], label='Internal')
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Energy [kJ]')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(simulation_df['Buffer_Displacement [mm]'], simulation_df['Buffer_Force [kN]'])
    ax2.set_xlabel('Buffer_Displacement [mm]')
    ax2.set_ylabel('Buffer_Force [kN]')
    ax2.grid(True)
    plt.tight_layout()
    return fig


def f_rigidwall_load_test_result(loading_address: str) -> pd.DataFrame:
    path = loading_address if loading_address.endswith('.csv') else loading_address + '.csv'
    return pd.read_csv(path)


def f_test_simulation_numerical_validation(test_df: pd.DataFrame,
                                            simulation_df: pd.DataFrame) -> dict:
    def _mape(A, F):
        A, F = np.array(A), np.array(F)
        return np.mean(np.abs((A - F) / (A + 1e-16))) * 100

    def _smape(A, F):
        A, F = np.array(A), np.array(F)
        return np.mean(2 * np.abs(F - A) / (np.abs(A) + np.abs(F) + 1e-16)) * 100

    def _rmse(A, F):
        return float(np.sqrt(mean_squared_error(A, F)))

    max_disp_test = test_df['Displacement [mm]'].max()
    max_disp_sim = simulation_df['Buffer_Displacement [mm]'].max()
    max_force_test = test_df['Force [kN]'].max()
    max_force_sim = simulation_df['Buffer_Force [kN]'].max()

    interval = 1.0
    max_range = min(max_disp_test, max_disp_sim)
    test_forces, sim_forces = [], []
    for val in np.arange(0, max_range, interval):
        test_forces.append(test_df[test_df['Displacement [mm]'] > val].iloc[0]['Force [kN]'])
        sim_forces.append(
            simulation_df[simulation_df['Buffer_Displacement [mm]'] > val].iloc[0]['Buffer_Force [kN]']
        )

    return {
        'max_disp_test': round(max_disp_test, 1),
        'max_disp_sim': round(max_disp_sim, 1),
        'max_force_test': round(max_force_test, 1),
        'max_force_sim': round(max_force_sim, 1),
        'mape': round(_mape(test_forces, sim_forces), 2),
        'smape': round(_smape(test_forces, sim_forces), 2),
        'rmse': round(_rmse(test_forces, sim_forces), 2),
    }
