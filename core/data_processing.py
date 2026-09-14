"""
core/data_processing.py
A_Data_Auto_Process.ipynb 의 모든 함수 추출
"""
import struct
import os
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

LABEL_SIZE = 20
TICK_SIZE = 15


def path_converter(path: str) -> str:
    return path.replace('\\', '/')


# ─── DAT / BIN 읽기 ────────────────────────────────────────────────────────────

def read_dat_header(dat_filepath: str) -> str:
    with open(dat_filepath, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def extract_channel_info(header_content: str) -> list:
    channels = []
    lines = header_content.split('\n')
    inside_channel = False
    channel_info = {}
    for line in lines:
        if line.startswith('#BEGINCHANNELHEADER'):
            inside_channel = True
            continue
        if line.startswith('#ENDCHANNELHEADER'):
            inside_channel = False
            channels.append(channel_info)
            channel_info = {}
            continue
        if inside_channel:
            key, value = line.split(',', 1)
            channel_info[int(key)] = value.strip()
    return channels


def generate_time_data(channel_info: dict) -> list:
    count = int(channel_info[220])
    offset = float(channel_info[240])
    delta = float(channel_info[241])
    return [offset + i * delta for i in range(count)]


def convert_bin_to_actual(channel_info: list, directory_path: str) -> dict:
    BIN_path, BIN_file, BIN_INT16, convert_BIN = {}, {}, {}, {}
    for i in range(1, len(channel_info)):
        BIN_path[i] = directory_path + '//' + channel_info[i][211]
        with open(BIN_path[i], 'rb') as f:
            BIN_file[i] = f.read()
            BIN_INT16[i] = struct.unpack(f"<{len(BIN_file[i]) // 2}h", BIN_file[i])
        convert_BIN[i] = [
            (v * float(channel_info[i][241])) + float(channel_info[i][240])
            for v in BIN_INT16[i][len(BIN_INT16[i]) - int(channel_info[i][220]):]
        ]
    return convert_BIN


def merge_data(convert_data: dict, time_data: list) -> pd.DataFrame:
    if len(convert_data) == 4:
        d = {
            "Time axis [s]": time_data,
            'force_1 [kN]': convert_data[1],
            "force_2 [kN]": convert_data[2],
            "force_3 [kN]": convert_data[3],
            "force_4 [kN]": convert_data[4],
        }
    else:
        d = {
            "Time axis [s]": time_data,
            'force_1 [kN]': convert_data[1],
            "force_2 [kN]": convert_data[2],
            "force_3 [kN]": convert_data[3],
            "force_4 [kN]": convert_data[4],
            "Laser disp [mm]": convert_data[5],
        }
    return pd.DataFrame(d)


def process_all_paths_and_store(dat_list: list) -> dict:
    results = {}
    for idx, path_dat in enumerate(dat_list):
        channel_info = extract_channel_info(read_dat_header(path_dat))
        time_data = generate_time_data(channel_info[0])
        bin_data = convert_bin_to_actual(channel_info, os.path.dirname(path_dat))
        results[idx] = merge_data(bin_data, time_data)
    return results


# ─── 필터링 ─────────────────────────────────────────────────────────────────────

def butterworth_lowpass_filter(data_dict: dict, cutoff_frequency: float,
                               force_zero_phase_condition: bool = True) -> dict:
    def _apply(time_data, signal_data, fc):
        fs = 1 / (time_data[1] - time_data[0])
        b, a = butter(4, fc / (0.5 * fs), btype='low', analog=False)
        return filtfilt(b, a, signal_data) if force_zero_phase_condition else lfilter(b, a, signal_data)

    for key in data_dict:
        df = data_dict[key]
        time_data = df['Time axis [s]'].values
        force_cols = ['force_1 [kN]', 'force_2 [kN]', 'force_3 [kN]', 'force_4 [kN]']
        filt_cols = ['FilteredS_force1 [kN]', 'Filtered_force2 [kN]',
                     'Filtered_force3 [kN]', 'Filtered_force4 [kN]']
        for fc, flt in zip(force_cols, filt_cols):
            if cutoff_frequency == 0:
                df[flt] = df[fc].values
            else:
                df[flt] = _apply(time_data, df[fc].values, cutoff_frequency)
        df['Force [kN]'] = df[filt_cols].sum(axis=1)
        data_dict[key] = df
    return data_dict


def filtered_disp_data(df: pd.DataFrame, filter: str = 'original',
                       filter_parameter: int = 150) -> pd.DataFrame:
    if filter == 'original':
        df['filtered_Laser_disp [mm]'] = df['Laser disp [mm]'].shift(-17)
    elif filter == 'moving_range_mean':
        df['filtered_Laser_disp [mm]'] = (
            df['Laser disp [mm]'].rolling(window=filter_parameter, center=True).mean().shift(-17)
        )
    elif filter == 'moving_range_median':
        df['filtered_Laser_disp [mm]'] = (
            df['Laser disp [mm]'].rolling(window=filter_parameter, center=True).median().shift(-17)
        )
    elif filter == 'gaussian_smoothing':
        df['filtered_Laser_disp [mm]'] = gaussian_filter(df['Laser disp [mm]'], sigma=filter_parameter)
        df['filtered_Laser_disp [mm]'] = df['filtered_Laser_disp [mm]'].shift(-17)
    return df


# ─── 변위 / 하중 동기화 ────────────────────────────────────────────────────────

def disPLoader(path: str) -> pd.DataFrame:
    for enc in ('cp949', 'utf-8'):
        try:
            df = pd.read_csv(path, sep='\t', header=0, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("파일의 인코딩을 확인할 수 없습니다.")
    df = df.iloc[2:, :]
    df.replace('X', np.nan, inplace=True)
    df = df.astype(float)
    return df


def findSumforce(df: pd.DataFrame, diff_range: float = 0,
                 operation_mode: bool = True) -> pd.DataFrame:
    if operation_mode:
        idnex_0 = df.loc[df['Time axis [s]'] == -0.05].index[0]
        data_after_0 = df.loc[idnex_0:]
        max_idx = data_after_0['Force [kN]'].idxmax()
        for idx in range(max_idx, -1, -1):
            if df['Force [kN]'].iloc[idx] <= 0:
                first_negative_idx = idx
                break
        for idx in range(first_negative_idx, len(df)):
            if abs(df['Force [kN]'].iloc[idx] - df['Force [kN]'].iloc[idx - 1]) >= diff_range:
                end_decrease = idx
                break
        result = data_after_0.loc[end_decrease:]
    else:
        idnex_0 = df.loc[df['Time axis [s]'] == 0].index[0]
        result = df.loc[idnex_0:]
    result = result.copy()
    result.loc[:, 'Time axis [s]'] = result['Time axis [s]'] - result['Time axis [s]'].values[0]
    return result.reset_index(drop=True)


def findCrashtime(df: pd.DataFrame) -> pd.DataFrame:
    first_negative_force_idx = df['Force [kN]'][10:].lt(0).idxmax()
    df1 = df.loc[first_negative_force_idx:]
    rounded_up = math.ceil(df1['Time axis [s]'].iloc[0] * 1000) / 1000
    start_index = df1.loc[round(df1['Time axis [s]'], 4) == rounded_up].index[0]
    return df.loc[:start_index]


def disp_processing(disp: pd.DataFrame, start_point: float,
                    finish_point: float, max_disp_mode: bool) -> pd.DataFrame:
    disp_new = disp[['Time', 'Default/Point#1']].copy()
    index_value = disp_new.loc[disp_new['Time'] == start_point].index[0]
    disp_new2 = disp_new.loc[index_value:].copy()
    if disp_new2['Time'].values[0] < 1:
        disp_new2.loc[:, 'Time'] -= disp_new2['Time'].values[0]
        disp_new2.loc[:, 'Default/Point#1'] -= disp_new2['Default/Point#1'].values[0]
    else:
        disp_new2.loc[:, 'Time'] = (disp_new2['Time'] - disp_new2['Time'].values[0]) / 1000
        disp_new2.loc[:, 'Default/Point#1'] = (
            disp_new2['Default/Point#1'] - disp_new2['Default/Point#1'].values[0]
        ) / 1000
    disp_new2['Time'] = disp_new2['Time'].round(3)
    if not max_disp_mode:
        finish_index = disp_new2.loc[disp_new2['Time'] == round(finish_point, 4)].index[0]
    else:
        finish_index = disp_new2.loc[
            disp_new2['Default/Point#1'] == disp_new2['Default/Point#1'].max()
        ].index[0]
    return disp_new2.loc[:finish_index].reset_index(drop=True)


def polynomialData(data: pd.DataFrame, order: int, counts: int) -> pd.DataFrame:
    x = data.iloc[:, 0].values.reshape(-1, 1)
    y = data.iloc[:, 1].values
    poly = PolynomialFeatures(degree=order, include_bias=False)
    x_poly = poly.fit_transform(x)
    model = LinearRegression(fit_intercept=False)
    model.fit(x_poly, y)
    x_new = np.linspace(x[0], x[-1], counts).reshape(-1, 1)
    y_new = model.predict(poly.transform(x_new))
    return pd.DataFrame({'X': x_new.ravel(), 'Displacement [mm]': y_new})


def integrate_energy(df: pd.DataFrame) -> pd.DataFrame:
    delta_disp = np.diff(df['Displacement [mm]'].values)
    delta_disp = np.insert(delta_disp, 0, 0)
    df_copy = df.copy()
    df_copy['Energy [J]'] = np.cumsum(delta_disp * df['Force [kN]'])
    df_copy['Energy [kJ]'] = df_copy['Energy [J]'] / 1000
    max_index = df_copy['Energy [kJ]'].idxmax()
    df_copy.loc[max_index + 1:, 'Energy [kJ]'] = np.nan
    return df_copy


def displayResult(force_dic: dict, disp_list: list, diff_list: list,
                  step_list: list, max_disp_mode: bool = True) -> dict:
    disp_dic, sum_force_df_dic, force_processing_dic = {}, {}, {}
    processing_disp_dic, poly_disp_dic, energy_dic = {}, {}, {}
    for i in range(len(disp_list)):
        disp_dic[i] = disPLoader(disp_list[i])
        sum_force_df_dic[i] = findSumforce(force_dic[i], diff_list[i])
        force_processing_dic[i] = findCrashtime(sum_force_df_dic[i])
        processing_disp_dic[i] = disp_processing(
            disp_dic[i], step_list[i],
            force_processing_dic[i]['Time axis [s]'].values[-1], max_disp_mode
        )
        force_dt = force_processing_dic[i].values[1][0]
        disp_dt = processing_disp_dic[i].values[1][0]
        increase_counts = int(disp_dt / force_dt)
        if not max_disp_mode:
            poly_disp_dic[i] = polynomialData(processing_disp_dic[i], 8, len(force_processing_dic[i]))
        else:
            poly_disp_dic[i] = polynomialData(
                processing_disp_dic[i], 8, len(processing_disp_dic[i]) * increase_counts
            )
        force_processing_dic[i]['Displacement [mm]'] = poly_disp_dic[i]['Displacement [mm]']
        energy_dic[i] = integrate_energy(force_processing_dic[i])
    return energy_dic


def makeDescribe(speeds: list, mass: float, df_dict: dict,
                 step_list: list, diff_list: list, cut_off_frequency: float):
    final_result, final_result_round = [], []

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # D-F 그래프
    ax1 = axes[0]
    ax1.set_xlabel('Displacement [mm]', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Reaction Force [kN]', fontsize=LABEL_SIZE)
    ax1.tick_params(axis='both', labelsize=TICK_SIZE)
    ax1.grid()
    ax1.set_title(f'Cutoff Frequency: {cut_off_frequency} Hz')

    # D-E 그래프
    ax2 = axes[1]
    ax2.set_xlabel('Displacement [mm]', fontsize=LABEL_SIZE)
    ax2.set_ylabel('Absorbed Energy [kJ]', fontsize=LABEL_SIZE)
    ax2.grid()

    for speed, (key, df) in zip(speeds, df_dict.items()):
        kinetic_E = (mass * 0.5 * (speed / 3.6) ** 2) / 1000
        peak_force = df['Force [kN]'].max()
        max_disp = df['Displacement [mm]'].max()
        mean_force = df['Energy [kJ]'].max() / max_disp * 1000
        absorbed_E = df['Energy [kJ]'].max() / kinetic_E * 100
        desc = {
            'Speed [km/h]': speed,
            'Peak Force [kN]': peak_force,
            'Maximum Displacement [mm]': max_disp,
            'Mean Force [kN]': mean_force,
            'Absorbed Energy [kJ]': df['Energy [kJ]'].max(),
            'Absorbed ratio': absorbed_E,
        }
        final_result.append(pd.DataFrame([desc]))
        final_result_round.append(pd.DataFrame([desc]).round(1))
        ax1.plot(df['Displacement [mm]'], df['Force [kN]'], label=f'{speed} km/h')
        df_E = df.loc[:df.loc[df['Energy [kJ]'] == df['Energy [kJ]'].max()].index[0]]
        ax2.plot(df_E['Displacement [mm]'], df_E['Energy [kJ]'], label=f'{speed} km/h')

    for ax in (ax1, ax2):
        handles, labels = ax.get_legend_handles_labels()
        sorted_hl = sorted(zip(handles, labels), key=lambda x: float(x[1].split()[0]))
        sh, sl = zip(*sorted_hl)
        ax.legend(sh, sl, loc='upper right')

    plt.tight_layout()

    result_df = pd.concat(final_result).reset_index(drop=True)
    result_df['disp_start_point'] = step_list
    result_df['Diff_range'] = diff_list
    result_df = result_df.sort_values('Speed [km/h]').reset_index(drop=True)

    result_round_df = pd.concat(final_result_round).reset_index(drop=True)
    result_round_df['disp_start_point'] = step_list
    result_round_df['Diff_range'] = diff_list
    result_round_df = result_round_df.sort_values('Speed [km/h]').reset_index(drop=True)

    return result_df, result_round_df, fig
