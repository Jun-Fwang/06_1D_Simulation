"""
core/data_sampling.py
B_Data_Sampling_by_KNN_Clustering.ipynb 의 모든 함수 추출
"""
import math
import random as rn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

LABEL_SIZE = 20
TICK_SIZE = 10
TITLE_SIZE = 15


def f_load_df(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def sampling_and_polynomial_fitting(data_frame: pd.DataFrame,
                                    interval: float,
                                    polynomial_order: int) -> pd.DataFrame:
    base_col = 'Displacement [mm]'
    target_col = 'Force [kN]'
    min_val = data_frame[base_col].min()
    max_val = data_frame[base_col].max()
    sampling_points = np.arange(min_val, max_val, interval)

    sampled = []
    for sp in sampling_points:
        subset = data_frame[
            (data_frame[base_col] >= sp) &
            (data_frame[base_col] < sp + interval)
        ]
        if len(subset) > 0:
            sampled.append(subset)

    if not sampled:
        return data_frame

    sampled_df = pd.concat(sampled).drop_duplicates().reset_index(drop=True)

    x = sampled_df[base_col].values.reshape(-1, 1)
    y = sampled_df[target_col].values

    poly = PolynomialFeatures(degree=polynomial_order, include_bias=False)
    x_poly = poly.fit_transform(x)
    model = LinearRegression(fit_intercept=False)
    model.fit(x_poly, y)

    sampled_df['Fitted_Force [kN]'] = model.predict(x_poly)

    vel_col = 'Velocity [mm/ms]' if 'Velocity [mm/ms]' in sampled_df.columns else None
    keep_cols = [base_col, target_col, 'Fitted_Force [kN]']
    if vel_col:
        keep_cols.insert(1, vel_col)
    return sampled_df[keep_cols]


def f_combine_df(file_path_list: list, interval: float,
                 polynomial_order: int) -> pd.DataFrame:
    df_dic = {}
    for idx, df_path in enumerate(file_path_list):
        df_dic[idx] = f_load_df(df_path)
        if polynomial_order >= 1:
            df_dic[idx] = sampling_and_polynomial_fitting(
                df_dic[idx], interval, polynomial_order
            )
    combined = pd.concat(df_dic.values()).reset_index(drop=True)
    return combined


def f_sampling_df(combined_df: pd.DataFrame, n_clusters: int,
                  show: bool = True, grid_size: int = 10):
    Displacement = combined_df['Displacement [mm]'].to_numpy()
    Velocity = combined_df['Velocity [mm/ms]'].to_numpy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(np.column_stack((Displacement, Velocity)))

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    clusters = kmeans.fit_predict(scaled_data)

    cluster_counts = np.bincount(clusters)
    min_cluster_size = cluster_counts.min()

    x_min = max(math.floor(combined_df['Displacement [mm]'].min()), 0)
    x_max = math.ceil(combined_df['Displacement [mm]'].max())
    y_min = max(math.floor(combined_df['Velocity [mm/ms]'].min()), 0)
    y_max = math.ceil(combined_df['Velocity [mm/ms]'].max())

    h = 0.01
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max + h, h),
        np.arange(y_min, y_max + h, h)
    )
    grid_data = np.c_[xx.ravel(), yy.ravel()]
    scaled_grid = scaler.transform(grid_data)
    Z = kmeans.predict(scaled_grid).reshape(xx.shape)

    indices_to_remove = []
    for cid in np.unique(clusters):
        idx_in = np.where(clusters == cid)[0]
        np.random.shuffle(idx_in)
        indices_to_remove.extend(idx_in[min_cluster_size:])

    sampling_df = combined_df.drop(indices_to_remove).reset_index(drop=True)

    fig = None
    if show:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].scatter(combined_df['Displacement [mm]'], combined_df['Velocity [mm/ms]'],
                       s=5, color='black', label='Original Data')
        axs[0].set_title('Original Data Points', fontsize=TITLE_SIZE)
        axs[0].set_xlabel('Displacement [mm]', fontsize=LABEL_SIZE)
        axs[0].set_ylabel('Velocity [mm/ms]', fontsize=LABEL_SIZE)
        axs[0].set_xlim([x_min, x_max])
        axs[0].set_ylim([y_min, y_max])
        axs[0].grid()
        axs[0].legend()

        axs[1].contourf(xx, yy, Z, cmap='viridis')
        axs[1].scatter(Displacement, Velocity, s=5, color='black')
        centers = scaler.inverse_transform(kmeans.cluster_centers_)
        axs[1].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=50, label='Centroids')
        axs[1].set_title('K-means Clustering', fontsize=TITLE_SIZE)
        axs[1].set_xlabel('Displacement [mm]', fontsize=LABEL_SIZE)
        axs[1].set_ylabel('Velocity [mm/ms]', fontsize=LABEL_SIZE)
        axs[1].set_xlim([x_min, x_max])
        axs[1].set_ylim([y_min, y_max])
        axs[1].legend(loc='upper right')
        plt.tight_layout()

    return sampling_df, fig


def f_save_df(df: pd.DataFrame, save_address: str):
    df.to_csv(save_address, index=False)
