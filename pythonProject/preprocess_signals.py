import numpy as np
import pandas as pd
from scipy.signal import medfilt


def preprocess_signals(fhr_time, fhr_values, uc_time, uc_values, fs=4):
    """
    Предобработка сигналов FHR и UC.

    Параметры:
    ----------
    fhr_time, fhr_values : list
        Время и значения FHR (ЧСС плода).
    uc_time, uc_values : list
        Время и значения UC (схватки).
    fs : int
        Целевая частота дискретизации (Гц). Обычно FHR ≈ 4 Гц.

    Возвращает:
    -----------
    pd.DataFrame со столбцами ["time", "fhr", "uc"]
    """

    # 1. Преобразуем в датафреймы
    fhr_df = pd.DataFrame({"time": fhr_time, "fhr": fhr_values})
    uc_df = pd.DataFrame({"time": uc_time, "uc": uc_values})

    # 2. Создаём единый временной ряд (целевую сетку времени)
    t_min = max(fhr_df["time"].min(), uc_df["time"].min())
    t_max = min(fhr_df["time"].max(), uc_df["time"].max())
    time_grid = np.arange(t_min, t_max, 1 / fs)

    # 3. Интерполяция на общую сетку
    fhr_interp = np.interp(time_grid, fhr_df["time"], fhr_df["fhr"])
    uc_interp = np.interp(time_grid, uc_df["time"], uc_df["uc"])

    # 4. Фильтрация FHR (медианный фильтр для удаления спайков)
    fhr_filtered = medfilt(fhr_interp, kernel_size=5)

    # 5. Убираем нереальные значения
    fhr_filtered = np.where((fhr_filtered < 50) | (fhr_filtered > 210), np.nan, fhr_filtered)
    uc_interp = np.where((uc_interp < 0), 0, uc_interp)

    # 6. Заполняем пропуски (скользящее среднее)
    fhr_clean = pd.Series(fhr_filtered).interpolate().fillna(method="bfill").fillna(method="ffill")
    uc_clean = pd.Series(uc_interp).interpolate().fillna(0)

    # 7. Возвращаем датафрейм
    return pd.DataFrame({"time": time_grid, "fhr": fhr_clean, "uc": uc_clean})
