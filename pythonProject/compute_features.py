# import numpy as np
# import pandas as pd
#
#
# def compute_features(fhr_df, uc_df):
#     """
#     fhr_df: DataFrame с колонками [time_sec, value]
#     uc_df: DataFrame с колонками [time_sec, value]
#     """
#     feats = {}
#
#     # FHR признаки
#     fhr = fhr_df['value'].dropna().values
#     if len(fhr) > 0:
#         feats['mean_fhr'] = np.mean(fhr)
#         feats['median_fhr'] = np.median(fhr)
#         feats['std_fhr'] = np.std(fhr)
#         feats['min_fhr'] = np.min(fhr)
#         feats['max_fhr'] = np.max(fhr)
#         feats['range_fhr'] = np.max(fhr) - np.min(fhr)
#         feats['baseline_fhr'] = np.percentile(fhr, 10)
#         feats['short_term_var'] = np.mean(np.abs(np.diff(fhr)))  # вариабельность на малых интервалах
#         rolling_mean = pd.Series(fhr).rolling(window=30, min_periods=1).mean()
#         feats['long_term_var'] = np.std(rolling_mean)
#
#         # акцелерации и децелерации
#         baseline = feats['baseline_fhr']
#         feats['accelerations'] = np.sum(fhr > baseline + 15)
#         feats['decelerations'] = np.sum(fhr < baseline - 15)
#     else:
#         for k in ["mean_fhr", "median_fhr", "std_fhr", "min_fhr", "max_fhr", "range_fhr",
#                   "baseline_fhr", "short_term_var", "long_term_var", "accelerations", "decelerations"]:
#             feats[k] = np.nan
#
#     # UC признаки
#     uc = uc_df['value'].dropna().values
#     if len(uc) > 0:
#         feats['mean_uc'] = np.mean(uc)
#         feats['std_uc'] = np.std(uc)
#         feats['max_uc'] = np.max(uc)
#         feats['uc_missing'] = 0
#     else:
#         feats['mean_uc'] = 0
#         feats['std_uc'] = 0
#         feats['max_uc'] = 0
#         feats['uc_missing'] = 1
#
#     # Кросс-корреляция FHR и UC
#     if len(fhr) > 0 and len(uc) > 0:
#         min_len = min(len(fhr), len(uc))
#         feats['cross_corr_fhr_uc'] = np.corrcoef(fhr[:min_len], uc[:min_len])[0, 1]
#     else:
#         feats['cross_corr_fhr_uc'] = 0
#
#     return feats

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> dict:
    """
    Вычисление признаков на основе предобработанного датафрейма.

    Параметры:
    ----------
    df : pd.DataFrame
        DataFrame со столбцами ["time", "fhr", "uc"], полученный из preprocess_signals.

    Возвращает:
    -----------
    dict : словарь с признаками
    """
    feats = {}

    # FHR признаки
    fhr = df['fhr'].dropna().values
    if len(fhr) > 0:
        feats['mean_fhr'] = float(np.mean(fhr))
        feats['median_fhr'] = float(np.median(fhr))
        feats['std_fhr'] = float(np.std(fhr))
        feats['min_fhr'] = float(np.min(fhr))
        feats['max_fhr'] = float(np.max(fhr))
        feats['range_fhr'] = float(np.max(fhr) - np.min(fhr))
        feats['baseline_fhr'] = float(np.percentile(fhr, 10))

        # вариабельность
        feats['short_term_var'] = float(np.mean(np.abs(np.diff(fhr))))
        rolling_mean = pd.Series(fhr).rolling(window=30, min_periods=1).mean()
        feats['long_term_var'] = float(np.std(rolling_mean))

        # акцелерации и децелерации
        baseline = feats['baseline_fhr']
        feats['accelerations'] = int(np.sum(fhr > baseline + 15))
        feats['decelerations'] = int(np.sum(fhr < baseline - 15))
    else:
        for k in [
            "mean_fhr", "median_fhr", "std_fhr", "min_fhr", "max_fhr",
            "range_fhr", "baseline_fhr", "short_term_var", "long_term_var",
            "accelerations", "decelerations"
        ]:
            feats[k] = np.nan

    # UC признаки
    uc = df['uc'].dropna().values
    if len(uc) > 0:
        feats['mean_uc'] = float(np.mean(uc))
        feats['std_uc'] = float(np.std(uc))
        feats['max_uc'] = float(np.max(uc))
        feats['uc_missing'] = 0
    else:
        feats['mean_uc'] = 0.0
        feats['std_uc'] = 0.0
        feats['max_uc'] = 0.0
        feats['uc_missing'] = 1

    # Кросс-корреляция FHR и UC
    if len(fhr) > 0 and len(uc) > 0:
        min_len = min(len(fhr), len(uc))
        feats['cross_corr_fhr_uc'] = float(np.corrcoef(fhr[:min_len], uc[:min_len])[0, 1])
    else:
        feats['cross_corr_fhr_uc'] = 0.0

    return feats

