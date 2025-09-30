from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
from typing import List
from compute_features import compute_features
from preprocess_signals import preprocess_signals
from features import USER_DESCRIPTIONS, FEATURE_DESCRIPTIONS

model = joblib.load("model.pkl")
explainer = shap.Explainer(model)

app = FastAPI()


class InputData(BaseModel):
    fhr_time: List[float]
    fhr_values: List[float]
    uc_time: List[float]
    uc_values: List[float]


@app.post("/features")
def get_features(data: InputData):
    clean_data = preprocess_signals(data.fhr_time, data.fhr_values,
                                    data.uc_time, data.uc_values)

    feats = compute_features(clean_data)

    return {
        "features": [
            feats['mean_fhr'],
            feats['baseline_fhr'],
            feats["min_fhr"],
            feats["max_fhr"],
            feats["accelerations"],
            feats["decelerations"],
            feats["mean_uc"],
            feats["max_uc"]
        ],
        "descriptions": USER_DESCRIPTIONS
    }


# @app.post("/predict")
# def predict(data: InputData):
#     fhr_df = pd.DataFrame({"time_sec": data.fhr_time, "value": data.fhr_values})
#     uc_df = pd.DataFrame({"time_sec": data.uc_time, "value": data.uc_values})
#
#     feats = compute_features(fhr_df, uc_df)
#
#     X = pd.DataFrame([feats])
#     proba = model.predict_proba(X)[0, 1]
#     pred = int(model.predict(X)[0])
#
#     return {
#         "prediction": pred,
#         "probability": float(proba),
#         "features": feats,
#         "descriptions": FEATURE_DESCRIPTIONS,
#     }

FEATURE_NAMES = [
    "mean_fhr", "median_fhr", "std_fhr", "min_fhr", "max_fhr", "range_fhr",
    "baseline_fhr", "short_term_var", "long_term_var", "accelerations", "decelerations",
    "mean_uc", "std_uc", "max_uc", "cross_corr_fhr_uc", "uc_missing"
]


def _safe_shap_1d(explainer, X):
    """
    Универсально получить shap-значения для одного примера в виде 1D numpy array длины = n_features.
    """
    sv = explainer.shap_values(X)

    if isinstance(sv, list):
        arr = np.array(sv[1]) if len(sv) > 1 else np.array(sv[0])
    else:
        arr = np.array(sv)

    if arr.ndim == 3:
        if arr.shape[-1] == len(FEATURE_NAMES):
            if arr.shape[0] >= 2:
                arr = arr[1]
            else:
                arr = arr[0]
        else:
            arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])[0]

    if arr.ndim == 2:
        arr = arr[0]

    arr = np.asarray(arr).astype(float)
    return arr


@app.post("/predict")
def predict(data: InputData):
    # 1) предобработка
    try:
        df = preprocess_signals(data.fhr_time, data.fhr_values, data.uc_time, data.uc_values,
                                fs=getattr(data, "fs", 4.0))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    # 2) вычисление всех признаков (dict из 16 фич)
    feats = compute_features(df)

    # 3) формируем DataFrame в точном порядке, ожидаемом моделью
    X = pd.DataFrame([[feats.get(name, 0.0) for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

    # 4) предсказание
    try:
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    # 5) объяснение: пробуем SHAP, иначе fallback на feature_importances_
    top_features = []
    shap_vals_1d = None
    importance_vals = None

    if 'explainer' in globals() and explainer is not None:
        try:
            shap_vals_1d = _safe_shap_1d(explainer, X)  # shape (n_features,)
            if shap_vals_1d.shape[0] != len(FEATURE_NAMES):
                # усечём/дополним до нужной длины
                tmp = np.zeros(len(FEATURE_NAMES), dtype=float)
                n = min(shap_vals_1d.shape[0], len(FEATURE_NAMES))
                tmp[:n] = shap_vals_1d[:n]
                shap_vals_1d = tmp
        except Exception:
            shap_vals_1d = None

    if shap_vals_1d is None:
        # fallback: попробуем взять feature_importances_ у модели (если есть)
        if hasattr(model, "feature_importances_"):
            try:
                importance_vals = np.array(model.feature_importances_, dtype=float)
                if importance_vals.shape[0] != len(FEATURE_NAMES):
                    importance_vals = None
            except Exception:
                importance_vals = None

    # 6) выбираем кандидатов: сначала топ-4 по SHAP (если есть), иначе по importance, иначе первые 4
    if shap_vals_1d is not None:
        sorted_idx = np.argsort(np.abs(shap_vals_1d))[::-1]  # по убыванию важности
    elif importance_vals is not None:
        sorted_idx = np.argsort(np.abs(importance_vals))[::-1]
    else:
        sorted_idx = np.arange(len(FEATURE_NAMES))

    # берем сначала 4 кандидата (в порядке важности)
    candidates = [int(i) for i in sorted_idx[:4].tolist()]

    # удаляем uc_missing, если он попал в кандидаты; потом дополняем до 3 оставшимися из sorted_idx
    filtered = [i for i in candidates if FEATURE_NAMES[i] != "uc_missing"]

    idx_pointer = 4
    while len(filtered) < 3 and idx_pointer < len(sorted_idx):
        idx = int(sorted_idx[idx_pointer])
        if FEATURE_NAMES[idx] != "uc_missing" and idx not in filtered:
            filtered.append(idx)
        idx_pointer += 1

    idx_pointer = 0
    while len(filtered) < 3 and idx_pointer < len(FEATURE_NAMES):
        idx = int(sorted_idx[idx_pointer])
        if FEATURE_NAMES[idx] != "uc_missing" and idx not in filtered:
            filtered.append(idx)
        idx_pointer += 1

    # 7) формируем топ-фичи: берём impact из shap если есть, иначе из importance (или None)
    for idx in filtered[:3]:
        name = FEATURE_NAMES[idx]
        label = FEATURE_DESCRIPTIONS.get(name, name)
        val = X.iloc[0, idx]
        if shap_vals_1d is not None:
            impact = float(shap_vals_1d[idx])
        elif importance_vals is not None:
            impact = float(importance_vals[idx])
        else:
            impact = None
        top_features.append({
            "name": name,
            "label": label,
            "value": float(val) if val is not None else None,
            "impact": impact
        })

    # 8) ответ
    return {
        "prediction": pred,
        "probability": proba,
        "top_features": top_features,
    }
