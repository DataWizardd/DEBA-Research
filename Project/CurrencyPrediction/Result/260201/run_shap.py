# run_shap.py
# ------------------------------------------------------------
# SHAP for: CNN-GRU | Macro + Event + Sentiment (Both) | lb=20 | shift=1
# ------------------------------------------------------------

import os
import re
import json
import time
import random
import platform
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Conv1D, BatchNormalization, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# SHAP
import shap


# ======================
# 0) 한글 폰트 설정
# ======================
def set_korean_font():
    if platform.system() == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf"
    elif platform.system() == "Darwin":
        font_path = "/System/Library/Fonts/AppleGothic.ttf"
    else:
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()


# ======================
# 1) Config
# ======================
CSV_PATH   = "./2026/data/df_final_2026_with_gdelt_and_news.csv"
DATE_COL   = "date"
TARGET_COL = "USD_KRW 종가"

OUT_DIR = "./output_paper_protocol_resume"
PRED_PATH = os.path.join(OUT_DIR, "holdout_predictions_partial_LSTM_GRU_CNNLSTM_CNNGRU.csv")
BESTCV_PATH = os.path.join(OUT_DIR, "best_cv_partial_LSTM_GRU_CNNLSTM_CNNGRU.csv")

# SHAP 대상 고정
CASE_NAME  = "Macro + Event + Sentiment (Both)"
MODEL_NAME = "CNN-GRU"
LOOKBACK   = 20
SHIFT      = 1
SEED       = 42

# 샘플링 (100~1000) 
N_BACKGROUND = 80
N_EXPLAIN_MAX = 600

# chunk 단위 저장
CHUNK_SIZE = 80

# 저장 경로
CACHE_DIR = os.path.join(OUT_DIR, "shap_cache_resume")
os.makedirs(CACHE_DIR, exist_ok=True)

VIZ_DIR = os.path.join(OUT_DIR, "viz_shap_lb20_best_CNNGRU_both")
os.makedirs(VIZ_DIR, exist_ok=True)

CACHE_KEY = f"shap_resume_{MODEL_NAME}__{CASE_NAME}__lb{LOOKBACK}__shift{SHIFT}__bg{N_BACKGROUND}__ex{N_EXPLAIN_MAX}__seed{SEED}"
CACHE_SAFE = re.sub(r"[^0-9a-zA-Z_]+", "_", CACHE_KEY)
CACHE_PATH = os.path.join(CACHE_DIR, f"{CACHE_SAFE}.npz")


# ======================
# 2) TensorFlow / GPU 설정
# ======================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

print("TF:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("GPU name:", gpus[0].name if gpus else "CPU")


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ======================
# 3) Feature set builder (케이스: Macro + Event + Sentiment (Both))
# ======================
def infer_gdelt_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("gkg_") or c.startswith("events_")]

def infer_news_cols_all(df: pd.DataFrame) -> List[str]:
    news_like = []
    for c in df.columns:
        if c.startswith("news_") or c.startswith("sent_"):
            news_like.append(c)
    # 보강 컬럼
    for c in [
        "abs_sent_mean", "pos_ratio", "neg_ratio", "neu_ratio",
        "direct_ratio", "indirect_ratio",
        "sent_net_ratio", "sent_net_count", "sent_std",
        "sent_w_std", "sent_w_mean_roll3", "sent_w_mean_roll7", "sent_w_mean_roll14",
        "sent_w_mean_lag1", "sent_w_mean"
    ]:
        if c in df.columns:
            news_like.append(c)
    return sorted(list(set(news_like)))

def build_case_df_macro_event_sent_both(df_raw: pd.DataFrame) -> pd.DataFrame:
    gdelt_cols = infer_gdelt_cols(df_raw)
    news_cols = infer_news_cols_all(df_raw)

    numeric_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c != TARGET_COL]
    macro_cols = [c for c in numeric_cols if (c not in gdelt_cols and c not in news_cols)]

    keep = [DATE_COL, TARGET_COL] + macro_cols + gdelt_cols + news_cols
    keep = list(dict.fromkeys(keep))
    df_case = df_raw[keep].copy()
    df_case = df_case.sort_values(DATE_COL).dropna(subset=[DATE_COL, TARGET_COL]).reset_index(drop=True)
    df_case = df_case.replace([np.inf, -np.inf], np.nan).dropna()
    return df_case


# ======================
# 4) Split & sequence maker 
# ======================
def final_holdout_split(df: pd.DataFrame, ratio: float = 0.20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_test = int(np.floor(n * ratio))
    n_test = max(1, n_test)
    split = n - n_test
    tr_dev = df.iloc[:split].reset_index(drop=True)
    te     = df.iloc[split:].reset_index(drop=True)
    return tr_dev, te

def make_sequences_residual_with_dates(X, y, dates, lookback, shift=1):
    X_seq, y_d, p_last, y_true, target_dates = [], [], [], [], []
    n = len(y)
    max_i = n - lookback - shift
    for i in range(max_i):
        idx_last   = i + lookback - 1
        target_idx = idx_last + shift
        prev_idx   = target_idx - 1

        X_seq.append(X[i:i+lookback])
        y_d.append(y[target_idx] - y[prev_idx])
        p_last.append(y[prev_idx])
        y_true.append(y[target_idx])
        target_dates.append(dates[target_idx])

    return (np.array(X_seq, np.float32),
            np.array(y_d,   np.float32),
            np.array(p_last, np.float32),
            np.array(y_true, np.float32),
            np.array(target_dates))


# ======================
# 5) Model builder (CNN-GRU)
# ======================
def build_cnn_gru(input_shape, hp: Dict):
    lr = float(hp["lr"])
    dropout = float(hp["dropout"])
    rnn1 = int(hp["rnn1"])
    rnn2 = int(hp["rnn2"])
    conv_filters = int(hp["conv_filters"])
    kernel = int(hp["kernel"])
    bn = bool(hp["bn"])

    def maybe_bn(flag: bool):
        return BatchNormalization() if flag else Lambda(lambda x: x)

    m = Sequential([
        Input(shape=input_shape),
        Conv1D(conv_filters, kernel_size=kernel, padding="causal", activation="relu"),
        maybe_bn(bn),
        Dropout(dropout),
        Conv1D(conv_filters, kernel_size=kernel, padding="causal", activation="relu"),
        maybe_bn(bn),
        Dropout(dropout),
        GRU(rnn1, return_sequences=True, recurrent_dropout=0.10),
        Dropout(dropout),
        GRU(rnn2, recurrent_dropout=0.10),
        Dropout(dropout),
        Dense(1),
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    m.compile(optimizer=opt, loss="mse")
    return m


def try_fit(model, X_train, y_train, X_val, y_val, epochs=60, batch=64, es_patience=5, rlrop_patience=3):
    cbs = [
        EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=rlrop_patience, min_lr=1e-5, verbose=0),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch,
        callbacks=cbs,
        verbose=0,
    )


# ======================
# 6) Load best HP from best_cv file
# ======================
def load_best_hp(bestcv_path: str, case: str, model: str, lookback: int, shift: int) -> Dict:
    df = pd.read_csv(bestcv_path, encoding="utf-8-sig")
    sub = df[
        (df["case"] == case) &
        (df["model"] == model) &
        (df["lookback"] == lookback) &
        (df["shift"] == shift)
    ]
    if sub.empty:
        raise ValueError(f"best_cv not found for: {case} | {model} | lb={lookback} | shift={shift}")
    hp_json = sub.iloc[-1]["best_hp_json"]
    hp = json.loads(hp_json)
    print("[HP] loaded keys:", list(hp.keys()))
    return hp


# ======================
# 7) Korean feature name mapping 
# ======================
BASE_KO_MAP = {
    # --- GKG Global ---
    "gkg_doc_cnt": "GKG 전체 문서 수",
    "gkg_tone_mean": "GKG 전체 톤 평균",
    "gkg_tone_std": "GKG 전체 톤 표준편차",
    "gkg_pos_ratio": "GKG 전체 긍정 비율",
    "gkg_neu_ratio": "GKG 전체 중립 비율",
    "gkg_neg_ratio": "GKG 전체 부정 비율",
    "gkg_net_tone": "GKG 전체 순톤 (긍·부정 결합)",

    # --- GKG Korea ---
    "gkg_kr_doc_cnt": "GKG 한국 문서 수",
    "gkg_kr_doc_share": "GKG 한국 문서 비중",
    "gkg_kr_pos_ratio": "GKG 한국 긍정 비율",
    "gkg_kr_neu_ratio": "GKG 한국 중립 비율",
    "gkg_kr_neg_ratio": "GKG 한국 부정 비율",
    "gkg_kr_pos_tone_mean": "GKG 한국 긍정 톤 평균",
    "gkg_kr_neu_tone_mean": "GKG 한국 중립 톤 평균",
    "gkg_kr_neg_tone_mean": "GKG 한국 부정 톤 평균",
    "gkg_kr_net_tone": "GKG 한국 순톤",
    "gkg_kr_tone_std": "GKG 한국 톤 분산",
    "gkg_kr_loc_cnt": "GKG 한국 로케이션 언급 수",

    # --- Events ---
    "events_total_cnt": "전체 이벤트 수",
    "events_kr_cnt": "한국 관련 이벤트 수",
    "events_pos_ratio": "이벤트 긍정 비율",
    "events_neu_ratio": "이벤트 중립 비율",
    "events_neg_ratio": "이벤트 부정 비율",
    "events_pos_tone_mean": "이벤트 긍정 톤 평균",
    "events_neg_tone_mean": "이벤트 부정 톤 평균",

    # --- News / Sentiment basic counts ---
    "news_total": "전체 뉴스 수",
    "news_direct": "직접 뉴스 수",
    "news_indirect": "간접 뉴스 수",
    "news_pos": "긍정 뉴스 수",
    "news_neg": "부정 뉴스 수",
    "news_neu": "중립 뉴스 수",

    # --- Sentiment stats ---
    "sent_mean": "평균 감정 점수",
    "sent_std": "감정 점수 표준편차",
    "sent_w_std": "가중 감정 표준편차",
    "abs_sent_mean": "감정 강도(절대값) 평균",

    "sent_net_count": "순감정 뉴스 수 (긍정 − 부정)",
    "sent_net_ratio": "순감정 비율",

    "pos_ratio": "긍정 뉴스 비율",
    "neg_ratio": "부정 뉴스 비율",
    "neu_ratio": "중립 뉴스 비율",
    "direct_ratio": "직접 뉴스 비율",
    "indirect_ratio": "간접 뉴스 비율",
}

def to_korean_feature_name(col: str) -> str:
    # 1) exact match
    if col in BASE_KO_MAP:
        return BASE_KO_MAP[col]

    # 2) rolling pattern: xxx_roll{w}
    m = re.match(r"^(news_total|sent_mean|sent_w_mean|sent_net_ratio|abs_sent_mean)_roll(\d+)$", col)
    if m:
        base, w = m.group(1), m.group(2)
        if base == "news_total":
            return f"최근 {w}일 뉴스 수"
        if base == "sent_mean":
            return f"최근 {w}일 평균 감정"
        if base == "sent_w_mean":
            return f"최근 {w}일 가중 평균 감정"
        if base == "sent_net_ratio":
            return f"최근 {w}일 순감정 비율"
        if base == "abs_sent_mean":
            return f"최근 {w}일 감정 강도"
        return col

    # 3) lag pattern: xxx_lag1
    m = re.match(r"^(news_total|news_direct|news_indirect|news_pos|news_neg|news_neu|sent_mean|sent_w_mean|sent_net_ratio|abs_sent_mean)_lag1$", col)
    if m:
        base = m.group(1)
        lag_map = {
            "news_total": "전일 뉴스 수",
            "news_direct": "전일 직접 뉴스 수",
            "news_indirect": "전일 간접 뉴스 수",
            "news_pos": "전일 긍정 뉴스 수",
            "news_neg": "전일 부정 뉴스 수",
            "news_neu": "전일 중립 뉴스 수",
            "sent_mean": "전일 평균 감정",
            "sent_w_mean": "전일 가중 감정",
            "sent_net_ratio": "전일 순감정 비율",
            "abs_sent_mean": "전일 감정 강도",
        }
        return lag_map.get(base, col)

    return col


# ======================
# 8) SHAP compute with resume cache
# ======================
def compute_shap_with_resume(
    model: tf.keras.Model,
    X_bg: np.ndarray,                # (B, lb, F)
    X_explain: np.ndarray,           # (N, lb, F)
    X_explain_flat: np.ndarray,      # (N, F) 
    cache_path: str,
    chunk_size: int = 80
):

    n = X_explain.shape[0]
    n_feat = X_explain.shape[-1]

    # 캐시 로드
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        shap_agg = z["shap_agg"]          # (N, F)
        X_agg = z["X_agg"]                # (N, F) beeswarm용
        done_mask = z["done_mask"]        # (N,)
        bg_idx = z["bg_idx"]
        ex_idx = z["ex_idx"]
        print("[CACHE] partial cache detected -> will resume SHAP computation.")
        print(f"[RESUME] loaded. done={int(done_mask.sum())}/{n}")
        return shap_agg, X_agg, done_mask, bg_idx, ex_idx

    # 캐시 신규 생성
    shap_agg = np.zeros((n, n_feat), dtype=np.float32)
    X_agg = X_explain_flat.astype(np.float32).copy()
    done_mask = np.zeros((n,), dtype=bool)

    bg_idx = np.arange(len(X_bg))
    ex_idx = np.arange(n)

    np.savez_compressed(cache_path, shap_agg=shap_agg, X_agg=X_agg, done_mask=done_mask, bg_idx=bg_idx, ex_idx=ex_idx)
    print("[CACHE] created:", cache_path, "| done: 0 /", n)

    return shap_agg, X_agg, done_mask, bg_idx, ex_idx


def save_cache(cache_path: str, shap_agg: np.ndarray, X_agg: np.ndarray, done_mask: np.ndarray, bg_idx: np.ndarray, ex_idx: np.ndarray):
    np.savez_compressed(cache_path, shap_agg=shap_agg, X_agg=X_agg, done_mask=done_mask, bg_idx=bg_idx, ex_idx=ex_idx)


def run_shap_chunks(
    model: tf.keras.Model,
    X_bg: np.ndarray,
    X_explain: np.ndarray,
    X_explain_flat: np.ndarray,
    cache_path: str,
    chunk_size: int = 80
):
    shap_agg, X_agg, done_mask, bg_idx, ex_idx = compute_shap_with_resume(
        model=model,
        X_bg=X_bg,
        X_explain=X_explain,
        X_explain_flat=X_explain_flat,
        cache_path=cache_path,
        chunk_size=chunk_size
    )

    n = X_explain.shape[0]
    remaining = np.where(~done_mask)[0]
    print("[INFO] remaining explain samples:", len(remaining))

    if len(remaining) == 0:
        return shap_agg, X_agg

    # Explainer 생성
    explainer = shap.GradientExplainer(model, X_bg)

    # chunk loop
    start = 0
    while start < len(remaining):
        batch_idx = remaining[start:start+chunk_size]
        X_batch = X_explain[batch_idx]  # (B, lb, F)

        # shap_values
        shap_vals = explainer.shap_values(X_batch)

        # shap API 호환
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        # (B, lb, F, 1) 또는 (B, lb, F) 또는 (B, F, 1) 등 변형 방어
        shap_vals = np.asarray(shap_vals)

        # GradientExplainer(Keras) -> (B, lb, F) 혹은 (B, lb, F, 1)
        if shap_vals.ndim == 4 and shap_vals.shape[-1] == 1:
            shap_vals = np.squeeze(shap_vals, axis=-1)  # (B, lb, F)

        # time축(lookback)을 sum으로 집계 -> (B, F)
        if shap_vals.ndim == 3:
            shap_batch = shap_vals.sum(axis=1)  # sum over lookback
        elif shap_vals.ndim == 2:
            shap_batch = shap_vals
        else:
            raise ValueError(f"Unexpected shap_vals shape: {shap_vals.shape}")

        # ---- (B, F, 1) 케이스 최종 방어 ----
        if shap_batch.ndim == 3 and shap_batch.shape[-1] == 1:
            shap_batch = np.squeeze(shap_batch, axis=-1)  # (B, F)

        if shap_batch.shape != (len(batch_idx), shap_agg.shape[1]):
            raise ValueError(f"shape mismatch after squeeze/agg: got {shap_batch.shape}, expected {(len(batch_idx), shap_agg.shape[1])}")

        shap_agg[batch_idx] = shap_batch
        done_mask[batch_idx] = True

        save_cache(cache_path, shap_agg, X_agg, done_mask, bg_idx, ex_idx)
        print(f"[CACHE] updated. done={int(done_mask.sum())}/{n} | last batch={len(batch_idx)}")

        start += chunk_size

    return shap_agg, X_agg


# ======================
# 9) Plotting (beeswarm + bar)
# ======================
def plot_beeswarm_and_bar(shap_values_2d: np.ndarray, X_2d: np.ndarray, feat_names: List[str], out_prefix: str, max_display: int = 50):
    """
    shap_values_2d: (N, F)
    X_2d: (N, F)
    """
    # 한글 feature names
    feat_names_ko = [to_korean_feature_name(c) for c in feat_names]

    # 중요도 (mean abs)
    imp = np.mean(np.abs(shap_values_2d), axis=0)
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "feature_ko": feat_names_ko,
        "mean_abs_shap": imp
    }).sort_values("mean_abs_shap", ascending=False)

    imp_csv = out_prefix + "_importance.csv"
    imp_df.to_csv(imp_csv, index=False, encoding="utf-8-sig")

    # SHAP summary plot용 객체
    expl = shap.Explanation(
        values=shap_values_2d,
        data=X_2d,
        feature_names=feat_names_ko
    )

    # beeswarm
    plt.figure(figsize=(10, 12))
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    beeswarm_path = out_prefix + "_beeswarm.png"
    plt.tight_layout()
    plt.savefig(beeswarm_path, dpi=220)
    plt.close()

    # bar
    plt.figure(figsize=(10, 8))
    shap.plots.bar(expl, max_display=max_display, show=False)
    bar_path = out_prefix + "_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=220)
    plt.close()

    print("[SAVED]", imp_csv)
    print("[SAVED]", beeswarm_path)
    print("[SAVED]", bar_path)


# ======================
# 10) Main
# ======================
def main():
    # sanity
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.exists(BESTCV_PATH):
        raise FileNotFoundError(f"best_cv not found: {BESTCV_PATH}")

    set_seed(SEED)

    # load data
    df_raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig").replace([np.inf, -np.inf], np.nan)
    df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL], errors="coerce")
    df_raw = df_raw.sort_values(DATE_COL).dropna(subset=[DATE_COL, TARGET_COL]).reset_index(drop=True)

    # build case df
    df_case = build_case_df_macro_event_sent_both(df_raw)

    # split
    df_train_dev, df_holdout = final_holdout_split(df_case, ratio=0.20)

    feats = [c for c in df_case.select_dtypes(include=[np.number]).columns if c != TARGET_COL]
    n_feat = len(feats)
    print("[INFO] n_features:", n_feat)
    print("[INFO] train_dev:", df_train_dev.shape, "| holdout:", df_holdout.shape)
    print("[INFO] holdout period:", df_holdout[DATE_COL].min(), "~", df_holdout[DATE_COL].max())

    # load best hp
    hp = load_best_hp(BESTCV_PATH, CASE_NAME, MODEL_NAME, LOOKBACK, SHIFT)

    from sklearn.preprocessing import RobustScaler

    X_tr_raw = df_train_dev[feats].values.astype(np.float32)
    y_tr_raw = df_train_dev[TARGET_COL].values.astype(np.float32)
    d_tr     = df_train_dev[DATE_COL].values

    X_te_raw = df_holdout[feats].values.astype(np.float32)
    y_te_raw = df_holdout[TARGET_COL].values.astype(np.float32)
    d_te     = df_holdout[DATE_COL].values

    x_scaler = RobustScaler()
    X_tr = x_scaler.fit_transform(X_tr_raw)
    X_te = x_scaler.transform(X_te_raw)

    # residual seq
    X_tr_seq, y_d_tr_raw, _, _, _ = make_sequences_residual_with_dates(X_tr, y_tr_raw, d_tr, LOOKBACK, shift=SHIFT)
    if len(X_tr_seq) < 40:
        raise ValueError("train_dev sequence too short for SHAP run")

    d_scaler = RobustScaler()
    y_d_tr_sc = d_scaler.fit_transform(y_d_tr_raw.reshape(-1, 1)).ravel()

    # train/val split
    n_tr_seq = len(X_tr_seq)
    val_sz = max(2, int(n_tr_seq * 0.1))
    X_train, y_train = X_tr_seq[:-val_sz], y_d_tr_sc[:-val_sz]
    X_val,   y_val   = X_tr_seq[-val_sz:],  y_d_tr_sc[-val_sz:]

    # build & fit model
    tf.keras.backend.clear_session()
    model = build_cnn_gru(input_shape=(LOOKBACK, n_feat), hp=hp)

    # speedplan 
    try_fit(model, X_train, y_train, X_val, y_val, epochs=60, batch=64, es_patience=5, rlrop_patience=3)

    # make holdout sequences (explain 대상)
    X_te_seq, _, p_last_te, y_true_te, ex_dates = make_sequences_residual_with_dates(X_te, y_te_raw, d_te, LOOKBACK, shift=SHIFT)
    if len(X_te_seq) < 20:
        raise ValueError("holdout sequence too short")

    # 샘플 수 결정 
    n_total = len(X_te_seq)
    n_explain = min(N_EXPLAIN_MAX, n_total)
    n_bg = min(N_BACKGROUND, n_total)

    # 재현성 있는 인덱스 샘플링 
    rng = np.random.RandomState(SEED + 12345)
    all_idx = np.arange(n_total)
    rng.shuffle(all_idx)
    ex_idx = np.sort(all_idx[:n_explain])

    # background는 explain과 겹쳐도 되지만 보통은 따로
    rest = all_idx[n_explain:]
    if len(rest) >= n_bg:
        bg_idx = np.sort(rest[:n_bg])
    else:
        bg_idx = np.sort(all_idx[:n_bg])

    X_explain = X_te_seq[ex_idx]  # (N, lb, F)
    X_bg = X_te_seq[bg_idx]       # (B, lb, F)

    X_explain_flat = X_explain[:, -1, :]

    print(f"[SAMPLE] background={len(X_bg)} | explain={len(X_explain)} | chunk={CHUNK_SIZE}")

    if not os.path.exists(CACHE_PATH):
        # placeholder create 후 idx 저장
        shap_agg0 = np.zeros((len(X_explain), n_feat), dtype=np.float32)
        done0 = np.zeros((len(X_explain),), dtype=bool)
        np.savez_compressed(CACHE_PATH, shap_agg=shap_agg0, X_agg=X_explain_flat.astype(np.float32),
                            done_mask=done0, bg_idx=bg_idx, ex_idx=ex_idx)
        print("[CACHE] saved:", CACHE_PATH, "| done: 0 /", len(X_explain))
    else:
        # 기존 캐시가 있으면 그 idx로 강제(일관성)
        z = np.load(CACHE_PATH, allow_pickle=True)
        cached_bg = z["bg_idx"]
        cached_ex = z["ex_idx"]
        print("[CACHE] partial cache detected -> will resume SHAP computation.")
        print("[RESUME] using cached bg_idx/ex_idx to keep consistency.")
        X_bg = X_te_seq[cached_bg]
        X_explain = X_te_seq[cached_ex]
        X_explain_flat = X_explain[:, -1, :]

    # run shap in chunks (resume-safe)
    shap_agg, X_agg = run_shap_chunks(
        model=model,
        X_bg=X_bg,
        X_explain=X_explain,
        X_explain_flat=X_explain_flat,
        cache_path=CACHE_PATH,
        chunk_size=CHUNK_SIZE
    )

    # 최종 산출물 저장/그리기
    out_prefix = os.path.join(
        VIZ_DIR,
        f"SHAP_{MODEL_NAME}_CASE_BOTH_lb{LOOKBACK}_shift{SHIFT}_seed{SEED}"
    )
    plot_beeswarm_and_bar(shap_agg, X_agg, feats, out_prefix, max_display=60)

    print("\nSHAP DONE")


if __name__ == "__main__":
    main()
