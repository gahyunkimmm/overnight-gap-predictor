# -*- coding: utf-8 -*-
"""
gap_model.py
============
야간 신호 -> 익일 한국 개장 갭 예측의 공용 로직.
CLI(overnight_gap_predict.py)와 대시보드(app.py)가 함께 사용한다.

모델: 표준화 + 릿지 회귀(λ=10), 미국 반도체/시장 신호 9종.
검증: 워크포워드(확장창) — 실전과 동일하게 '과거로 학습 → 다음날 예측'을 반복.
부가: 개장→종가(장중) 예측력도 함께 측정해 갭 예측과 비교한다.
"""
import os
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf

START = "2022-01-01"
RIDGE = 10.0
MIN_FEATURES = 3  # 가용 신호가 이보다 적으면 신뢰 불가로 본다

KR = {"삼성전자": "005930", "SK하이닉스": "000660", "삼성전기": "009150", "한미반도체": "042700"}

US_TICKERS = {
    "MU": "MU",        # 마이크론 - 메모리 직접 경쟁사
    "SOX": "^SOX",     # 필라델피아 반도체지수
    "EWY": "EWY",      # iShares 한국 ETF
    "USDKRW": "KRW=X", # 원/달러
    "NVDA": "NVDA",    # 엔비디아
    "TSM": "TSM",      # TSMC ADR
    "NASDAQ": "^IXIC", # 나스닥 종합
    "VIX": "^VIX",     # 변동성 지수
    "SP500": "^GSPC",  # S&P 500
}
FEATURES = list(US_TICKERS.keys())

SIGNAL_LABELS = {
    "MU": "마이크론", "SOX": "반도체지수", "EWY": "한국ETF", "USDKRW": "원/달러",
    "NVDA": "엔비디아", "TSM": "TSMC", "NASDAQ": "나스닥", "VIX": "VIX", "SP500": "S&P500",
}


# ----------------------------- 데이터 -----------------------------
def get_kr(code):
    """한국 시가/종가. FinanceDataReader(네이버/KRX) 우선, 실패 시 yfinance .KS 폴백."""
    try:
        df = fdr.DataReader(code, START)
        out = df[["Open", "Close"]].copy()
        out.index = pd.to_datetime(out.index).normalize()
        out = _clean_prices(out)
        if len(out.dropna()) >= 50:
            return out
    except Exception:
        pass

    d = yf.download(f"{code}.KS", start=START, progress=False, auto_adjust=True)
    if d is None or len(d) == 0:
        raise RuntimeError(f"{code} 한국 주가 수집 실패 (fdr/yfinance 모두)")
    out = d[["Open", "Close"]].copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out.index = pd.to_datetime(out.index).normalize()
    return _clean_prices(out)


def _clean_prices(out):
    """시가/종가가 0 이하인 잘못된 데이터 행 제거(0 나눗셈 → inf 방지)."""
    return out[(out["Open"] > 0) & (out["Close"] > 0)]


def get_us_returns():
    cols = {}
    for name, tk in US_TICKERS.items():
        try:
            d = yf.download(tk, start=START, progress=False, auto_adjust=True)
        except Exception:
            d = None
        if d is None or len(d) == 0:
            continue
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        ret = close.pct_change() * 100.0
        ret.index = pd.to_datetime(ret.index).normalize()
        cols[name] = ret
    return pd.DataFrame(cols)


def available_features(us):
    """수집에 성공한 신호만 반환(원래 FEATURES 순서 유지)."""
    return [f for f in FEATURES if f in us.columns]


# ----------------------------- 모델 -----------------------------
def _standardize(Xtr):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def ridge_fit(X, y, lam=RIDGE):
    """표준화 + 릿지 closed-form. 절편은 정규화 제외. 반환: (beta, mu, sd)."""
    mu, sd = _standardize(X)
    Xs = (X - mu) / sd
    Xb = np.column_stack([np.ones(len(Xs)), Xs])
    p = Xb.shape[1]
    A = Xb.T @ Xb + lam * np.eye(p)
    A[0, 0] -= lam
    beta = np.linalg.solve(A, Xb.T @ y)
    return beta, mu, sd


def ridge_predict(beta, mu, sd, X):
    Xs = (X - mu) / sd
    return np.column_stack([np.ones(len(Xs)), Xs]) @ beta


def build_dataset(kr, us, feats, target="gap"):
    """한국 거래일 d의 타깃 <- d 직전 미국 등락 매칭(미래 정보 누설 없음).
    target='gap'      : 개장 갭 = (시가/전일종가 - 1)
    target='intraday' : 장중 = (종가/시가 - 1)
    """
    kr = kr.copy()
    kr["prev_close"] = kr["Close"].shift(1)
    if target == "gap":
        kr["y"] = (kr["Open"] / kr["prev_close"] - 1) * 100.0
    elif target == "intraday":
        kr["y"] = (kr["Close"] / kr["Open"] - 1) * 100.0
    else:
        raise ValueError(target)
    kr["y"] = kr["y"].replace([np.inf, -np.inf], np.nan)  # 0 나눗셈 잔재 제거

    us_sorted = us[feats].sort_index()
    rows = []
    for d, y in kr["y"].items():
        if pd.isna(y):
            continue
        prev = us_sorted.loc[us_sorted.index < d]   # 엄격히 '이전'만 → 누설 차단
        if len(prev) == 0:
            continue
        sig = prev.iloc[-1]
        if sig.isna().any():
            continue
        rows.append((d, y, *sig.values))
    return pd.DataFrame(rows, columns=["date", "y"] + feats).set_index("date").dropna()


def walk_forward(data, feats, start_frac=0.7):
    """확장창 워크포워드: 각 시점을 '그 이전 전부'로 학습해 1스텝 예측.
    반환: dict(r2, hit, mae, n)."""
    X = data[feats].values
    y = data["y"].values
    n = len(data)
    start = max(int(n * start_frac), 50)
    preds, acts = [], []
    for i in range(start, n):
        beta, mu, sd = ridge_fit(X[:i], y[:i])
        preds.append(ridge_predict(beta, mu, sd, X[i:i + 1])[0])
        acts.append(y[i])
    preds, acts = np.array(preds), np.array(acts)
    if len(acts) == 0:
        return {"r2": float("nan"), "hit": float("nan"), "mae": float("nan"), "n": 0}
    ss_res = np.sum((acts - preds) ** 2)
    ss_tot = np.sum((acts - acts.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    hit = np.mean(np.sign(preds) == np.sign(acts)) * 100
    mae = np.mean(np.abs(preds - acts))
    return {"r2": float(r2), "hit": float(hit), "mae": float(mae), "n": int(len(acts))}


def predict_all():
    """전체 예측. 반환: (latest_signal, latest_date, feats, results[list of dict])"""
    us = get_us_returns()
    feats = available_features(us)
    if len(feats) < MIN_FEATURES:
        raise RuntimeError(
            f"가용 신호가 부족합니다({len(feats)}/{len(FEATURES)}). 데이터 소스 점검 필요."
        )

    us_clean = us[feats].sort_index().dropna()
    latest = us_clean.iloc[-1]
    latest_date = us_clean.index[-1].date()

    results = []
    for name, code in KR.items():
        kr = get_kr(code)
        last_close = kr["Close"].dropna().iloc[-1]
        last_close_date = kr["Close"].dropna().index[-1].date()

        # 갭 예측
        data = build_dataset(kr, us, feats, target="gap")
        wf = walk_forward(data, feats)
        beta, mu, sd = ridge_fit(data[feats].values, data["y"].values)
        resid_std = np.std(data["y"].values - ridge_predict(beta, mu, sd, data[feats].values))

        x = latest[feats].values.astype(float).reshape(1, -1)
        pred_gap = float(ridge_predict(beta, mu, sd, x)[0])
        pred_open = last_close * (1 + pred_gap / 100.0)
        lo = last_close * (1 + (pred_gap - resid_std) / 100.0)
        hi = last_close * (1 + (pred_gap + resid_std) / 100.0)

        recent = data.tail(30).copy()
        recent["fitted"] = ridge_predict(beta, mu, sd, recent[feats].values)
        recent = recent.rename(columns={"y": "gap"})[["gap", "fitted"]]

        # 장중(개장→종가) 예측력 — 정직한 비교용
        intraday = build_dataset(kr, us, feats, target="intraday")
        wf_intra = walk_forward(intraday, feats)

        results.append({
            "name": name, "code": code,
            "last_close": float(last_close), "last_close_date": last_close_date,
            "pred_gap": pred_gap, "pred_open": float(pred_open),
            "lo": float(lo), "hi": float(hi),
            "r2_wf": wf["r2"], "hit_wf": wf["hit"], "n": wf["n"],
            "intraday_r2": wf_intra["r2"], "intraday_hit": wf_intra["hit"],
            "recent": recent,
        })

    return latest, latest_date, feats, results


# ----------------------- 사후 정확도(실측) -----------------------
def realized_accuracy(log_path):
    """예측 로그를 실제 개장 갭과 대조해 실측 정확도를 계산한다.
    반환: (summary_df, detail_df). 데이터 없으면 (None, None)."""
    if not os.path.exists(log_path):
        return None, None
    try:
        log = pd.read_csv(log_path, encoding="utf-8-sig")
    except Exception:
        return None, None
    if log.empty or "last_close_date" not in log.columns:
        return None, None

    kr_cache = {}
    detail = []
    for _, row in log.iterrows():
        code = str(row["code"]).zfill(6)
        if code not in kr_cache:
            try:
                k = get_kr(code).copy()
                k["prev_close"] = k["Close"].shift(1)
                k["actual_gap"] = (k["Open"] / k["prev_close"] - 1) * 100.0
                kr_cache[code] = k
            except Exception:
                kr_cache[code] = None
        k = kr_cache[code]
        if k is None:
            continue
        lcd = pd.to_datetime(row["last_close_date"], errors="coerce")
        if pd.isna(lcd):
            continue
        lcd = lcd.normalize()
        future = k.loc[k.index > lcd]
        if len(future) == 0:
            continue  # 타깃일 아직 미도래
        actual_gap = future.iloc[0]["actual_gap"]
        if pd.isna(actual_gap):
            continue
        pred_gap = float(row["pred_gap_pct"])
        detail.append({
            "stock": row["stock"], "target_date": future.index[0].date(),
            "pred_gap": pred_gap, "actual_gap": float(actual_gap),
            "hit": int(np.sign(pred_gap) == np.sign(actual_gap)),
            "abs_err": abs(pred_gap - actual_gap),
        })

    if not detail:
        return None, None
    det = pd.DataFrame(detail).drop_duplicates(subset=["stock", "target_date"])
    summary = det.groupby("stock").agg(
        n=("hit", "size"),
        hit_rate=("hit", lambda s: round(s.mean() * 100, 1)),
        MAE=("abs_err", lambda s: round(s.mean(), 2)),
    ).reset_index()
    return summary, det.sort_values("target_date")
