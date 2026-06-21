# -*- coding: utf-8 -*-
"""
gap_model.py
============
야간 신호 -> 익일 한국 개장 갭 예측의 공용 로직.
CLI(overnight_gap_predict.py)와 대시보드(app.py)가 함께 사용한다.

모델: 표준화 + 릿지 회귀(λ=10). 미국 반도체/시장 신호 9종을 사용한다.
신호 선택 근거는 experiment.py 참고(확장 신호가 방향 적중률을 개선).
"""
import os
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import yfinance as yf

START = "2022-01-01"
RIDGE = 10.0

KR = {"삼성전자": "005930", "SK하이닉스": "000660", "삼성전기": "009150", "현대차": "005380"}

# 미국 야간 신호 (전부 yfinance로 안정 수집)
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
    """한국 시가/종가. FinanceDataReader(네이버/KRX) 우선, 해외 서버 등에서
    실패하면 yfinance의 .KS 티커로 폴백한다."""
    try:
        df = fdr.DataReader(code, START)
        out = df[["Open", "Close"]].copy()
        out.index = pd.to_datetime(out.index).normalize()
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
    return out


def get_us_returns():
    cols = {}
    for name, tk in US_TICKERS.items():
        d = yf.download(tk, start=START, progress=False, auto_adjust=True)
        if d is None or len(d) == 0:
            continue
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        ret = close.pct_change() * 100.0
        ret.index = pd.to_datetime(ret.index).normalize()
        cols[name] = ret
    return pd.DataFrame(cols)


# ----------------------------- 모델 -----------------------------
def _standardize(Xtr):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def ridge_fit(X, y, lam=RIDGE):
    """표준화 + 릿지 closed-form. 반환: (beta, mu, sd). 절편은 정규화 제외."""
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


def build_dataset(kr, us):
    """한국 거래일 d의 갭 <- d 직전 미국 등락 매칭."""
    kr = kr.copy()
    kr["prev_close"] = kr["Close"].shift(1)
    kr["gap"] = (kr["Open"] / kr["prev_close"] - 1) * 100.0
    us_sorted = us.sort_index()
    rows = []
    for d, gap in kr["gap"].items():
        if pd.isna(gap):
            continue
        prev = us_sorted.loc[us_sorted.index < d]
        if len(prev) == 0:
            continue
        sig = prev.iloc[-1]
        if sig.isna().any():
            continue
        rows.append((d, gap, *sig.values))
    cols = ["date", "gap"] + FEATURES
    return pd.DataFrame(rows, columns=cols).set_index("date").dropna()


def train_and_score(data):
    X = data[FEATURES].values
    y = data["gap"].values
    n = len(data)
    cut = int(n * 0.7)

    beta_tr, mu_tr, sd_tr = ridge_fit(X[:cut], y[:cut])
    yhat_te = ridge_predict(beta_tr, mu_tr, sd_tr, X[cut:])
    yte = y[cut:]
    ss_res = np.sum((yte - yhat_te) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    r2_out = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    hit = np.mean(np.sign(yhat_te) == np.sign(yte)) * 100

    # 최종 예측엔 전체 데이터로 재학습
    beta, mu, sd = ridge_fit(X, y)
    resid = y - ridge_predict(beta, mu, sd, X)
    resid_std = np.std(resid)
    return (beta, mu, sd), r2_out, hit, resid_std, n


def predict_all():
    """전체 예측. 반환: (latest_signal, latest_date, results[list of dict])"""
    us = get_us_returns()
    if us.empty:
        raise RuntimeError("미국 신호 데이터 수집 실패")

    us_clean = us.sort_index().dropna()
    latest = us_clean.iloc[-1]
    latest_date = us_clean.index[-1].date()

    results = []
    for name, code in KR.items():
        kr = get_kr(code)
        last_close = kr["Close"].dropna().iloc[-1]
        last_close_date = kr["Close"].dropna().index[-1].date()

        data = build_dataset(kr, us)
        (beta, mu, sd), r2_out, hit, resid_std, n = train_and_score(data)

        x = latest[FEATURES].values.astype(float).reshape(1, -1)
        pred_gap = float(ridge_predict(beta, mu, sd, x)[0])
        pred_open = last_close * (1 + pred_gap / 100.0)
        lo = last_close * (1 + (pred_gap - resid_std) / 100.0)
        hi = last_close * (1 + (pred_gap + resid_std) / 100.0)

        recent = data.tail(30).copy()
        recent["fitted"] = ridge_predict(beta, mu, sd, recent[FEATURES].values)

        results.append({
            "name": name, "code": code,
            "last_close": float(last_close), "last_close_date": last_close_date,
            "pred_gap": pred_gap, "pred_open": float(pred_open),
            "lo": float(lo), "hi": float(hi),
            "r2_out": float(r2_out), "hit": float(hit), "n": int(n),
            "recent": recent[["gap", "fitted"]],
        })

    return latest, latest_date, results


# ----------------------- 사후 정확도(실측) -----------------------
def realized_accuracy(log_path):
    """예측 로그를 실제 개장 갭과 대조해 실측 정확도를 계산한다.
    반환: (summary_df, detail_df). 데이터 없으면 (None, None).
    """
    if not os.path.exists(log_path):
        return None, None
    try:
        log = pd.read_csv(log_path, encoding="utf-8-sig")
    except Exception:
        return None, None
    if log.empty or "last_close_date" not in log.columns:
        # 구버전 로그(타깃일 정보 없음)는 평가 불가
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
            continue  # 구버전 로그(타깃일 정보 없음) 건너뜀
        lcd = lcd.normalize()
        future = k.loc[k.index > lcd]
        if len(future) == 0:
            continue  # 타깃일 아직 미도래
        target_day = future.index[0]
        actual_gap = future.iloc[0]["actual_gap"]
        if pd.isna(actual_gap):
            continue
        pred_gap = float(row["pred_gap_pct"])
        detail.append({
            "stock": row["stock"], "target_date": target_day.date(),
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
