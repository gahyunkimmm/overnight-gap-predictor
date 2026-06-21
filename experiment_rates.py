# -*- coding: utf-8 -*-
"""
실험: 미국 국채 수익률(시장금리) 신호가 개장 갭 예측을 개선하는가?
baseline(현재 9신호) vs +금리 신호를 워크포워드로 비교.

금리 신호(yfinance):
  ^TNX  미국 10년물 국채 수익률
  ^IRX  미국 13주 단기물(정책금리 근접)
  ^FVX  미국 5년물
  ^TYX  미국 30년물
정책금리(기준금리)는 일별 변동이 없어 일간 모델 신호로 부적합 → 제외.
"""
import sys, io
import numpy as np
import pandas as pd
import yfinance as yf

import gap_model as gm

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

RATES = {"US10Y": "^TNX", "US13W": "^IRX", "US5Y": "^FVX", "US30Y": "^TYX"}


def rate_returns():
    cols = {}
    for name, tk in RATES.items():
        try:
            d = yf.download(tk, start=gm.START, progress=False, auto_adjust=True)
        except Exception:
            d = None
        if d is None or len(d) == 0:
            continue
        c = d["Close"]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[:, 0]
        # 수익률(레벨)의 일간 '변화'를 신호로 사용 (절대 bp 변화에 비례)
        cols[name] = c.diff()
        cols[name].index = pd.to_datetime(cols[name].index).normalize()
    return pd.DataFrame(cols)


def main():
    print("미국 시장 신호 + 금리 신호 수집...")
    us = gm.get_us_returns()
    base = gm.available_features(us)
    rates = rate_returns()
    print(f"  기존 신호 {base}")
    print(f"  금리 신호 {list(rates.columns)}")

    us_all = us.join(rates, how="outer").sort_index()
    rate_feats = list(rates.columns)

    for name, code in gm.KR.items():
        kr = gm.get_kr(code)
        print(f"\n{'='*64}\n  {name} ({code})\n{'='*64}")
        print(f"  {'설정':30s} {'WF R²':>8s} {'적중%':>7s} {'MAE':>7s}")
        configs = [
            ("baseline (9신호)", base),
            ("+ US10Y", base + ["US10Y"]),
            ("+ US10Y+US13W", base + ["US10Y", "US13W"]),
            ("+ 금리 4종 전부", base + rate_feats),
        ]
        for label, feats in configs:
            feats = [f for f in feats if f in us_all.columns]
            data = gm.build_dataset(kr, us_all, feats, target="gap")
            wf = gm.walk_forward(data, feats)
            print(f"  {label:30s} {wf['r2']:8.3f} {wf['hit']:6.1f}% {wf['mae']:6.2f}  (n={wf['n']})")


if __name__ == "__main__":
    main()
