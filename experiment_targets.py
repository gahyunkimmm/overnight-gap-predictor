# -*- coding: utf-8 -*-
"""
실험: 야간 미국 신호가 어떤 '한국 일일 가격'을 가장 잘 설명하는가?
  - gap      : 전일종가 → 시가 (개장 갭, 현행 타깃)
  - daily    : 전일종가 → 종가 (일반적인 일간수익률)
  - intraday : 시가 → 종가 (장중)
또한 슈퍼사이클에서 '개장 갭' 연관성이 최근 약해졌는지(recency) 점검.
"""
import sys, io
import numpy as np
import pandas as pd
import gap_model as gm

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass


def main():
    print("미국 신호 수집...")
    us = gm.get_us_returns()
    feats = gm.available_features(us)

    print("\n[1] 타깃별 예측력 (워크포워드)")
    print(f"  {'종목':12s} | {'gap(개장갭)':>16s} | {'daily(일간)':>16s} | {'intraday(장중)':>16s}")
    print("  " + "-" * 72)
    for name, code in gm.KR.items():
        kr = gm.get_kr(code)
        cells = []
        for tgt in ["gap", "daily", "intraday"]:
            d = gm.build_dataset(kr, us, feats, target=tgt)
            wf = gm.walk_forward(d, feats)
            cells.append(f"{wf['hit']:.0f}% / R²{wf['r2']:+.2f}")
        print(f"  {name:12s} | {cells[0]:>16s} | {cells[1]:>16s} | {cells[2]:>16s}")

    print("\n[2] 슈퍼사이클 영향: 개장 갭과 신호 상관 (전반부 vs 최근 120일)")
    print(f"  {'종목':12s} | {'corr(SOX) 전반':>14s} | {'corr(SOX) 최근':>14s} | {'corr(EWY) 전반':>14s} | {'corr(EWY) 최근':>14s}")
    print("  " + "-" * 80)
    for name, code in gm.KR.items():
        kr = gm.get_kr(code)
        d = gm.build_dataset(kr, us, feats, target="gap")
        if len(d) < 200:
            continue
        early, recent = d.iloc[:-120], d.iloc[-120:]
        def c(df, col):
            return np.corrcoef(df[col], df["y"])[0, 1]
        print(f"  {name:12s} | {c(early,'SOX'):>14.2f} | {c(recent,'SOX'):>14.2f} | "
              f"{c(early,'EWY'):>14.2f} | {c(recent,'EWY'):>14.2f}")


if __name__ == "__main__":
    main()
