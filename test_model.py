# -*- coding: utf-8 -*-
"""
네트워크 불필요한 단위 테스트.
    python test_model.py
"""
import sys
import io

import numpy as np
import pandas as pd

import gap_model as gm

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass


def test_no_lookahead():
    """build_dataset은 한국 거래일 d에 '엄격히 이전(<d)'의 미국 신호만 매칭해야 한다."""
    d = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    us = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=d[:3])
    # 한국 거래일: 1/2, 1/3, 1/4  → 직전 미국 신호 10, 20, 30 이어야 함
    kr = pd.DataFrame(
        {"Open": [101, 102, 103, 104], "Close": [100, 101, 102, 103]}, index=d
    )
    ds = gm.build_dataset(kr, us, ["A"], target="gap")
    # 1/1은 prev_close가 없어 제외됨 → 남는 날의 신호값 검증
    assert list(ds["A"].values) == [10.0, 20.0, 30.0], ds["A"].values
    # 같은 날(=d) 신호가 새지 않았는지: 1/4 행의 A는 30(=1/3), 40 아님
    print("✓ test_no_lookahead")


def test_ridge_recovers_signal():
    """릿지가 선형 관계를 복원해 강한 예측력을 보여야 한다."""
    rng = np.random.default_rng(0)
    n, p = 600, 3
    X = rng.normal(size=(n, p))
    true = np.array([1.5, -2.0, 0.5])
    y = X @ true + rng.normal(scale=0.3, size=n)
    cut = 400
    beta, mu, sd = gm.ridge_fit(X[:cut], y[:cut], lam=1.0)
    yhat = gm.ridge_predict(beta, mu, sd, X[cut:])
    yte = y[cut:]
    r2 = 1 - np.sum((yte - yhat) ** 2) / np.sum((yte - yte.mean()) ** 2)
    assert r2 > 0.8, r2
    print(f"✓ test_ridge_recovers_signal (R²={r2:.3f})")


def test_walk_forward_runs():
    """워크포워드가 합리적 범위의 지표를 반환해야 한다."""
    rng = np.random.default_rng(1)
    n = 300
    A = rng.normal(size=n)
    y = 1.2 * A + rng.normal(scale=0.5, size=n)
    data = pd.DataFrame({"y": y, "A": A})
    out = gm.walk_forward(data, ["A"], start_frac=0.5)
    assert out["n"] > 0 and 0 <= out["hit"] <= 100 and out["r2"] > 0.5, out
    print(f"✓ test_walk_forward_runs ({out})")


if __name__ == "__main__":
    test_no_lookahead()
    test_ridge_recovers_signal()
    test_walk_forward_runs()
    print("\n전체 테스트 통과 ✅")
