# -*- coding: utf-8 -*-
"""
실험: 어떤 한국 종목이 '미국 반도체 신호 -> 익일 개장 갭' 로직에 가장 적합한가?
여러 후보를 동일 모델(9신호, 릿지, 워크포워드)로 돌려 적중률/R²로 순위를 매긴다.
"""
import sys, io
import gap_model as gm

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

CANDIDATES = {
    # 반도체 밸류체인
    "삼성전자": "005930", "SK하이닉스": "000660", "삼성전기": "009150",
    "한미반도체": "042700", "DB하이텍": "000990", "리노공업": "058470",
    "이오테크닉스": "039030", "하나마이크론": "067310",
    # 플랫폼/2차전지(대조군)
    "네이버": "035420", "카카오": "035720",
    "LG에너지솔루션": "373220", "삼성SDI": "006400",
    # 자동차(현행)
    "현대차": "005380", "기아": "000270",
}


def main():
    print("미국 신호 수집...")
    us = gm.get_us_returns()
    feats = gm.available_features(us)
    print(f"  신호 {len(feats)}종\n")

    rows = []
    for name, code in CANDIDATES.items():
        try:
            kr = gm.get_kr(code)
            data = gm.build_dataset(kr, us, feats, target="gap")
            wf = gm.walk_forward(data, feats)
            rows.append((name, code, wf["hit"], wf["r2"], wf["mae"], wf["n"]))
        except Exception as e:
            rows.append((name, code, float("nan"), float("nan"), float("nan"), 0))
            print(f"  [실패] {name}({code}): {e}")

    rows.sort(key=lambda r: (r[2] if r[2] == r[2] else -1), reverse=True)  # 적중률 내림차순
    print(f"\n{'순위':>3} {'종목':14s} {'코드':7s} {'적중%':>7s} {'WF R²':>7s} {'MAE':>6s} {'표본':>5s}")
    print("-" * 60)
    for i, (name, code, hit, r2, mae, n) in enumerate(rows, 1):
        print(f"{i:>3} {name:14s} {code:7s} {hit:6.1f}% {r2:7.3f} {mae:6.2f} {n:>5}")


if __name__ == "__main__":
    main()
