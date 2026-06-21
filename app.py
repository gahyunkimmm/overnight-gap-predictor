# -*- coding: utf-8 -*-
"""
app.py — 야간 신호 기반 개장 갭 예측 대시보드 (Streamlit)

실행:
    streamlit run app.py
브라우저에서 http://localhost:8501 자동 오픈.
"""
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

import gap_model as gm

KST = timezone(timedelta(hours=9))  # 한국 표준시 (DST 없음)

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_log.csv")

st.set_page_config(page_title="개장 갭 예측", page_icon="📈", layout="wide")


@st.cache_data(ttl=900)  # 15분 캐시
def load():
    return gm.predict_all()


@st.cache_data(ttl=900)
def load_accuracy():
    return gm.realized_accuracy(LOG_PATH)


st.title("📈 한국 주식 개장 갭 예측")
st.caption("어젯밤 미국 시장 신호 9종으로 다음 한국 거래일의 개장 갭을 추정합니다.")

col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    latest, latest_date, feats, results = load()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

st.success(
    f"최신 미국 신호일: **{latest_date}**  ·  "
    f"조회시각 {datetime.now(KST):%Y-%m-%d %H:%M} (KST, 한국시간)"
)
if len(feats) < len(gm.FEATURES):
    st.warning(f"일부 신호 수집 실패 — 가용 신호 {len(feats)}/{len(gm.FEATURES)}개로 계산했습니다.")

# ---------------- 신호 요약 (모바일 친화 표) ----------------
st.subheader("어젯밤 신호 (미국 종가 등락률)")
sig_df = pd.DataFrame({
    "신호": [f"{gm.SIGNAL_LABELS.get(f, f)} ({gm.US_TICKERS[f]})" for f in feats],
    "등락률": [f"{latest[f]:+.2f}%" for f in feats],
})
st.dataframe(sig_df, hide_index=True, use_container_width=True, height=350)

st.divider()

# ---------------- 종목별 예측 (한 줄에 2개씩) ----------------
st.subheader("오늘 개장 갭 예측")
PER_ROW = 2
for i in range(0, len(results), PER_ROW):
    row = results[i:i + PER_ROW]
    cards = st.columns(PER_ROW)
    for c, r in zip(cards, row):
        with c:
            st.markdown(f"### {r['name']} ({r['code']})")
            direction = "🔺 상승" if r["pred_gap"] > 0 else "🔻 하락"
            st.metric(
                label=f"예상 개장 갭  ·  {direction}",
                value=f"{r['pred_gap']:+.2f}%",
                delta=f"예상 개장가 {r['pred_open']:,.0f}원",
            )
            st.caption(
                f"직전 종가({r['last_close_date']}) {r['last_close']:,.0f}원  →  "
                f"±1σ 범위 {r['lo']:,.0f} ~ {r['hi']:,.0f}원"
            )
            st.caption(
                f"신뢰도(워크포워드): R²={r['r2_wf']:.2f} · 방향적중 {r['hit_wf']:.0f}% · 검증 {r['n']}일"
            )
            chart_df = r["recent"].rename(columns={"gap": "실제 갭", "oos": "예측 갭"})
            st.line_chart(chart_df, height=220)

st.divider()

# ---------------- 갭 vs 장중 예측력 (정직한 비교) ----------------
st.subheader("🔬 개장 갭 vs 장중(개장→종가) 예측력")
st.caption(
    "갭은 개장가에 이미 반영되어 일반 투자자가 취하기 어렵습니다. "
    "실제 수익이 되려면 '개장 후 종가까지'가 예측되어야 하는데, 아래처럼 장중 예측력은 크게 낮습니다."
)
cmp_df = pd.DataFrame({
    "종목": [r["name"] for r in results],
    "갭 적중%": [round(r["hit_wf"]) for r in results],
    "갭 R²": [round(r["r2_wf"], 2) for r in results],
    "장중 적중%": [round(r["intraday_hit"]) for r in results],
    "장중 R²": [round(r["intraday_r2"], 2) for r in results],
})
st.dataframe(cmp_df, hide_index=True, use_container_width=True)

# ---------------- 사후 실측 정확도 (데이터 쌓이면 표시) ----------------
summary, detail = load_accuracy()
if summary is not None:
    st.divider()
    st.subheader("📊 실측 정확도 (예측 로그 vs 실제 개장 갭)")
    sc = st.columns(len(summary))
    for c, (_, row) in zip(sc, summary.iterrows()):
        c.metric(
            f"{row['stock']} 실측 적중률",
            f"{row['hit_rate']:.0f}%",
            delta=f"n={int(row['n'])} · MAE {row['MAE']}%p",
        )
    with st.expander("최근 예측 vs 실제 상세"):
        show = detail.copy()
        show["예측갭%"] = show["pred_gap"].round(2)
        show["실제갭%"] = show["actual_gap"].round(2)
        show["적중"] = show["hit"].map({1: "✅", 0: "❌"})
        st.dataframe(
            show[["target_date", "stock", "예측갭%", "실제갭%", "적중"]].tail(30),
            use_container_width=True, hide_index=True,
        )

st.divider()
with st.expander("⚠️ 이 도구의 한계 (반드시 읽어주세요)"):
    st.markdown(
        """
- 예측 대상은 **개장 갭**이며, 갭은 **개장가 그 자체에 이미 반영되어 열립니다.**
  예측이 맞아도 일반 투자자가 그 갭 수익을 그대로 취하긴 어렵습니다.
- 장중(개장→종가)은 예측력이 급감합니다(거의 랜덤).
- 최근 반도체 슈퍼사이클 데이터에 적합되어 있어, 업황 체제 변화 시 상관관계가 깨질 수 있습니다.
- **투자 권유가 아니며**, 갭 방향·강도를 가늠하는 참고용 분석 도구입니다.
        """
    )
st.caption("교육·연구 목적. 투자 결정의 책임은 사용자에게 있습니다.")
