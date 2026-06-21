# -*- coding: utf-8 -*-
"""
app.py — 야간 신호 기반 개장 갭 예측 대시보드 (Streamlit)

실행:
    streamlit run app.py
브라우저에서 http://localhost:8501 자동 오픈.
"""
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import gap_model as gm

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_log.csv")

st.set_page_config(page_title="개장 갭 예측", page_icon="📈", layout="wide")


@st.cache_data(ttl=900)  # 15분 캐시
def load():
    return gm.predict_all()


@st.cache_data(ttl=900)
def load_accuracy():
    return gm.realized_accuracy(LOG_PATH)


st.title("📈 야간 신호 기반 한국 반도체·대형주 개장 갭 예측")
st.caption(
    "어젯밤 미국 시장 신호(반도체·지수·변동성·환율) 9종으로 다음 한국 거래일의 개장 갭을 추정합니다. "
    "삼성전자·SK하이닉스·현대차는 미국 ADR이 없어 상관 자산을 신호로 사용합니다. "
    "모델: 표준화 + 릿지 회귀."
)

col_btn, _ = st.columns([1, 4])
with col_btn:
    if st.button("🔄 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    latest, latest_date, results = load()
except Exception as e:
    st.error(f"데이터 로드 실패: {e}")
    st.stop()

st.success(f"최신 미국 신호일: **{latest_date}**  ·  조회시각 {datetime.now():%Y-%m-%d %H:%M}")

# ---------------- 신호 요약 (9종, 5개씩 두 줄) ----------------
st.subheader("어젯밤 신호 (미국 종가 등락률)")
feats = gm.FEATURES
for i in range(0, len(feats), 5):
    chunk = feats[i:i + 5]
    cols = st.columns(len(chunk))
    for c, f in zip(cols, chunk):
        c.metric(gm.SIGNAL_LABELS.get(f, f), f"{latest[f]:+.2f}%")

st.divider()

# ---------------- 종목별 예측 ----------------
st.subheader("오늘 개장 갭 예측")
cards = st.columns(len(results))
for c, r in zip(cards, results):
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
            f"모델 신뢰도(백테스트): OOS R²={r['r2_out']:.2f} · 방향적중 {r['hit']:.0f}% · 표본 {r['n']}일"
        )
        chart_df = r["recent"].rename(columns={"gap": "실제 갭", "fitted": "모델 추정"})
        st.line_chart(chart_df, height=220)

st.divider()

# ---------------- 사후 실측 정확도 ----------------
st.subheader("📊 실측 정확도 (예측 로그 vs 실제 개장 갭)")
summary, detail = load_accuracy()
if summary is None:
    st.info(
        "아직 평가할 누적 예측이 없습니다. GitHub Actions가 매일 예측을 기록하면 "
        "실제 개장 갭과 대조한 실측 적중률이 여기에 표시됩니다. "
        "(위 '모델 신뢰도'는 과거 데이터 백테스트 수치입니다.)"
    )
else:
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
