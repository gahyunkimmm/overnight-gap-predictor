# -*- coding: utf-8 -*-
"""
app.py — 야간 신호 기반 개장 갭 예측 대시보드 (Streamlit)

실행:
    streamlit run app.py
브라우저에서 http://localhost:8501 자동 오픈.
"""
from datetime import datetime

import pandas as pd
import streamlit as st

import gap_model as gm

st.set_page_config(page_title="개장 갭 예측", page_icon="📈", layout="wide")


@st.cache_data(ttl=900)  # 15분 캐시
def load():
    return gm.predict_all()


st.title("📈 야간 신호 기반 한국 반도체주 개장 갭 예측")
st.caption(
    "어젯밤 미국 반도체 시장(MU·SOX·EWY) 신호로 다음 한국 거래일의 개장 갭을 추정합니다. "
    "삼성전자·SK하이닉스는 미국 ADR이 없어 상관 자산을 신호로 사용합니다."
)

col_btn, col_info = st.columns([1, 4])
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

# 신호 요약
st.subheader("어젯밤 신호 (미국 종가 등락률)")
sig_cols = st.columns(len(gm.FEATURES))
labels = {"MU": "마이크론", "SOX": "필라델피아반도체", "EWY": "한국 ETF", "USDKRW": "원/달러"}
for c, feat in zip(sig_cols, gm.FEATURES):
    c.metric(labels.get(feat, feat), f"{latest[feat]:+.2f}%")

st.divider()

# 종목별 예측
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
            f"모델 신뢰도: OOS R²={r['r2_out']:.2f} · 방향적중 {r['hit']:.0f}% · 표본 {r['n']}일"
        )
        # 최근 30일 실제 갭 vs 모델 적합치
        chart_df = r["recent"].rename(columns={"gap": "실제 갭", "fitted": "모델 추정"})
        st.line_chart(chart_df, height=220)

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
