# 프로젝트 컨텍스트 (Claude 인수인계용)

## 한 줄 요약
어젯밤 미국 시장 신호(반도체·지수·변동성·환율 9종)로 **다음 한국 거래일의 개장 갭**을
예측하는 검증된 회귀 모델 + Streamlit 대시보드. 핵심 가치는 **지적 정직성**
(갭은 잘 맞히지만 그걸로 수익은 못 낸다는 걸 데이터로 증명).

## 작업 스타일 (사용자 선호)
- 직접적·구조화된 답변, 군더더기 없이. AI스러운 문체 지양.
- **검증된 것만 반영** — 아이디어는 experiment_*.py로 실측 후 채택/기각. 음성 결과도 기록으로 보존.
- 한국어 직무문서는 명사형 종결.
- 변경은 항상 테스트 + 로컬 스모크(streamlit HTTP 200) 후 커밋·푸시.

## 아키텍처
- `gap_model.py` — 공용 로직(데이터 수집·릿지·워크포워드·build_dataset·predict_all·realized_accuracy)
- `app.py` — Streamlit 대시보드 (배포본)
- `overnight_gap_predict.py` — CLI, 매일 예측을 predictions_log.csv에 누적
- `test_model.py` — 네트워크 불필요 단위테스트(시점누설/릿지/워크포워드)
- `experiment*.py` — 의사결정 근거 기록(신호선택/금리/종목/타깃)
- `.github/workflows/daily-predict.yml` — 평일 07:00 KST 자동 예측+로그 커밋(테스트 포함)

## 핵심 설계 결정 (왜 이렇게 했나)
- 대상 종목: 삼성전자·SK하이닉스·삼성전기·**한미반도체** (반도체 밸류체인).
  현대차/자동차·플랫폼은 미국 반도체 신호와 연동 약해 제외(experiment_stocks.py).
- 신호 9종(MU/SOX/EWY/USDKRW/NVDA/TSM/NASDAQ/VIX/SP500): 확장 시 적중률↑(experiment.py).
- 모델: 표준화 + 릿지(λ=10), 다중공선성 안정화.
- 검증: 70/30이 아니라 **확장창 워크포워드**.
- 금리 신호: 추가해도 개선 없어 미채택(experiment_rates.py).
- 예측력 3분해: 갭(72~75%) > 일간 종가→종가(56~64%) > 장중(≈50%, 무예측력).
- 슈퍼사이클에도 갭↔신호 상관 약화 아님(오히려 강화, experiment_targets.py).
- 차트는 in-sample 아님 — walk_forward_series로 '답 안 본' 사전 추정.
- cache_data는 gm.predict_all 내부변경 감지 못함 → app.py의 `CACHE_VERSION`을
  인자로 넘겨 스키마 변경 시 캐시 무효화. **결과 dict 키를 바꾸면 CACHE_VERSION을 올릴 것.**

## 실행/검증
```bash
pip install -r requirements.txt
python test_model.py                 # 단위테스트
python overnight_gap_predict.py      # CLI 예측
streamlit run app.py                 # 대시보드(localhost:8501)
```
환경: Python 3.12+ 권장(3.14 로컬에선 scipy/sklearn 깨져서 numpy만 사용). Streamlit Cloud는 3.12.

## 배포/자동화 현황
- Streamlit Community Cloud 배포됨(공개). main 푸시 시 자동 재배포.
- GitHub Actions가 평일 매일 예측을 predictions_log.csv에 누적 커밋.
- 커밋 이메일은 익명(gahyunkimmm@users.noreply.github.com)으로 정리됨.

## 다음 할 일 (로드맵)
- [ ] 실측 정확도 데이터 2~4주 누적 후 대시보드 패널 표시(데이터 없으면 자동 숨김 상태).
- [ ] 그 시점에 LinkedIn 공개. 앵글: "예측기를 만들고, 이걸로 돈은 못 번다는 걸 증명".
- [ ] (공개 전) README 상단 배포링크+스크린샷, "이건 무엇/무엇이 아닌지" 한 줄.
- [ ] (선택) 이메일/슬랙 알림, 예측이력 다운로드.

## 한계 (반드시 유지할 정직성)
개장 갭은 개장가에 이미 반영되어 일반 투자자가 취할 수 없음. 투자권유 아님(교육·연구용).
