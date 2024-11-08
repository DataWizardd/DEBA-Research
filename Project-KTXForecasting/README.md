## 🏢 연구실 소개
- 김경원 교수 소개 (<a href="https://sites.google.com/view/thekimk" target="_blank"><img src="https://img.shields.io/badge/Homepage-4285F4?style=flat-square&logo=Google&logoColor=white"/></a> <a href="https://scholar.google.com/citations?hl=ko&user=nHPe-4UAAAAJ&view_op=list_works&sortby=pubdate" target="_blank"><img src="https://img.shields.io/badge/Google Scholar-4285F4?style=flat-square&logo=Google Scholar&logoColor=white"/></a> <a href="https://www.youtube.com/channel/UCEYxJNI5dhnn_CdC9BEWTuA" target="_blank"><img src="https://img.shields.io/badge/YouTube-FF0000?style=flat-square&logo=YouTube&logoColor=white"/></a> <a href="https://github.com/thekimk" target="_blank"><img src="https://img.shields.io/badge/Github-181717?style=flat-square&logo=Github&logoColor=white"/></a>)
- DEBA 연구실은 디지털경제의 표준이 되어가고 있는 `빅데이터와 머신/딥러닝이란 인공지능 이론을 활용`하여,     
`경제산업 분야의 문제`를 `데이터`를 기반으로 `정량적으로 분석하고 의사결정`함으로써 `전략적으로 문제를 해결`하기 위한 방법을 연구합니다.

---

### ❓KTX 승차수요 단기예측 분석
- **데이터:** 전체 6개 파일 중 `4개의 일별 데이터를 월별로 집계`한 것
- **데이터기간:** `2015년1월~2024년3월`
> - **Training:** `2015년1월~2023년3월`
> - **Validation:** `2023년4월~2024년3월`
> - **Test:** `2024년4월~2025년12월`
- **분석노선:** `전체/주말/주중 경부선, 경전선, 동해선, 전라선, 호남선 15종`
- **분석대상(Y):** `월별 일평균 승차인원수`
- **활용변수(X):** `시간정보/주말갯수/주중갯수/공유일/경제환경/이벤트 등` 활용
- **요약:**
<img src="https://github.com/user-attachments/assets/8bc029b3-394c-4fd6-9796-b560b8c51213" width="70%">

- **활용 알고리즘 후보:**
<img src="https://github.com/user-attachments/assets/9b5063b4-8743-4bbb-b49a-2c4ae527ad21" width="40%">

---

### 💡KTX 단기예측 결과

**(1) 모델링 검증(Validation):**

<img src="https://github.com/user-attachments/assets/64fd3b6f-3d53-4381-ae2e-490998058a83" width="70%">
<img src="https://github.com/user-attachments/assets/d824133c-dd40-4f92-99fa-4bad6dd63aaf" width="70%">

**(2) 최적 성능 예측(Test):**

<img src="https://github.com/user-attachments/assets/50823188-3108-4e93-8e5e-2ee9ff7c255a" width="70%">
