# 파이선과 오픈소스를 이용한 딥러닝 (180108 ~ 180112, 5일간)
- 이성주 강사 (email : seongjoo@codebasic.io)


## 3 Day

Kernel SVM의 Kernel의 역할을 아래 [링크](https://ratsgo.github.io/machine%20learning/2017/05/30/SVM3/)에서 쉽게 확인 가능하며, 예제는 [여기](http://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html)에서 확인 가능합니다!


### 딴짓
1. 통계학 기초
Mit Open Coursewar
Introduction to Probability and Statistics
MIT에서 제공하는 기초 통계학 강좌이다. 주로 PDF로 구성되어 있으며 중간중간 퀴즈가 제공된다. 통계학의 핵심적인 내용을 잘 정리해둔 강좌로, Assignment와 Exam에 나온 문제들만 익숙하게 풀 수 있도록 공부하면 앞으로 데이터 사이언스를 공부하는데 든든한 버팀목이 될 것이다.

1-4주차는 기본적인 내용이라 통계학에 익숙한 사람은 넘겨도 되고, 5-8주차는 꼭 한번 풀어볼 것을 권한다. 9주차 이후부터는 Bayesian Inference와 NHST(Null Hypothesis Significance Testing), regression 등 심화내용이 나오는데 여유가 없거나 내용을 따라가기 어렵다면 넘겨도 좋다. 추후에 머신러닝 분야를 깊게 공부하고 싶다면 꼭 짚고 넘어 가보길 바란다.



2. 선형대수학 기초
Texas University
Linear Algebra – Foundations to Frontiers
통계학과 마찬가지로 선형대수학은 빼놓을 수 없는 필수 과목이다. 위 강좌는 군더더기 없이 필요한 내용만 담았다.
http://earlybird.ai/data_science_free_curriculum/

4. 검정 및 추정 (확률론)
우리가 데이터를 분석하는 근본적인 이유는 그 데이터를 바탕으로 새로운 정보를 얻기 위해서이다. 이는 가설 설정과 데이터를 통한 가설 검정이라는 비교적 단순한 과정으로 통해 이루어지는데, 이를 체계적으로 학문화 시킨 것이 Statistical Inference 이다.




## 2 Day

### 딴짓
모연 강남지점 랩에 재미난게 등장
http://www.modulabs.co.kr/RL4RWS_free/15898
정말 소름 1초만에 60배 벌수 있었던..
https://coinone.co.kr/talk/notice/333/




## 1 Day




### 파이선 기초
- 파이선 3.6이 나온지 10년이나 흘렀다고, 데이터 분석과 딥러닝을 위해서는 64비트로 가는게 맞음.
- 파이선 아나콘도 설치를 할경우 환경변수 추가하는 부분은 제외하자( 기존의 파이선 바이너리 셋팅의 혼선을 줄이기 위해)
  . 대신에 아나콘다에서 제공하는 Anaconda Prompt를 사용하면 된다.
  . 프로그램에서 where python 하면 아나콘다 내의 파이선이 실행되는 걸 볼 수 있다
  . 아나콘다 4.4.0으로 교육 진행  (이 다음 버젼이 바로 5.0이라 차이는 없고, 패키지 조금 추가됨)

### 주피터
- 장점
 . 마지막으로 실행된 결과의 스냅샷을 볼수 있다
 . 문서와 코드와의 찰떢궁합
 . 쉬운 러닝커브

- 인증과 패스워드 방법이 있으며 패스워드는 jupyter notebook password 같이 셋팅하여 사용가능

- 주피터의 무한 반복 해결
 . 너무 허무... Ctrl + C란다..

- 너무 파이선 기초 천천히 나가는거 아닌감.. 하악..






### 딴짓

-  공유할 것
  . [자정거래](https://blog.naver.com/timeloader/221179095119)
  . [코인별 백테스트 결과](https://blog.naver.com/PostView.nhn?blogId=pjt3591oo&logNo=221178810736&parentCategoryNo=&categoryNo=106&viewDate=&isShowPopularPosts=false&from=postView)
  . [비트코인봇](https://www.ddengle.com/bitcoindeveloper/4686909)

- [부동산 EDA 하는 글 참고](http://wwwhihaho.synology.me/hoon/?cat=10)

https://www.pyhoon.com/




파이선 히로쿠에 올리기
https://cjh5414.github.io/heroku-python/

로컬 에이전트 설치
https://devcenter.heroku.com/articles/heroku-cli#download-and-install

flask 히로쿠에 올리기
http://blog.weirdx.io/post/9008
http://nextir.blogspot.kr/2014/07/heroku-python.html

갭투자 고민해볼것
이런 아파트는
http://land.naver.com/article/articleDetailInfo.nhn?atclNo=1800223408&atclRletTypeCd=A01&rletTypeCd=A01&tradeTypeCd=A1
예1) 뉴타운이나 재개발사업이 주변 주택가격에 미치는 영향
   - 지역 : 왕십리 뉴타운
   - 이슈시점 : 뉴타운 확정 발표시점, 착공시점, 입주시작 시점
   - 통계 : 왕십리 뉴타운 지역 및 주변지역의 아파트/빌라 시세(매매. 전월세) 변화

 예2) 지하철 개통이 주변에 미치는 영향
   - 지역 : 신분당선 XXX역, XXX역
   - 이슈시점 : 개통 확정 발표시점, 개통 후
   - 통계 : XXX역 인근지역의 아파트/빌라 시세(매매. 전월세) 변화
               XXX역에서 대중교통으로 10~15분정도 떨어진 지역의아파트/빌라 시세(매매. 전월세) 변화

금주해야 할 것
