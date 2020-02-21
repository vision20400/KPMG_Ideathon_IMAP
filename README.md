# KPMG_Ideathon_IMAP
> 지금도 전 세계의 수많은 기업들은 누적된 대량의 아날로그-디지털 형식으로 섞인 문서들을 적절하게 관리하지 못하여 소중한 인적, 물적 자원을 낭비하고 있습니다. 기업이 본연의 영업활동에 역량을 집중하기 위해서는 문서의 생성, 관리를 자동화하고 효율적으로 관리할 수 있어야 합니다.

> 우리의 서비스는 마이크로소프트의 Azure를 활용하여, 회의의 시작과 동시에 실시간으로 화자별,시간별, 내용 등이 체계적으로 분류된 디지털 문서를 작성합니다. 또한, 해당 문서들의 내용을 자연어처리 LDA기법을 통해 과학적으로 분석하여 원하는 주제를 손쉽게 처리할 수 있도록 전사적 문서관리 솔루션을 제공합니다.

## 프로젝트 개요 
![flow chart](https://user-images.githubusercontent.com/41162249/75070619-f5e14e80-5536-11ea-97ae-a25b16e3f8ad.JPG)
- 회의 내용을 별다른 타이핑없이 웹 사이트에서 녹음하여 STT(Speech to text) 기능을 통해 곧바로 텍스트파일로 변환하여 Database에 추가한다.

- 쌓여 있는 수많은 데이터(ex. 회의록/ e-mail)를 빠르고 쉽게 분석하여, 직관적으로 모델링한다. 분석하고자 하는 데이터는 LDA(Latent Dirichlet Allocatio)과정을 거쳐 버블. 그래프 등 다양한 형태로 모델링되어 보여으로써 사용자가 보다 빠르게 데이터를 해석 / 분석할 수 있도록 도와준다.

- 새로운 데이터가 기존의 데이터들과 어떠한 차이를 보이는가에 대한 분석을 필요로 할 때, KNN(K-Nearest Neighbor) 머신 러닝을 통해 쉽게 비교 해석할 수 있다.

## 사용한 공공 데이터 & API & 플랫폼..
> Data : [amazon Annual reports, proxies and shareholder letters](https://ir.aboutamazon.com/annual-reports)

> Data : [Microsoft Annual reports](https://www.microsoft.com/en-us/Investor/annual-reports.aspx)


> API : azure.cognitiveservices.speech (Microsoft)


> Azuer, MySql, AWS server, Jupyter, Flask



## Source

  STT

  LDA 
 
## Team member
>Olga Chernyaeva  김수민  손창영  이세진  박재성
