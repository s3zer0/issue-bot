# 🤖 AI 기반 실시간 이슈 모니터링 및 3단계 환각 탐지 봇

LLM(거대 언어 모델)을 활용하여 특정 주제에 대한 최신 이슈를 실시간으로 검색, 분석하고, LLM-as-a-Judge, RePPL, 자기 일관성(Self-Consistency) 검사를 포함한 3단계 환각 탐지 시스템을 통해 정보의 신뢰도를 교차 검증하는 Discord 봇입니다. PDF 보고서 생성 기능을 포함하여 더욱 전문적인 분석 결과를 제공합니다.

## ✨ 주요 기능

- **지능형 키워드 생성**: 사용자가 입력한 단순 주제(예: "양자 컴퓨팅")를 바탕으로, LLM이 전문적인 검색에 필요한 핵심 키워드, 관련 용어, 맥락 키워드의 세 가지 카테고리로 키워드를 확장 및 생성합니다.
- **실시간 이슈 검색**: 생성된 키워드를 활용하여 Perplexity API (`llama-3.1-sonar-large-128k-online`)를 통해 최신 뉴스, 기술 블로그, 문서 등을 실시간으로 검색하고 요약합니다.
- **3단계 환각 탐지 시스템 (3-Stage Hallucination Detection)**: 검색된 정보의 신뢰도를 높이기 위해 여러 탐지 방법을 통합하여 교차 검증을 수행합니다.
  - **LLM-as-a-Judge 탐지기**: GPT-4o를 평가자(Judge)로 사용하여 내용의 **사실적 정확성(Factual Accuracy)**, **논리적 일관성(Logical Consistency)**, **맥락 관련성(Contextual Relevance)**, **출처 신뢰성(Source Reliability)**을 종합적으로 평가합니다.
  - **RePPL 탐지기**: 내용의 반복성(Repetition), GPT-4o API를 이용한 퍼플렉시티(Perplexity), 그리고 문장 간의 **의미적 엔트로피(Semantic Entropy)**를 분석하여 텍스트의 자연스러움과 논리적 개연성을 측정합니다.
  - **자기 일관성 검사기 (Self-Consistency Checker)**: 동일 주제로 여러 변형 프롬프트를 생성하여 다수의 응답을 얻은 후, 응답 간의 의미적, 내용적 일관성을 분석합니다.
- **자동 재시도 및 키워드 재생성**: 초기 검색 및 분석 결과, 신뢰도 높은 이슈가 부족할 경우, 더 나은 결과를 얻기 위해 자동으로 다른 관점의 키워드를 재생성하여 검색을 재시도합니다.
- **상세 보고서 자동 생성**: 최종적으로 검증된 이슈들을 종합하여 상세한 분석 내용이 담긴 Markdown 및 PDF 형식의 리포트를 생성하고 `reports/` 디렉토리에 타임스탬프가 찍힌 파일로 저장하여 Discord에 제공합니다.
- **향상된 환각 탐지 성능**: Adaptive timeout, Progressive deepening, 병렬 처리 최적화를 통해 더욱 빠르고 정확한 환각 탐지를 수행합니다.
- **다중 LLM 키워드 생성**: OpenAI GPT, Perplexity, Grok 등 여러 LLM을 활용한 다각도 키워드 생성으로 더욱 포괄적인 검색을 수행합니다.

## 🛠️ 기술 스택

- **언어**: Python 3.12
- **Discord 봇**: `discord.py`
- **LLM API**:
  - **OpenAI (gpt-4o)**: 키워드 생성, Perplexity 계산, LLM-as-a-Judge 평가
  - **Perplexity (llama-3.1-sonar-large-128k-online)**: 실시간 이슈 검색
- **NLP & 데이터 처리**:
  - `sentence-transformers`: 문장 임베딩 및 의미 유사도 분석
  - `scikit-learn`: 코사인 유사도 계산
  - `numpy`: 데이터 처리
- **PDF 생성**: `reportlab` - 전문적인 PDF 보고서 생성
- **HTTP 통신**: `httpx`
- **설정 및 로깅**: `python-dotenv`, `loguru`
- **테스트**: `pytest`, `pytest-asyncio`

## 🚀 시작하기

### 1. 사전 요구사항

- Python 3.10 이상
- Git

### 2. 설치 및 설정

#### 프로젝트 클론

```bash
git clone [저장소_URL]
cd issue-bot
```

#### 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\activate  # Windows
```

#### 의존성 라이브러리 설치

```bash
pip install -r requirements.txt
```

#### `.env` 파일 설정

프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 내용을 실제 값으로 채워넣으세요.

```env
# Discord Bot
DISCORD_BOT_TOKEN=your_discord_bot_token_here

# LLM APIs
OPENAI_API_KEY=your_openai_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# Settings (선택 사항)
DEVELOPMENT_MODE=true
LOG_LEVEL=INFO
OPENAI_MODEL=gpt-4o
```

### 3. 실행 방법

#### Discord 봇 실행

봇을 온라인 상태로 만듭니다.

```bash
python src/bot/bot.py
```

#### 자동 테스트 스크립트 실행

Discord 봇을 실행하지 않고, 터미널에서 키워드 생성부터 보고서 생성까지의 전체 흐름을 테스트합니다. `auto_test.py` 파일 내의 `TEST_TOPIC`을 수정하여 다른 주제로 테스트할 수 있습니다.

```bash
python auto_test.py
```

#### 단위/통합 테스트 실행

프로젝트의 모든 테스트 코드를 실행하여 각 모듈의 안정성을 검증합니다.

```bash
pytest
```

## 💬 사용법 (Discord 명령어)

- `/monitor [주제] [기간]`: 특정 주제에 대한 이슈 모니터링을 시작합니다.
  - **주제** (필수): 분석하고 싶은 주제 (예: "AI 반도체")
  - **기간** (선택): 검색할 기간 (예: "3일", "2주일", 기본값: "1주일")
  - 결과: Markdown 보고서와 PDF 보고서가 자동으로 생성되어 Discord에 업로드됩니다.
- `/status`: 봇의 현재 API 키 설정 상태와 활성화된 단계를 확인합니다.
- `/help`: 봇의 사용법과 명령어 목록을 보여줍니다.

## 📊 보고서 형식

### Markdown 보고서
- 파일명: `report_[주제]_[날짜]_[시간]_validated.md`
- Discord 채팅창에서 바로 확인 가능한 텍스트 형식
- 이슈별 상세 분석 및 환각 탐지 점수 포함

### PDF 보고서  
- 파일명: `report_[주제]_[날짜]_[시간]_enhanced.pdf`
- 전문적인 디자인의 시각화된 보고서
- LLM을 활용한 향상된 분석 내용
- 한글 폰트 지원 (NotoSansKR)

## 📁 프로젝트 구조

```
issue-bot/
├── .venv/                          # 가상환경
├── Fonts/                          # PDF 생성용 한글 폰트
│   ├── NotoSans-VariableFont_wdth,wght.ttf
│   └── NotoSansKR-VariableFont_wght.ttf
├── logs/                           # 실행 로그 파일 저장 (bot.log, error.log)
├── reports/                        # 생성된 Markdown 및 PDF 보고서 저장
├── scripts/                        # 유틸리티 스크립트
│   ├── demo_prompt_generation.py  # 프롬프트 생성 데모
│   └── generate_prompt.py         # 프롬프트 생성 스크립트
├── src/                            # 소스 코드
│   ├── bot/                       # Discord 봇 관련
│   │   ├── __init__.py
│   │   └── bot.py                # Discord 봇 메인 로직
│   ├── clients/                   # 외부 API 클라이언트
│   │   └── perplexity_client.py
│   ├── detection/                 # 탐지 관련 모듈
│   │   ├── __init__.py
│   │   ├── hallucination_detector.py
│   │   └── keyword_generator.py
│   ├── hallucination_detection/   # 환각 탐지 로직 패키지
│   │   ├── base.py               # 탐지기 기본 추상 클래스
│   │   ├── consistency_checker.py # 자기 일관성 검사기
│   │   ├── enhanced_reporting.py  # 향상된 보고서 생성
│   │   ├── enhanced_reporting_with_pdf.py # PDF 포함 보고서
│   │   ├── enhanced_searcher.py   # 여러 탐지기를 조율하는 메인 검색기
│   │   ├── llm_judge.py          # LLM-as-a-Judge 탐지기
│   │   ├── models.py             # 환각 탐지 관련 데이터 모델
│   │   ├── reppl_detector.py     # RePPL 탐지기
│   │   └── threshold_manager.py   # 임계값 관리
│   ├── keyword_generation/        # 키워드 생성 패키지
│   │   ├── __init__.py
│   │   ├── base.py               # 키워드 생성 기본 클래스
│   │   ├── extractors/           # LLM별 키워드 추출기
│   │   │   ├── __init__.py
│   │   │   ├── gpt_extractor.py
│   │   │   ├── grok_extractor.py
│   │   │   └── perplexity_extractor.py
│   │   ├── manager.py            # 키워드 생성 관리자
│   │   └── similarity.py         # 유사도 계산
│   ├── reporting/                # 보고서 생성
│   │   ├── __init__.py
│   │   ├── pdf_report_generator.py # PDF 보고서 생성
│   │   └── reporting.py          # Markdown 보고서 생성
│   ├── search/                   # 검색 관련
│   │   ├── __init__.py
│   │   └── issue_searcher.py     # 이슈 검색 및 분석
│   ├── utils/                    # 유틸리티
│   │   ├── __init__.py
│   │   ├── project_analyzer.py   # 프로젝트 분석
│   │   ├── prompt_generator.py   # 프롬프트 생성
│   │   └── prompt_templates.py   # 프롬프트 템플릿
│   ├── config.py                 # 환경 변수 및 설정 관리
│   └── models.py                 # 프로젝트 전역 데이터 모델
├── tests/                        # Pytest 테스트 코드
│   ├── clients/
│   │   └── test_perplexity_client.py
│   ├── test_*.py                # 각 모듈별 단위/통합 테스트
│   └── testconf.py              # 테스트 설정
├── .env                         # (직접 생성) API 키 등 비밀 정보
├── .gitignore
├── auto_test.py                 # 전체 기능 자동 테스트 스크립트
├── pytest.ini                   # Pytest 설정 파일
├── requirements.txt             # Python 의존성 목록
├── test_fixes.py               # 테스트 수정 스크립트
└── README.md                    # 프로젝트 설명 (이 파일)
```
