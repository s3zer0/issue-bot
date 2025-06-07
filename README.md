# 🤖 AI 기반 이슈 모니터링 및 환각 탐지 봇

LLM(거대 언어 모델)을 활용하여 특정 주제에 대한 최신 이슈를 실시간으로 검색, 분석하고, **RePPL 방법론에 기반한 환각 탐지(Hallucination Detection) 시스템**을 통해 정보의 신뢰도를 검증하는 Discord 봇입니다.

## ✨ 주요 기능

-   **AI 기반 키워드 생성**: 사용자가 입력한 주제(예: "양자 컴퓨팅")를 바탕으로 심층적인 검색에 필요한 핵심 키워드, 관련 용어 등을 자동으로 생성합니다.
-   **실시간 이슈 검색**: 생성된 키워드를 활용하여 Perplexity API를 통해 최신 뉴스, 기술 블로그, 문서 등을 검색하고 요약합니다.
-   **환각 탐지 (RePPL)**: 검색된 정보의 신뢰도를 높이기 위해 RePPL(Repetition as Pre-Perplexity) 이론에 기반한 검증 시스템을 도입했습니다.
    -   **반복성(Repetition)**: 내용의 불필요한 반복을 분석합니다.
    -   **퍼플렉시티(Perplexity)**: GPT-4o API를 이용해 문장의 자연스러움과 논리적 개연성을 측정합니다.
    -   **의미적 엔트로피(Semantic Entropy)**: 문장 간의 의미적 다양성을 분석하여 내용의 풍부함을 평가합니다.
-   **자동 재시도 및 키워드 재생성**: 환각 탐지 결과 신뢰도가 낮은 경우, 더 나은 결과를 얻기 위해 자동으로 다른 관점의 키워드를 재생성하여 검색을 재시도합니다.
-   **상세 보고서 자동 생성**: 최종적으로 검증된 이슈들을 종합하여 상세한 분석 내용이 담긴 Markdown 형식의 리포트를 생성하고 Discord에 파일로 제공합니다.

## 🛠️ 기술 스택

-   **언어**: Python 3.12
-   **Discord 봇**: `discord.py`
-   **LLM API**: `OpenAI (gpt-4o)`, `Perplexity (llama-3.1-sonar-large-128k-online)`
-   **NLP & 데이터 처리**: `sentence-transformers`, `scikit-learn`, `numpy`
-   **HTTP 통신**: `httpx`
-   **설정 및 로깅**: `python-dotenv`, `loguru`
-   **테스트**: `pytest`, `pytest-asyncio`

## 🚀 시작하기

### 1. 사전 요구사항

-   Python 3.10 이상
-   Git

### 2. 설치 및 설정

1.  **프로젝트 클론**
    ```bash
    git clone [저장소_URL]
    cd issue-bot
    ```

2.  **가상환경 생성 및 활성화**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    # .\.venv\Scripts\activate  # Windows
    ```

3.  **의존성 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

4.  **.env 파일 설정**
    프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 아래 내용을 채워넣으세요. `.env.example` 파일을 복사해서 사용해도 됩니다.

    ```dotenv
    # Discord Bot
    DISCORD_BOT_TOKEN=your_discord_bot_token_here

    # LLM APIs
    OPENAI_API_KEY=your_openai_api_key_here
    PERPLEXITY_API_KEY=your_perplexity_api_key_here

    # Settings
    DEVELOPMENT_MODE=true
    LOG_LEVEL=INFO
    ```

### 3. 실행 방법

#### Discord 봇 실행
봇을 온라인 상태로 만듭니다.
```bash
python src/bot.py
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

-   `/monitor [주제] [기간]`: 특정 주제에 대한 이슈 모니터링을 시작합니다.
    -   `주제` (필수): 분석하고 싶은 주제 (예: "AI 반도체")
    -   `기간` (선택): 검색할 기간 (예: "3일", "2주일", 기본값: "1주일")
-   `/status`: 봇의 현재 설정 상태와 활성화된 단계를 확인합니다.
-   `/help`: 봇의 사용법과 명령어 목록을 보여줍니다.

## 📁 프로젝트 구조

```
issue-bot/
├── .venv/
├── logs/                 # 실행 로그 파일 저장
├── reports/              # 생성된 Markdown 보고서 저장
├── src/                  # 소스 코드
│   ├── clients/          # 외부 API 클라이언트
│   │   └── perplexity_client.py
│   ├── __init__.py
│   ├── bot.py            # Discord 봇 메인 로직
│   ├── config.py         # 환경 변수 및 설정 관리
│   ├── hallucination_detector.py # RePPL 환각 탐지 로직
│   ├── issue_searcher.py # 이슈 검색 및 분석 로직
│   ├── keyword_generator.py # 키워드 생성 로직
│   ├── models.py         # 데이터 클래스 (구조체) 정의
│   └── reporting.py      # 보고서 생성 및 파일 저장 로직
├── tests/                # Pytest 테스트 코드
│   ├── __init__.py
│   ├── conftest.py       # 공통 테스트 픽스처
│   └── ... (각 모듈별 테스트 파일)
├── .env                  # (직접 생성) API 키 등 비밀 정보
├── .env.example          # .env 파일 샘플
├── .gitignore
├── auto_test.py          # 전체 기능 자동 테스트 스크립트
├── pytest.ini            # Pytest 설정 파일
├── requirements.txt      # Python 의존성 목록
└── README.md             # 프로젝트 설명
```