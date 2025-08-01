"""
설정 관리 모듈.

이 모듈은 .env 파일에서 환경 변수를 로드하고,
프로젝트 전반에서 사용될 설정 값들에 대한 접근자(getter)를 제공합니다.
API 키와 같은 민감한 정보나 애플리케이션의 동작 모드를 관리합니다.
"""

import os
import sys
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger


class Config:
    """설정 관리 클래스.

    프로젝트의 모든 설정을 관리하며, .env 파일 로딩, 경로 설정,
    단계별 기능 활성화 여부 검증 등의 역할을 수행합니다.
    이 클래스의 인스턴스인 `config`가 전역적으로 사용됩니다.
    """

    def __init__(self):
        """Config 인스턴스를 초기화하고 환경 설정을 수행합니다."""
        # 로거 초기화
        self.logger = logger
        # 환경 설정 및 .env 파일 로드를 순차적으로 수행
        self._setup_environment()
        self._load_env_file()

    def _setup_environment(self):
        """프로젝트 루트를 기준으로 Python 경로를 설정합니다.

        `src` 폴더를 sys.path에 추가하여, 어떤 위치에서 스크립트를 실행하더라도
        `from src...` 형태의 절대 경로 임포트가 가능하도록 합니다.
        """
        try:
            # 현재 파일의 위치를 기준으로 프로젝트 루트 디렉토리 계산
            project_root = Path(__file__).resolve().parent.parent
            src_path = project_root / 'src'
            # src 경로가 존재하고 sys.path에 없으면 추가
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            # 프로젝트 루트를 환경 변수로 설정
            os.environ['PROJECT_ROOT'] = str(project_root)
        except Exception as e:
            # 환경 설정 실패 시 경고 로그 출력
            logger.warning(f"환경 설정 중 오류 발생: {e}")

    def _load_env_file(self):
        """'.env' 파일을 찾아 환경 변수를 로드합니다. 파일이 없으면 샘플을 생성합니다."""
        # 프로젝트 루트에서 .env 파일 경로 설정
        project_root = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
        env_path = project_root / '.env'
        # .env 파일 로드 시도
        if not load_dotenv(dotenv_path=env_path):
            # 로드 실패 시 .env.example 샘플 파일 생성
            logger.warning(f".env 파일을 찾을 수 없어, .env.example 샘플 파일을 생성합니다.")
            self._create_sample_env_file(project_root)

    def _create_sample_env_file(self, project_root: Path):
        """사용자가 설정을 쉽게 할 수 있도록 .env.example 파일을 생성합니다."""
        sample_path = project_root / '.env.example'
        # 이미 파일이 존재하면 생성하지 않음
        if sample_path.exists():
            return
        # 샘플 .env 파일의 기본 템플릿 정의
        content = (
            "# Discord Bot\nDISCORD_BOT_TOKEN=your_token_here\n\n"
            "# LLM APIs\nOPENAI_API_KEY=your_key_here\n"
            "PERPLEXITY_API_KEY=your_key_here\n"
            "GROK_API_KEY=your_key_here\n\n"
            "# Settings\nDEVELOPMENT_MODE=true\nLOG_LEVEL=INFO\n"
            "OPENAI_MODEL=gpt-4o\nGROK_MODEL=grok-beta\n"
            "# Thresholds for hallucination detection\n"
            "MIN_CONFIDENCE_THRESHOLD=0.5\n"
            "# API Timeouts\nGROK_TIMEOUT=60\n"
        )
        try:
            # 샘플 파일 생성 및 내용 작성
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"샘플 설정 파일 생성됨: {sample_path}")
        except IOError as e:
            # 파일 생성 실패 시 에러 로그 출력
            logger.error(f"샘플 파일 생성 실패: {e}")

    # --- API 키 Getter ---
    def get_discord_token(self) -> Optional[str]:
        """Discord 봇 토큰을 반환합니다. 플레이스홀더 값은 None으로 처리합니다."""
        token = os.getenv('DISCORD_BOT_TOKEN')
        # 유효한 토큰인지 확인 후 반환
        return token if token and 'your_token_here' not in token else None

    def get_openai_api_key(self) -> Optional[str]:
        """OpenAI API 키를 반환합니다. 플레이스홀더 값은 None으로 처리합니다."""
        api_key = os.getenv('OPENAI_API_KEY')
        # 유효한 API 키인지 확인 후 반환
        return api_key if api_key and 'your_key_here' not in api_key else None

    def get_perplexity_api_key(self) -> Optional[str]:
        """Perplexity API 키를 반환합니다. 플레이스홀더 값은 None으로 처리합니다."""
        api_key = os.getenv('PERPLEXITY_API_KEY')
        # 유효한 API 키인지 확인 후 반환
        return api_key if api_key and 'your_key_here' not in api_key else None

    def get_grok_api_key(self) -> Optional[str]:
        """Grok API 키를 반환합니다. 플레이스홀더 값은 None으로 처리합니다."""
        api_key = os.getenv('GROK_API_KEY')
        # 유효한 API 키인지 확인 후 반환
        return api_key if api_key and 'your_key_here' not in api_key else None

    # --- 설정 Getter ---
    def is_development_mode(self) -> bool:
        """개발 모드 활성화 여부를 반환합니다."""
        # 환경 변수에서 DEVELOPMENT_MODE 값을 확인, 기본값은 False
        return os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'

    def get_log_level(self) -> str:
        """로그 레벨을 반환합니다."""
        # 환경 변수에서 LOG_LEVEL 값을 가져오며, 기본값은 INFO
        return os.getenv('LOG_LEVEL', 'INFO').upper()

    def get_openai_model(self) -> str:
        """OpenAI 모델 이름을 반환합니다."""
        # 환경 변수에서 OPENAI_MODEL 값을 가져오며, 기본값은 gpt-4o
        return os.getenv('OPENAI_MODEL', 'gpt-4o')

    def get_openai_temperature(self) -> float:
        """OpenAI API의 temperature 설정 값을 반환합니다."""
        try:
            # 환경 변수에서 값을 가져와 float로 변환, 기본값은 0.7
            return float(os.getenv('OPENAI_TEMPERATURE', 0.7))
        except (ValueError, TypeError):
            # 변환 실패 시 경고 로그 출력 후 기본값 반환
            logger.warning("OPENAI_TEMPERATURE 값이 잘못되어 기본값(0.7)을 사용합니다.")
            return 0.7

    def get_openai_max_tokens(self) -> int:
        """OpenAI API의 max_tokens 설정 값을 반환합니다."""
        try:
            # 환경 변수에서 값을 가져와 int로 변환, 기본값은 4000
            return int(os.getenv('OPENAI_MAX_TOKENS', 4000))
        except (ValueError, TypeError):
            # 변환 실패 시 경고 로그 출력 후 기본값 반환
            logger.warning("OPENAI_MAX_TOKENS 값이 잘못되어 기본값(4000)을 사용합니다.")
            return 4000

    def get_grok_model(self) -> str:
        """Grok 모델 이름을 반환합니다."""
        # 환경 변수에서 GROK_MODEL 값을 가져오며, 기본값은 grok-beta
        return os.getenv('GROK_MODEL', 'grok-3-lastest')

    def get_grok_timeout(self) -> int:
        """Grok API의 타임아웃 설정 값을 반환합니다."""
        try:
            # 환경 변수에서 값을 가져와 int로 변환, 기본값은 60
            return int(os.getenv('GROK_TIMEOUT', 60))
        except (ValueError, TypeError):
            # 변환 실패 시 경고 로그 출력 후 기본값 반환
            logger.warning("GROK_TIMEOUT 값이 잘못되어 기본값(60)을 사용합니다.")
            return 60

    def get_perplexity_model(self) -> str:
        """Perplexity 모델 이름을 반환합니다."""
        # 환경 변수에서 PERPLEXITY_MODEL 값을 가져오며, 기본값은 llama-3.1-sonar-large-128k-online
        return os.getenv('PERPLEXITY_MODEL', 'llama-3.1-sonar-large-128k-online')
    
    def get_perplexity_max_tokens(self) -> int:
        """Perplexity API의 max_tokens 설정 값을 반환합니다."""
        try:
            # 환경 변수에서 값을 가져와 int로 변환, 기본값은 4000
            tokens = int(os.getenv('PERPLEXITY_MAX_TOKENS', 4000))
            # 음수나 비정상적인 값 체크
            if tokens <= 0:
                logger.warning(f"PERPLEXITY_MAX_TOKENS 값이 유효하지 않음({tokens}). 기본값(4000)을 사용합니다.")
                return 4000
            return tokens
        except (ValueError, TypeError):
            # 변환 실패 시 경고 로그 출력 후 기본값 반환
            logger.warning("PERPLEXITY_MAX_TOKENS 값이 잘못되어 기본값(4000)을 사용합니다.")
            return 4000

    def get_keyword_generation_timeout(self) -> int:
        """키워드 생성 시 타임아웃(초) 설정 값을 반환합니다."""
        try:
            # 환경 변수에서 값을 가져와 int로 변환, 기본값은 300
            return int(os.getenv('KEYWORD_GENERATION_TIMEOUT', 300))
        except (ValueError, TypeError):
            # 변환 실패 시 경고 로그 출력 후 기본값 반환
            logger.warning("KEYWORD_GENERATION_TIMEOUT 값이 잘못되어 기본값(30)을 사용합니다.")
            return 30

    def get_max_retry_count(self) -> int:
        """API 호출 실패 시 최대 재시도 횟수를 반환합니다."""
        try:
            # 환경 변수에서 값을 가져와 int로 변환, 기본값은 3
            return int(os.getenv('MAX_RETRY_COUNT', 3))
        except (ValueError, TypeError):
            # 변환 실패 시 경고 로그 출력 후 기본값 반환
            logger.warning("MAX_RETRY_COUNT 값이 잘못되어 기본값(3)을 사용합니다.")
            return 3

    # --- 범용 Getter (새로 추가된 부분) ---
    def get_env_var(self, key: str, default: Any = None, cast: Optional[Callable] = None) -> Any:
        """
        환경 변수에서 값을 가져옵니다. 기본값 및 타입 캐스팅을 지원합니다.
        `config.get_env_var('MY_VAR', default=10, cast=int)`와 같이 사용합니다.

        Args:
            key (str): 가져올 환경 변수의 이름입니다.
            default (Any, optional): 변수가 없을 때 반환할 기본값입니다. Defaults to None.
            cast (Optional[Callable], optional): 값을 변환할 함수(e.g., int, float). Defaults to None.

        Returns:
            Any: 환경 변수 값. 캐스팅되었거나 기본값이 반환될 수 있습니다.
        """
        value = os.getenv(key)

        if value is None:
            # 환경 변수가 없으면 기본값을 반환
            return default

        if cast:
            try:
                # 캐스팅 함수가 있으면 값 변환 시도
                return cast(value)
            except (ValueError, TypeError):
                # 변환 실패 시 경고 로그를 남기고 기본값을 반환
                logger.warning(
                    f"환경 변수 '{key}'의 값 '{value}'를 '{cast.__name__}' 타입으로 변환할 수 없습니다. "
                    f"기본값인 '{default}'를 사용합니다."
                )
                return default

        # 캐스팅이 없으면 문자열 값 그대로 반환
        return value

    def get(self, key: str, default: Any = None, cast: Optional[Callable] = None) -> Any:
        """
        환경 변수에서 값을 가져옵니다. 기본값 및 타입 캐스팅을 지원합니다.
        `config.get('MY_VAR', default=10, cast=int)`와 같이 사용합니다.

        Args:
            key (str): 가져올 환경 변수의 이름입니다.
            default (Any, optional): 변수가 없을 때 반환할 기본값입니다. Defaults to None.
            cast (Optional[Callable], optional): 값을 변환할 함수(e.g., int, float). Defaults to None.

        Returns:
            Any: 환경 변수 값. 캐스팅되었거나 기본값이 반환될 수 있습니다.
        """
        value = os.getenv(key)

        if value is None:
            # 환경 변수가 없으면 기본값을 반환
            return default

        if cast:
            try:
                # 캐스팅 함수가 있으면 값 변환 시도
                return cast(value)
            except (ValueError, TypeError):
                # 변환 실패 시 경고 로그를 남기고 기본값을 반환
                logger.warning(
                    f"환경 변수 '{key}'의 값 '{value}'를 '{cast.__name__}' 타입으로 변환할 수 없습니다. "
                    f"기본값인 '{default}'를 사용합니다."
                )
                return default

        # 캐스팅이 없으면 문자열 값 그대로 반환
        return value

    # --- 단계 검증 ---
    def validate_stage1_requirements(self) -> bool:
        """1단계(Discord 봇 실행) 요구사항 충족 여부를 검증합니다."""
        # Discord 봇 토큰이 유효한지 확인
        return bool(self.get_discord_token())

    def validate_stage2_requirements(self) -> bool:
        """2단계(키워드 생성) 요구사항 충족 여부를 검증합니다."""
        # 1단계 요구사항 + OpenAI API 키 유효성 확인
        return self.validate_stage1_requirements() and bool(self.get_openai_api_key())

    def validate_stage3_requirements(self) -> bool:
        """3단계(이슈 검색) 요구사항 충족 여부를 검증합니다."""
        # 2단계 요구사항 + Perplexity API 키 유효성 확인
        return self.validate_stage2_requirements() and bool(self.get_perplexity_api_key())

    def get_current_stage(self) -> int:
        """현재 실행 가능한 최고 단계를 반환합니다.

        모든 키가 설정되면 4단계(모든 기능 활성화)로 간주합니다.

        Returns:
            int: 현재 활성화된 최고 단계 (0, 1, 2, 4).
        """
        # 단계별 요구사항을 순차적으로 검증하여 최고 단계 반환
        if self.validate_stage3_requirements(): return 4
        if self.validate_stage2_requirements(): return 2
        if self.validate_stage1_requirements(): return 1
        return 0

    def get_stage_info(self) -> Dict[str, bool]:
        """각 단계별 설정 완료 여부를 딕셔너리로 반환합니다."""
        # 각 단계 및 개발 모드의 상태를 딕셔너리로 반환
        return {
            "stage1_discord": self.validate_stage1_requirements(),
            "stage2_openai": self.validate_stage2_requirements(),
            "stage3_perplexity": self.validate_stage3_requirements(),
            "development_mode": self.is_development_mode()
        }


# 전역 설정 인스턴스: 프로젝트 어디서든 'from src.config import config'로 사용
config = Config()