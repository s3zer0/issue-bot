"""
설정 관리 모듈.

이 모듈은 .env 파일에서 환경 변수를 로드하고,
프로젝트 전반에서 사용될 설정 값들에 대한 접근자(getter)를 제공합니다.
API 키와 같은 민감한 정보나 애플리케이션의 동작 모드를 관리합니다.
"""

import os
import sys
from typing import Optional, Dict
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
            "# LLM APIs\nOPENAI_API_KEY=your_key_here\nPERPLEXITY_API_KEY=your_key_here\n\n"
            "# Settings\nDEVELOPMENT_MODE=true\nLOG_LEVEL=INFO\n"
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
            # 환경 변수에서 값을 가져와 int로 변환, 기본값은 1500
            return int(os.getenv('OPENAI_MAX_TOKENS', 1500))
        except (ValueError, TypeError):
            # 변환 실패 시 경고 로그 출력 후 기본값 반환
            logger.warning("OPENAI_MAX_TOKENS 값이 잘못되어 기본값(1500)을 사용합니다.")
            return 1500

    def get_keyword_generation_timeout(self) -> int:
        """키워드 생성 시 타임아웃(초) 설정 값을 반환합니다."""
        try:
            # 환경 변수에서 값을 가져와 int로 변환, 기본값은 30
            return int(os.getenv('KEYWORD_GENERATION_TIMEOUT', 30))
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