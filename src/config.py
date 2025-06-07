# src/config.py
"""
설정 관리 모듈 (최종 완성본)
"""
import os
import sys
from typing import Optional, Dict
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

class Config:
    """설정 관리 클래스"""
    # --- 기존 코드 (동일) ---
    def __init__(self):
        self._setup_environment()
        self._load_env_file()

    def _setup_environment(self):
        try:
            project_root = Path(__file__).resolve().parent.parent
            src_path = project_root / 'src'
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            os.environ['PROJECT_ROOT'] = str(project_root)
        except Exception as e:
            logger.warning(f"환경 설정 중 오류: {e}")

    def _load_env_file(self):
        project_root = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
        env_path = project_root / '.env'
        if not load_dotenv(dotenv_path=env_path):
            self._create_sample_env_file(project_root)

    def _create_sample_env_file(self, project_root: Path):
        sample_path = project_root / '.env.example'
        if sample_path.exists(): return
        content = (
            "# Discord Bot\nDISCORD_BOT_TOKEN=your_token_here\n\n"
            "# LLM APIs\nOPENAI_API_KEY=your_key_here\nPERPLEXITY_API_KEY=your_key_here\n\n"
            "# Settings\nDEVELOPMENT_MODE=true\nLOG_LEVEL=INFO\n"
        )
        try:
            with open(sample_path, 'w', encoding='utf-8') as f: f.write(content)
            logger.info(f"샘플 설정 파일 생성됨: {sample_path}")
        except IOError as e: logger.error(f"샘플 파일 생성 실패: {e}")

    # --- API 키 Getter (기존 코드와 동일) ---
    def get_discord_token(self) -> Optional[str]:
        token = os.getenv('DISCORD_BOT_TOKEN')
        return token if token and 'your_token_here' not in token else None

    def get_openai_api_key(self) -> Optional[str]:
        api_key = os.getenv('OPENAI_API_KEY')
        return api_key if api_key and 'your_key_here' not in api_key else None

    def get_perplexity_api_key(self) -> Optional[str]:
        api_key = os.getenv('PERPLEXITY_API_KEY')
        return api_key if api_key and 'your_key_here' not in api_key else None

    # --- 설정 Getter (기존 코드와 동일) ---
    def is_development_mode(self) -> bool:
        return os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'

    def get_log_level(self) -> str:
        return os.getenv('LOG_LEVEL', 'INFO').upper()

    def get_openai_model(self) -> str: return os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

    def get_openai_temperature(self) -> float:
        try:
            return float(os.getenv('OPENAI_TEMPERATURE', 0.7))
        except (ValueError, TypeError):
            logger.warning("OPENAI_TEMPERATURE 값이 잘못되어 기본값(0.7)을 사용합니다.")
            return 0.7

    def get_openai_max_tokens(self) -> int:
        try:
            return int(os.getenv('OPENAI_MAX_TOKENS', 1500))
        except (ValueError, TypeError):
            logger.warning("OPENAI_MAX_TOKENS 값이 잘못되어 기본값(1500)을 사용합니다.")
            return 1500

    def get_keyword_generation_timeout(self) -> int:
        try:
            return int(os.getenv('KEYWORD_GENERATION_TIMEOUT', 30))
        except (ValueError, TypeError):
            logger.warning("KEYWORD_GENERATION_TIMEOUT 값이 잘못되어 기본값(30)을 사용합니다.")
            return 30

    def get_max_retry_count(self) -> int:
        try:
            return int(os.getenv('MAX_RETRY_COUNT', 3))
        except (ValueError, TypeError):
            logger.warning("MAX_RETRY_COUNT 값이 잘못되어 기본값(3)을 사용합니다.")
            return 3

    # --- 단계 검증 (기존 코드와 동일) ---
    def validate_stage1_requirements(self) -> bool: return bool(self.get_discord_token())
    def validate_stage2_requirements(self) -> bool: return self.validate_stage1_requirements() and bool(self.get_openai_api_key())
    def validate_stage3_requirements(self) -> bool: return self.validate_stage2_requirements() and bool(self.get_perplexity_api_key())

    # --- 💡 [수정] 현재 단계 확인 로직 ---
    def get_current_stage(self) -> int:
        """현재 실행 가능한 최고 단계를 반환합니다."""
        # 3단계 요구사항(Perplexity 키)까지 모두 만족하면 4단계로 간주
        if self.validate_stage3_requirements(): return 4
        if self.validate_stage2_requirements(): return 2
        if self.validate_stage1_requirements(): return 1
        return 0

    def get_stage_info(self) -> Dict[str, bool]:
        return {
            "stage1_discord": self.validate_stage1_requirements(),
            "stage2_openai": self.validate_stage2_requirements(),
            "stage3_perplexity": self.validate_stage3_requirements(),
            "development_mode": self.is_development_mode()
        }

# 전역 설정 인스턴스
config = Config()