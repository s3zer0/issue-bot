"""
설정 관리 모듈
"""
import os
import sys
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger


class Config:
    """설정 관리 클래스"""

    def __init__(self):
        self._setup_environment()
        self._load_env_file()

    def _setup_environment(self):
        """프로젝트 루트를 기준으로 Python 경로 설정"""
        try:
            # config.py 파일의 위치를 기준으로 프로젝트 루트를 찾습니다.
            # (src/config.py -> project_root)
            project_root = Path(__file__).resolve().parent.parent

            # Python 경로에 src 디렉토리 추가
            src_path = project_root / 'src'
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            os.environ['PROJECT_ROOT'] = str(project_root)
            logger.debug(f"프로젝트 루트: {project_root}")
            logger.debug(f"PYTHONPATH에 '{src_path}' 추가됨")

        except Exception as e:
            logger.warning(f"환경 설정 중 오류: {e}")

    def _load_env_file(self):
        """
        python-dotenv를 사용하여 .env 파일 로드.
        파일이 없으면 샘플 파일을 생성합니다.
        """
        project_root = Path(os.getenv('PROJECT_ROOT', Path.cwd()))
        env_path = project_root / '.env'

        # load_dotenv는 .env 파일을 찾아 환경변수를 로드합니다.
        # 파일이 있으면 True, 없으면 False를 반환합니다.
        if load_dotenv(dotenv_path=env_path):
            logger.debug(f".env 파일 로드 완료: {env_path}")
        else:
            logger.warning(f".env 파일을 찾을 수 없습니다. ({env_path})")
            self._create_sample_env_file(project_root)

    def _create_sample_env_file(self, project_root: Path):
        """샘플 .env 파일 생성"""
        sample_path = project_root / '.env.example'
        if sample_path.exists():
            logger.info(f"이미 {sample_path} 파일이 존재합니다.")
            return

        sample_content = """# Discord 이슈 모니터링 봇 설정 파일

# Discord 봇 토큰 (필수)
DISCORD_BOT_TOKEN=your_discord_bot_token_here

# OpenAI API 키 (키워드 생성용)
OPENAI_API_KEY=your_openai_api_key_here

# Perplexity API 키 (이슈 검색용)
PERPLEXITY_API_KEY=your_perplexity_api_key_here

# 개발 모드 설정
DEVELOPMENT_MODE=true

# 로그 레벨 설정
LOG_LEVEL=INFO

# OpenAI 설정
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1500
KEYWORD_GENERATION_TIMEOUT=30
MAX_RETRY_COUNT=3
"""
        try:
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)
            logger.info(f"샘플 .env 파일 생성됨: {sample_path}")
            logger.info("실제 API 키로 수정한 후 '.env'로 이름을 변경하여 사용하세요.")
        except Exception as e:
            logger.error(f"샘플 .env 파일 생성 실패: {e}")

    # API 키 반환 메서드들
    def get_discord_token(self) -> Optional[str]:
        token = os.getenv('DISCORD_BOT_TOKEN')
        if not token or token == 'your_discord_bot_token_here':
            logger.error("DISCORD_BOT_TOKEN이 설정되지 않았습니다")
            return None
        return token

    def get_openai_api_key(self) -> Optional[str]:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your_openai_api_key_here':
            logger.error("OPENAI_API_KEY가 설정되지 않았습니다")
            return None
        return api_key

    def get_perplexity_api_key(self) -> Optional[str]:
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if not api_key or api_key == 'your_perplexity_api_key_here':
            logger.warning("PERPLEXITY_API_KEY가 설정되지 않았습니다")
            return None
        return api_key

    # 환경 설정 메서드들
    def is_development_mode(self) -> bool:
        return os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'

    def get_log_level(self) -> str:
        return os.getenv('LOG_LEVEL', 'INFO').upper()

    # 단계별 검증 메서드들
    def validate_stage1_requirements(self) -> bool:
        """1단계(Discord 봇) 필수 요구사항 검증"""
        return bool(self.get_discord_token())

    def validate_stage2_requirements(self) -> bool:
        """2단계(키워드 생성) 필수 요구사항 검증"""
        return bool(self.get_discord_token() and self.get_openai_api_key())

    def validate_stage3_requirements(self) -> bool:
        """3단계(이슈 검색) 필수 요구사항 검증"""
        return bool(self.get_discord_token() and self.get_openai_api_key() and self.get_perplexity_api_key())

    def get_current_stage(self) -> int:
        """현재 구현 가능한 최고 단계 반환"""
        if self.validate_stage3_requirements():
            return 4  # 4단계까지 구현됨
        elif self.validate_stage2_requirements():
            return 2
        elif self.validate_stage1_requirements():
            return 1
        else:
            return 0

    def get_stage_info(self) -> dict:
        """현재 단계별 준비 상태 반환"""
        return {
            "stage1_discord": self.validate_stage1_requirements(),
            "stage2_openai": self.validate_stage2_requirements(),
            "stage3_perplexity": self.validate_stage3_requirements(),
            "development_mode": self.is_development_mode(),
            "log_level": self.get_log_level()
        }

    def print_stage_status(self) -> None:
        """단계별 상태를 콘솔에 출력"""
        stage_info = self.get_stage_info()
        current_stage = self.get_current_stage()
        project_root = os.getenv('PROJECT_ROOT', '알 수 없음')

        print("\n🔧 === 설정 상태 확인 ===")
        print(f"현재 실행 가능한 단계: {current_stage}단계")
        print(f"프로젝트 디렉토리: {project_root}")
        print(f"개발 모드: {'ON' if stage_info['development_mode'] else 'OFF'}")
        print(f"로그 레벨: {stage_info['log_level']}")
        print("\n📋 단계별 준비 상태:")
        print(f"  1단계 (Discord 봇): {'✅' if stage_info['stage1_discord'] else '❌'}")
        print(f"  2단계 (키워드 생성): {'✅' if stage_info['stage2_openai'] else '❌'}")
        print(f"  3-4단계 (이슈 검색): {'✅' if stage_info['stage3_perplexity'] else '❌'}")

        if current_stage < 4:
            print(f"\n💡 다음 단계 진행을 위해 필요한 설정:")
            if current_stage < 1:
                print("  • .env 파일에 DISCORD_BOT_TOKEN 설정")
            elif current_stage < 2:
                print("  • .env 파일에 OPENAI_API_KEY 설정")
            elif current_stage < 4:
                print("  • .env 파일에 PERPLEXITY_API_KEY 설정")

        env_path = Path(project_root) / '.env'
        if env_path.exists():
            print(f"\n📄 .env 파일 위치: {env_path}")
        else:
            print(f"\n📄 .env 파일: ❌ 찾을 수 없음 ({env_path})")

        print("========================\n")

    # OpenAI 설정 메서드들
    def get_openai_model(self) -> str:
        return os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

    def get_openai_temperature(self) -> float:
        try:
            return float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        except (ValueError, TypeError):
            logger.warning("OPENAI_TEMPERATURE 값이 잘못되었습니다. 기본값 0.7 사용")
            return 0.7

    def get_openai_max_tokens(self) -> int:
        try:
            return int(os.getenv('OPENAI_MAX_TOKENS', '1500'))
        except (ValueError, TypeError):
            logger.warning("OPENAI_MAX_TOKENS 값이 잘못되었습니다. 기본값 1500 사용")
            return 1500

    def get_keyword_generation_timeout(self) -> int:
        try:
            return int(os.getenv('KEYWORD_GENERATION_TIMEOUT', '30'))
        except (ValueError, TypeError):
            logger.warning("KEYWORD_GENERATION_TIMEOUT 값이 잘못되었습니다. 기본값 30 사용")
            return 30

    def get_max_retry_count(self) -> int:
        try:
            return int(os.getenv('MAX_RETRY_COUNT', '3'))
        except (ValueError, TypeError):
            logger.warning("MAX_RETRY_COUNT 값이 잘못되었습니다. 기본값 3 사용")
            return 3


# 전역 설정 인스턴스
config = Config()


def get_config() -> Config:
    """전역 설정 인스턴스 반환"""
    return config


def is_ready_for_stage(stage: int) -> bool:
    """특정 단계 실행 준비 여부 확인"""
    current_stage = config.get_current_stage()
    if stage == 1:
        return current_stage >= 1
    elif stage == 2:
        return current_stage >= 2
    elif stage >= 3:
        return current_stage >= 3
    else:
        return False


if __name__ == "__main__":
    config.print_stage_status()