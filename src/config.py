"""
설정 관리 모듈
"""

import os
import sys
from typing import Optional
from pathlib import Path
from loguru import logger


class Config:
    """설정 관리 클래스"""

    def __init__(self, load_env_file: bool = True):
        self._setup_environment()
        if load_env_file:
            self._load_env_file()

    def _setup_environment(self):
        """작업 디렉토리 및 Python 경로 설정"""
        try:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent

            # issue-bot 디렉토리 찾기
            while project_root.name != 'issue-bot' and project_root.parent != project_root:
                project_root = project_root.parent

            if project_root.name != 'issue-bot':
                # 폴백: 절대 경로로 시도
                possible_paths = [
                    Path('/Users/choeseyeong/Documents/issue-bot'),
                    Path.cwd() / 'issue-bot',
                    Path.cwd()
                ]

                for path in possible_paths:
                    if path.exists() and (path / 'src' / 'config.py').exists():
                        project_root = path
                        break
                else:
                    project_root = Path.cwd()

            if os.getcwd() != str(project_root):
                os.chdir(project_root)
                logger.debug(f"작업 디렉토리 변경: {project_root}")

            # Python 경로에 src 디렉토리 추가
            src_path = project_root / 'src'
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            os.environ['PROJECT_ROOT'] = str(project_root)

        except Exception as e:
            logger.warning(f"환경 설정 중 오류: {e}")

    def _find_env_file(self) -> Optional[str]:
        """프로젝트 루트에서 .env 파일 찾기"""
        current_dir = Path.cwd()

        env_paths = [
            current_dir / '.env',
            Path(__file__).parent / '.env',
            Path(__file__).parent.parent / '.env',
        ]

        if 'PROJECT_ROOT' in os.environ:
            env_paths.append(Path(os.environ['PROJECT_ROOT']) / '.env')

        for env_path in env_paths:
            abs_path = env_path.resolve()
            if abs_path.exists():
                logger.debug(f".env 파일 발견: {abs_path}")
                return str(abs_path)

        logger.warning(f".env 파일을 찾을 수 없습니다")
        return None

    def _load_env_file(self):
        """환경변수 파일 로드"""
        try:
            env_file_path = self._find_env_file()

            if env_file_path:
                with open(env_file_path, 'r', encoding='utf-8') as f:
                    loaded_count = 0
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            try:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"\'')

                                # 템플릿 값 제외
                                if not value.startswith('your_'):
                                    os.environ[key] = value
                                    loaded_count += 1

                            except ValueError:
                                logger.warning(f".env 파일 {line_num}행 파싱 오류: {line}")

                logger.debug(f".env 파일 로드 완료: ({loaded_count}개 변수)")
            else:
                self._create_sample_env_file()

        except Exception as e:
            logger.error(f".env 파일 로드 실패: {e}")

    def _create_sample_env_file(self):
        """샘플 .env 파일 생성"""
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
            sample_path = Path.cwd() / '.env.example'
            with open(sample_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)

            logger.info(f"샘플 .env 파일 생성됨: {sample_path}")
            logger.info("실제 API 키로 수정한 후 .env로 이름을 변경하세요")

        except Exception as e:
            logger.error(f"샘플 .env 파일 생성 실패: {e}")

    # API 키 반환 메서드들
    def get_discord_token(self) -> Optional[str]:
        token = os.getenv('DISCORD_BOT_TOKEN')
        if not token:
            logger.error("DISCORD_BOT_TOKEN이 설정되지 않았습니다")
        return token

    def get_openai_api_key(self) -> Optional[str]:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY가 설정되지 않았습니다")
        return api_key

    def get_perplexity_api_key(self) -> Optional[str]:
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if not api_key:
            logger.warning("PERPLEXITY_API_KEY가 설정되지 않았습니다")
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
            "stage1_discord": bool(self.get_discord_token()),
            "stage2_openai": bool(self.get_openai_api_key()),
            "stage3_perplexity": bool(self.get_perplexity_api_key()),
            "development_mode": self.is_development_mode(),
            "log_level": self.get_log_level()
        }

    def print_stage_status(self) -> None:
        """단계별 상태를 콘솔에 출력"""
        stage_info = self.get_stage_info()
        current_stage = self.get_current_stage()

        print("\n🔧 === 설정 상태 확인 ===")
        print(f"현재 실행 가능한 단계: {current_stage}단계")
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        print(f"개발 모드: {'ON' if stage_info['development_mode'] else 'OFF'}")
        print(f"로그 레벨: {stage_info['log_level']}")
        print("\n📋 단계별 준비 상태:")
        print(f"  1단계 (Discord 봇): {'✅' if stage_info['stage1_discord'] else '❌'}")
        print(f"  2단계 (키워드 생성): {'✅' if stage_info['stage2_openai'] else '❌'}")
        print(f"  3-4단계 (이슈 검색): {'✅' if stage_info['stage3_perplexity'] else '❌'}")

        if current_stage < 4:
            print(f"\n💡 다음 단계 진행을 위해 필요한 설정:")
            if current_stage < 1:
                print("  • DISCORD_BOT_TOKEN 설정")
            elif current_stage < 2:
                print("  • OPENAI_API_KEY 설정")
            elif current_stage < 4:
                print("  • PERPLEXITY_API_KEY 설정")

        env_file = self._find_env_file()
        if env_file:
            print(f"\n📄 .env 파일 위치: {env_file}")
        else:
            print(f"\n📄 .env 파일: ❌ 찾을 수 없음")

        print("========================\n")

    # OpenAI 설정 메서드들
    def get_openai_model(self) -> str:
        return os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

    def get_openai_temperature(self) -> float:
        try:
            return float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        except ValueError:
            logger.warning("OPENAI_TEMPERATURE 값이 잘못되었습니다. 기본값 0.7 사용")
            return 0.7

    def get_openai_max_tokens(self) -> int:
        try:
            return int(os.getenv('OPENAI_MAX_TOKENS', '1500'))
        except ValueError:
            logger.warning("OPENAI_MAX_TOKENS 값이 잘못되었습니다. 기본값 1500 사용")
            return 1500

    def get_keyword_generation_timeout(self) -> int:
        try:
            return int(os.getenv('KEYWORD_GENERATION_TIMEOUT', '30'))
        except ValueError:
            logger.warning("KEYWORD_GENERATION_TIMEOUT 값이 잘못되었습니다. 기본값 30 사용")
            return 30

    def get_max_retry_count(self) -> int:
        try:
            return int(os.getenv('MAX_RETRY_COUNT', '3'))
        except ValueError:
            logger.warning("MAX_RETRY_COUNT 값이 잘못되었습니다. 기본값 3 사용")
            return 3


# 전역 설정 인스턴스
config = Config()


def get_config() -> Config:
    """전역 설정 인스턴스 반환"""
    return config


def is_ready_for_stage(stage: int) -> bool:
    """특정 단계 실행 준비 여부 확인"""
    if stage == 1:
        return config.validate_stage1_requirements()
    elif stage == 2:
        return config.validate_stage2_requirements()
    elif stage >= 3:
        return config.validate_stage3_requirements()
    else:
        return False


if __name__ == "__main__":
    config.print_stage_status()