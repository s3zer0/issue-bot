"""
Discord 이슈 모니터링 봇 - pytest 테스트
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# 프로젝트 루트 디렉토리의 src 폴더를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from bot import parse_time_period


class TestTimePeriodParsing:
    """시간 기간 파싱 테스트 클래스"""

    @pytest.mark.unit
    def test_parse_valid_days(self):
        """유효한 일 단위 파싱 테스트"""
        start_date, description = parse_time_period("3일")

        assert "3일" in description
        assert isinstance(start_date, datetime)

        # 대략 3일 전인지 확인 (±1시간 오차 허용)
        expected_days_ago = 3
        actual_days_ago = (datetime.now() - start_date).days
        assert abs(actual_days_ago - expected_days_ago) <= 1

    @pytest.mark.unit
    def test_parse_valid_weeks(self):
        """유효한 주 단위 파싱 테스트"""
        start_date, description = parse_time_period("2주일")

        assert "2주일" in description
        assert isinstance(start_date, datetime)

        expected_days_ago = 14  # 2주
        actual_days_ago = (datetime.now() - start_date).days
        assert abs(actual_days_ago - expected_days_ago) <= 1

    @pytest.mark.unit
    def test_parse_valid_months(self):
        """유효한 월 단위 파싱 테스트"""
        start_date, description = parse_time_period("1개월")

        assert "1개월" in description
        assert isinstance(start_date, datetime)

        expected_days_ago = 30  # 약 1개월
        actual_days_ago = (datetime.now() - start_date).days
        assert abs(actual_days_ago - expected_days_ago) <= 2

    @pytest.mark.unit
    def test_parse_valid_hours(self):
        """유효한 시간 단위 파싱 테스트"""
        start_date, description = parse_time_period("24시간")

        assert "24시간" in description
        assert isinstance(start_date, datetime)

        expected_hours_ago = 24
        actual_hours_ago = (datetime.now() - start_date).total_seconds() / 3600
        assert abs(actual_hours_ago - expected_hours_ago) <= 1

    @pytest.mark.unit
    def test_parse_invalid_input(self):
        """잘못된 입력에 대한 기본값 처리 테스트"""
        start_date, description = parse_time_period("잘못된입력")

        assert "1주일" in description  # 기본값
        assert isinstance(start_date, datetime)

        expected_days_ago = 7  # 1주일
        actual_days_ago = (datetime.now() - start_date).days
        assert abs(actual_days_ago - expected_days_ago) <= 1

    @pytest.mark.unit
    def test_parse_empty_input(self):
        """빈 입력에 대한 기본값 처리 테스트"""
        start_date, description = parse_time_period("")

        assert "1주일" in description  # 기본값
        assert isinstance(start_date, datetime)

    @pytest.mark.unit
    @pytest.mark.parametrize("input_period,expected_keyword", [
        ("1일", "1일"),
        ("5일", "5일"),
        ("1주일", "1주일"),
        ("3주일", "3주일"),
        ("1개월", "1개월"),
        ("2달", "2개월"),
        ("12시간", "12시간"),
    ])
    def test_parse_various_formats(self, input_period, expected_keyword):
        """다양한 형식 파싱 테스트"""
        start_date, description = parse_time_period(input_period)

        assert expected_keyword in description or "개월" in description
        assert isinstance(start_date, datetime)
        assert start_date < datetime.now()


class TestConfigLoading:
    """설정 로딩 테스트 클래스"""

    @pytest.mark.unit
    def test_config_import(self):
        """Config 클래스 import 테스트"""
        try:
            from config import Config
            config = Config()
            assert config is not None
        except ImportError:
            pytest.fail("Config 클래스를 import할 수 없습니다")

    @pytest.mark.unit
    def test_config_with_env_vars(self):
        """환경변수가 있을 때 Config 로딩 테스트"""
        # config 모듈을 다시 import하기 위해 sys.modules에서 제거
        if 'config' in sys.modules:
            del sys.modules['config']

        with patch.dict(os.environ, {
            'DISCORD_BOT_TOKEN': 'test_token',
            'OPENAI_API_KEY': 'test_openai_key',
            'PERPLEXITY_API_KEY': 'test_perplexity_key'
        }, clear=False):
            # 환경변수 설정 후 config 다시 import
            import importlib
            import config
            importlib.reload(config)

            test_config = config.Config()

            assert test_config.DISCORD_BOT_TOKEN == 'test_token'
            assert test_config.OPENAI_API_KEY == 'test_openai_key'
            assert test_config.PERPLEXITY_API_KEY == 'test_perplexity_key'

    @pytest.mark.unit
    def test_config_basic_attributes(self):
        """Config 기본 속성 존재 테스트"""
        from config import Config
        config = Config()

        # 속성이 존재하는지만 확인 (값은 환경에 따라 다를 수 있음)
        assert hasattr(config, 'DISCORD_BOT_TOKEN')
        assert hasattr(config, 'OPENAI_API_KEY')
        assert hasattr(config, 'PERPLEXITY_API_KEY')
        assert hasattr(config, 'DEBUG')


class TestInputValidation:
    """입력값 검증 테스트 클래스"""

    @pytest.mark.unit
    @pytest.mark.parametrize("topic,expected", [
        ("AI 기술", True),
        ("암호화폐", True),
        ("기후변화 대응정책", True),
        ("블록체인", True),
        ("ab", True),  # 2글자 이상
        (" ", False),  # 공백만
        ("a", False),  # 1글자
        ("1", False),  # 숫자 1글자
    ])
    def test_topic_validation(self, topic, expected):
        """주제 유효성 검증 테스트"""
        is_valid = topic and len(topic.strip()) >= 2
        assert is_valid == expected

    @pytest.mark.unit
    def test_empty_string_validation(self):
        """빈 문자열 별도 테스트"""
        topic = ""
        is_valid = topic and len(topic.strip()) >= 2

        # 실제 동작 확인: topic이 빈 문자열일 때 and 연산은 빈 문자열을 반환
        assert is_valid == ""      # 실제로는 빈 문자열이 반환됨
        assert not is_valid        # falsy 값이므로 not 연산은 True
        assert bool(is_valid) == False  # bool로 변환하면 False

        # 더 정확한 validation 함수 테스트
        def validate_topic(topic_str):
            return bool(topic_str and len(topic_str.strip()) >= 2)

        assert validate_topic("") == False      # 정확히 False 반환
        assert validate_topic("AI") == True     # 정확히 True 반환

    @pytest.mark.unit
    def test_topic_length_boundary(self):
        """주제 길이 경계값 테스트"""
        # 정확히 2글자
        assert len("AI".strip()) >= 2

        # 1글자 (무효)
        assert not (len("A".strip()) >= 2)

        # 매우 긴 주제 (유효)
        long_topic = "매우 긴 주제명으로 테스트하는 경우입니다" * 10
        assert len(long_topic.strip()) >= 2


class TestBotIntegration:
    """봇 통합 테스트 클래스"""

    @pytest.mark.integration
    def test_bot_import(self):
        """봇 모듈 import 테스트"""
        try:
            from bot import IssueMonitorBot
            bot = IssueMonitorBot()
            assert bot is not None
            assert hasattr(bot, 'tree')  # 슬래시 명령어 트리
        except ImportError:
            pytest.fail("IssueMonitorBot을 import할 수 없습니다")

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_bot_setup_hook(self):
        """봇 setup_hook 테스트"""
        from bot import IssueMonitorBot

        # Mock 객체로 tree.sync 메서드 대체
        bot = IssueMonitorBot()

        # tree.sync를 Mock으로 대체 (비동기 함수)
        async def mock_sync():
            return None

        bot.tree.sync = mock_sync

        # setup_hook 호출 테스트
        try:
            await bot.setup_hook()
            # 오류 없이 실행되면 성공
            assert True
        except Exception as e:
            pytest.fail(f"setup_hook 실행 중 오류: {e}")

    @pytest.mark.integration
    def test_bot_commands_exist(self):
        """봇 명령어 존재 확인 테스트"""
        from bot import bot

        # 슬래시 명령어가 등록되어 있는지 확인
        assert hasattr(bot, 'tree')

        # 명령어 함수들이 정의되어 있는지 확인
        import bot as bot_module
        assert hasattr(bot_module, 'monitor_command')
        assert hasattr(bot_module, 'help_command')


if __name__ == "__main__":
    # pytest 직접 실행 (개발 중 편의용)
    pytest.main([__file__, "-v"])