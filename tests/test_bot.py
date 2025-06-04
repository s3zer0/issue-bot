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

from src.bot import (
    parse_time_period,
    validate_topic,
    validate_period,
    IssueMonitorBot
)


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
        result = validate_topic(topic)
        assert result == expected

    @pytest.mark.unit
    def test_empty_string_validation(self):
        """빈 문자열 별도 테스트"""
        assert validate_topic("") == False
        assert validate_topic("AI") == True

    @pytest.mark.unit
    def test_topic_length_boundary(self):
        """주제 길이 경계값 테스트"""
        # 정확히 2글자
        assert validate_topic("AI") == True

        # 1글자 (무효)
        assert validate_topic("A") == False

        # 매우 긴 주제 (유효)
        long_topic = "매우 긴 주제명으로 테스트하는 경우입니다" * 10
        assert validate_topic(long_topic) == True

    @pytest.mark.unit
    @pytest.mark.parametrize("period,expected", [
        ("1일", True),
        ("3주일", True),
        ("2개월", True),
        ("24시간", True),
        ("", True),  # 빈 값은 기본값 사용
        ("잘못된형식", False),
        ("abc일", False),
    ])
    def test_period_validation(self, period, expected):
        """기간 유효성 검증 테스트"""
        result = validate_period(period)
        assert result == expected


class TestBotIntegration:
    """봇 통합 테스트 클래스"""

    @pytest.mark.integration
    def test_bot_initialization(self):
        """봇 초기화 테스트"""
        bot = IssueMonitorBot()
        assert bot is not None
        assert hasattr(bot, 'tree')  # 슬래시 명령어 트리
        assert bot.command_prefix == '!'

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_bot_setup_hook(self):
        """봇 setup_hook 테스트"""
        bot = IssueMonitorBot()

        # tree.sync를 Mock으로 대체 (비동기 함수)
        async def mock_sync():
            return []

        bot.tree.sync = mock_sync

        # setup_hook 호출 테스트
        try:
            await bot.setup_hook()
            # 오류 없이 실행되면 성공
            assert True
        except Exception as e:
            pytest.fail(f"setup_hook 실행 중 오류: {e}")


class TestConfigIntegration:
    """설정 통합 테스트 클래스"""

    @pytest.mark.unit
    def test_config_import(self):
        """Config 클래스 import 테스트"""
        try:
            from src.config import Config, config
            assert Config is not None
            assert config is not None
        except ImportError:
            pytest.fail("Config 클래스를 import할 수 없습니다")

    @pytest.mark.unit
    def test_config_basic_attributes(self):
        """Config 기본 속성 존재 테스트"""
        from src.config import config

        # 속성이 존재하는지만 확인 (값은 환경에 따라 다를 수 있음)
        assert hasattr(config, 'get_discord_token')
        assert hasattr(config, 'get_openai_api_key')
        assert hasattr(config, 'get_perplexity_api_key')
        assert hasattr(config, 'is_development_mode')

    @pytest.mark.unit
    def test_config_with_env_vars(self):
        """환경변수가 있을 때 Config 로딩 테스트"""
        with patch.dict(os.environ, {
            'DISCORD_BOT_TOKEN': 'test_token',
            'OPENAI_API_KEY': 'test_openai_key',
            'PERPLEXITY_API_KEY': 'test_perplexity_key'
        }, clear=True):
            from src.config import Config

            test_config = Config(load_env_file=False)  # 💡 수정된 부분

            assert test_config.get_discord_token() == 'test_token'
            assert test_config.get_openai_api_key() == 'test_openai_key'
            assert test_config.get_perplexity_api_key() == 'test_perplexity_key'


if __name__ == "__main__":
    # pytest 직접 실행 (개발 중 편의용)
    pytest.main([__file__, "-v"])