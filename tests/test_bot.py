"""
Discord 이슈 모니터링 봇 - pytest 테스트 (최종 완성본)
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

# 경로 설정 및 의존성 모듈 임포트
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.bot import help_command, run_bot, IssueMonitorBot, validate_topic, parse_time_period

# --- Helper Fixtures ---
@pytest.fixture
def mock_discord_interaction():
    interaction = MagicMock()
    interaction.user.name = "TestUser"
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    return interaction

# --- 기능 단위 테스트 ---
class TestHelperFunctions:
    """bot.py의 순수 헬퍼 함수들을 테스트합니다."""
    @pytest.mark.unit
    def test_parse_valid_days(self):
        assert "3일" in parse_time_period("3일")[1]

    @pytest.mark.unit
    def test_topic_validation(self):
        assert validate_topic("AI 기술") is True
        assert validate_topic("A") is False

# --- 비동기 명령어 테스트 클래스 ---
@pytest.mark.asyncio
class TestBotCommands:
    """봇의 비동기 명령어들을 테스트합니다."""
    @patch('src.bot.config')
    async def test_help_command(self, mock_config, mock_discord_interaction):
        mock_config.get_current_stage.return_value = 4
        await help_command.callback(mock_discord_interaction)

        call_args = mock_discord_interaction.response.send_message.call_args
        embed = call_args.kwargs['embed']
        assert "이슈 모니터링 봇 사용법" in embed.title

# --- 봇 이벤트 테스트 클래스 ---
@pytest.mark.asyncio
class TestBotEvents:
    """봇의 이벤트를 테스트합니다."""
    @patch('src.bot.config')
    async def test_on_ready_event(self, mock_config):
        """on_ready 이벤트가 상태 메시지를 설정하는지 테스트"""
        bot_instance = IssueMonitorBot()

        # bot.user, bot.guilds는 읽기 전용이므로, 내부 _connection 객체를 모킹
        bot_instance._connection = MagicMock()
        bot_instance._connection.user = MagicMock()
        bot_instance._connection.guilds = []

        bot_instance.change_presence = AsyncMock()

        mock_config.get_current_stage.return_value = 4
        await bot_instance.on_ready()

        bot_instance.change_presence.assert_called_once()
        activity = bot_instance.change_presence.call_args.kwargs['activity']
        assert "Stage 4" in activity.name

# 동기 함수 테스트를 위한 별도 클래스 분리 (PytestWarning 해결)
class TestRunBot:
    """run_bot 함수의 실행 경로를 테스트합니다."""
    @patch('src.bot.config')
    @patch('src.bot.bot')
    def test_run_bot_success(self, mock_bot_instance, mock_config):
        """봇이 정상적으로 실행되는 경로 테스트"""
        mock_config.get_discord_token.return_value = "fake_token"
        run_bot()
        mock_bot_instance.run.assert_called_once_with("fake_token", log_handler=None)

    @patch('src.bot.config')
    @patch('src.bot.bot')
    def test_run_bot_no_token(self, mock_bot_instance, mock_config):
        """토큰이 없을 때 봇이 실행되지 않는지 테스트"""
        mock_config.get_discord_token.return_value = None
        run_bot()
        mock_bot_instance.run.assert_not_called()