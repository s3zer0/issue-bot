"""
고급 시나리오 및 예외 처리에 대한 테스트 파일
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock
import httpx

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import Config
from src.detection.keyword_generator import KeywordGenerator
from src.models import KeywordResult


@pytest.fixture
def mock_discord_interaction():
    """Mock Discord Interaction 객체 픽스처"""
    interaction = MagicMock()
    interaction.user = MagicMock()
    interaction.user.name = "TestUser"
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    interaction.is_done = MagicMock(return_value=True)
    return interaction


# --- 1. config.py 테스트 ---

@patch.dict(os.environ, {
    "OPENAI_TEMPERATURE": "invalid_value",
    "MAX_RETRY_COUNT": "not_a_number"
}, clear=True)
@patch('src.config.load_dotenv', return_value=True)
def test_config_fallback_on_invalid_env_vars(mock_load_dotenv):
    """
    환경 변수에 잘못된 값이 있을 때 기본값으로 대체되는지 테스트

    Args:
        mock_load_dotenv: load_dotenv 함수의 모의 객체
    """
    config_instance = Config()

    assert config_instance.get_openai_temperature() == 0.7
    assert config_instance.get_max_retry_count() == 3


# --- 2. keyword_generator.py 테스트 ---

@pytest.mark.asyncio
@patch('src.keyword_generator._generate_keywords', new_callable=AsyncMock)
async def test_keyword_generator_retry_logic(mock_generate_keywords):
    """
    레거시 래퍼가 새 시스템을 올바르게 호출하는지 테스트
    (기존 재시도 로직 테스트는 새로운 생성 시스템으로 이전되어야 함)

    Args:
        mock_generate_keywords: _generate_keywords 함수의 모의 객체
    """
    # 모의 결과 설정
    mock_result = KeywordResult(
        topic="테스트",
        primary_keywords=["성공"],
        related_terms=[],
        context_keywords=[],
        confidence_score=0.95,
        generation_time=1.2,
        raw_response="{}"
    )
    mock_generate_keywords.return_value = mock_result

    # 테스트 실행
    generator = KeywordGenerator(api_key="fake_key")
    result = await generator.generate_keywords("테스트")

    # 검증
    mock_generate_keywords.assert_awaited_once_with("테스트", None)
    assert "성공" in result.primary_keywords
    assert isinstance(result, KeywordResult)


# --- 3. bot.py 테스트 ---

@pytest.mark.asyncio
@patch('src.bot.generate_keywords_for_topic')
@patch('src.bot.config')
async def test_monitor_command_general_exception(mock_config, mock_generate_keywords, mock_discord_interaction):
    """
    /monitor 명령어 실행 중 예상치 못한 예외 처리 테스트

    Discord 봇 명령어 함수를 직접 import하여 테스트합니다.
    """
    # bot.py에서 monitor_command 함수를 import
    from src.bot.bot import monitor_command

    mock_config.get_current_stage.return_value = 4
    error_message = "예상치 못한 심각한 오류"
    mock_generate_keywords.side_effect = Exception(error_message)

    # monitor_command 함수를 직접 호출
    await monitor_command(mock_discord_interaction, 주제="오류 테스트", 기간="1일")

    # 에러 메시지 전송 확인
    mock_discord_interaction.followup.send.assert_called()
    call_args = mock_discord_interaction.followup.send.call_args
    sent_embed = call_args.kwargs['embed']

    assert "시스템 오류 발생" in sent_embed.title
    assert error_message in sent_embed.description or "오류" in sent_embed.description
    assert call_args.kwargs.get('ephemeral', False) is True