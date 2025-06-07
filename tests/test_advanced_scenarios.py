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
from src.keyword_generator import KeywordGenerator
from src.bot import monitor_command


# 'fixture not found' 오류 해결을 위해 fixture를 파일 내에 직접 정의
@pytest.fixture
def mock_discord_interaction():
    """Mock Discord Interaction 객체 픽스처"""
    interaction = MagicMock()
    interaction.user.name = "TestUser"
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    return interaction

# --- 1. config.py 테스트 ---

@patch.dict(os.environ, {
    "OPENAI_TEMPERATURE": "invalid_value",
    "MAX_RETRY_COUNT": "not_a_number"
}, clear=True)
@patch('src.config.load_dotenv', return_value=True)
def test_config_fallback_on_invalid_env_vars(mock_load_dotenv):
    """
    [config.py] 환경 변수에 잘못된 값이 있을 때 기본값으로 대체되는지 테스트
    """
    # loguru와 caplog의 호환성 문제로 로그 검증 대신 반환 값 검증에 집중
    config_instance = Config()

    assert config_instance.get_openai_temperature() == 0.7
    assert config_instance.get_max_retry_count() == 3

# --- 2. keyword_generator.py 테스트 ---

@pytest.mark.asyncio
@patch('src.keyword_generator.config')
async def test_keyword_generator_retry_logic(mock_config):
    """
    [keyword_generator.py] API 호출 재시도 로직 테스트
    """
    mock_config.get_max_retry_count.return_value = 3

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"primary_keywords": ["성공"], "related_terms": [], "context_keywords": []}'

    side_effects = [
        httpx.RequestError("Network error"),
        mock_response
    ]

    with patch('openai.resources.chat.completions.AsyncCompletions.create', new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = side_effects

        generator = KeywordGenerator(api_key="fake_key")
        result = await generator.generate_keywords("테스트")

        assert mock_create.call_count == 2
        assert "성공" in result.primary_keywords

# --- 3. bot.py 테스트 ---
@pytest.mark.asyncio
@patch('src.bot.generate_keywords_for_topic')
@patch('src.bot.config')
async def test_monitor_command_general_exception(
    mock_config, mock_generate_keywords, mock_discord_interaction # 이제 conftest.py의 fixture가 주입됩니다.
):
    """
    [bot.py] /monitor 명령어 실행 중 예상치 못한 예외 처리 테스트
    """
    mock_config.get_current_stage.return_value = 4
    error_message = "예상치 못한 심각한 오류"
    mock_generate_keywords.side_effect = Exception(error_message)

    await monitor_command.callback(mock_discord_interaction, 주제="오류 테스트", 기간="1일", 세부분석=True)

    final_call_args = mock_discord_interaction.followup.send.call_args
    sent_embed = final_call_args.kwargs['embed']

    assert "시스템 오류 발생" in sent_embed.title
    assert error_message in sent_embed.description
    assert final_call_args.kwargs['ephemeral'] is True
