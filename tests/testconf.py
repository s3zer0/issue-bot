# tests/conftest.py
"""
Pytest 공통 픽스처(fixture)를 정의하는 파일
"""
import pytest
from unittest.mock import MagicMock, AsyncMock

@pytest.fixture
def mock_discord_interaction():
    """
    여러 테스트에서 공통으로 사용할 수 있는
    Mock Discord Interaction 객체 픽스처
    """
    interaction = MagicMock()
    interaction.user.name = "TestUser"
    interaction.guild.name = "TestServer"
    interaction.response = AsyncMock()
    interaction.followup = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    return interaction