# tests/clients/test_perplexity_client.py
import pytest
import sys
import os
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.clients.perplexity_client import PerplexityClient


@pytest.mark.asyncio
class TestPerplexityClient:
    """PerplexityClient의 API 호출 로직을 테스트합니다."""

    @patch('httpx.AsyncClient')
    async def test_make_api_call_success(self, mock_async_client):
        """API 호출이 성공하는 경우를 테스트합니다."""
        # httpx.Response 대신 MagicMock 사용
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": [{"message": {"content": "성공"}}]}
        # 200 응답 시 raise_for_status()는 아무것도 하지 않으므로 그대로 둠
        mock_response.raise_for_status = MagicMock()

        mock_post = AsyncMock(return_value=mock_response)
        mock_async_client.return_value.__aenter__.return_value.post = mock_post

        client = PerplexityClient(api_key="fake_key")
        response = await client._make_api_call("테스트 프롬프트")

        assert response["choices"][0]["message"]["content"] == "성공"
        mock_post.assert_called_once()
        mock_response.raise_for_status.assert_called_once()  # 호출되었는지 확인

    @patch('httpx.AsyncClient')
    async def test_make_api_call_retry_on_429(self, mock_async_client):
        """429 (Rate Limit) 오류 발생 시 재시도하는지 테스트합니다."""
        # 성공 응답과 429 오류 응답 모두 MagicMock으로 설정
        mock_success_response = MagicMock(spec=httpx.Response)
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {"success": True}
        mock_success_response.raise_for_status = MagicMock()

        mock_429_response = MagicMock(spec=httpx.Response)
        mock_429_response.status_code = 429
        # 429 응답 시 raise_for_status()는 HTTPStatusError를 발생시켜야 함
        mock_429_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Rate limit exceeded", request=MagicMock(), response=mock_429_response
        )

        mock_post = AsyncMock(side_effect=[mock_429_response, mock_success_response])
        mock_async_client.return_value.__aenter__.return_value.post = mock_post

        client = PerplexityClient(api_key="fake_key")
        await client._make_api_call("재시도 테스트")

        assert mock_post.call_count == 2