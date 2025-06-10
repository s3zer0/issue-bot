# tests/clients/test_perplexity_client.py
import pytest
import sys
import os
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
import asyncio

# 프로젝트의 루트 디렉토리를 시스템 경로에 추가하여 모듈을 임포트할 수 있도록 합니다.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 테스트 대상인 PerplexityClient를 임포트합니다.
from src.clients.perplexity_client import PerplexityClient


@pytest.mark.asyncio
# PerplexityClient 내부에서 사용하는 httpx.AsyncClient를 모의(mock) 객체로 대체합니다.
# 이렇게 하면 실제 네트워크 요청 없이 API 클라이언트의 동작을 테스트할 수 있습니다.
@patch("httpx.AsyncClient")
class TestPerplexityClient:
    """PerplexityClient의 API 호출 로직을 테스트합니다."""

    async def test_make_api_call_success(self, mock_async_client_class):
        """API 호출이 성공하는 경우를 테스트합니다."""
        # API가 성공적으로 응답(상태 코드 200)하는 상황을 가정합니다.
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        # API 응답 본문을 모의 처리합니다.
        mock_response.json.return_value = {"choices": [{"message": {"content": "성공"}}]}
        # response.raise_for_status()가 호출될 때 아무런 예외도 발생시키지 않도록 설정합니다.
        mock_response.raise_for_status = MagicMock()

        # PerplexityClient 생성자에서 만들어지는 `httpx.AsyncClient`의 모의 인스턴스를 가져옵니다.
        mock_instance = mock_async_client_class.return_value
        # 해당 인스턴스의 `post` 메서드가 `await` 키워드로 호출될 수 있도록 `AsyncMock`으로 설정하고,
        # 위에서 정의한 성공 응답(mock_response)을 반환하도록 합니다.
        mock_instance.post = AsyncMock(return_value=mock_response)

        # 테스트할 클라이언트를 생성합니다.
        client = PerplexityClient(api_key="fake_key")
        # 실제 API 호출 메서드를 실행합니다.
        response = await client._make_api_call("테스트 프롬프트")

        # 반환된 응답이 예상과 같은지 확인합니다.
        assert response["choices"][0]["message"]["content"] == "성공"
        # `post` 메서드가 정확히 한 번 호출되었는지 확인합니다.
        mock_instance.post.assert_called_once()
        # `raise_for_status`가 한 번 호출되었는지 확인합니다.
        mock_response.raise_for_status.assert_called_once()

    async def test_make_api_call_retry_on_429(self, mock_async_client_class):
        """429 (Rate Limit) 오류 발생 시 재시도하는지 테스트합니다."""
        # 성공 응답(200) 모의 객체를 생성합니다.
        mock_success_response = MagicMock(spec=httpx.Response)
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = {"success": True}
        mock_success_response.raise_for_status = MagicMock()

        # Rate Limit 오류 응답(429) 모의 객체를 생성합니다.
        mock_429_response = MagicMock(spec=httpx.Response)
        mock_429_response.status_code = 429
        # `raise_for_status`가 호출될 때 `HTTPStatusError` 예외를 발생시키도록 설정합니다.
        mock_429_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Rate limit exceeded", request=MagicMock(), response=mock_429_response
        )

        # `httpx.AsyncClient`의 모의 인스턴스를 가져옵니다.
        mock_instance = mock_async_client_class.return_value
        # `post` 메서드가 처음 호출될 때는 429 오류를, 두 번째 호출될 때는 성공 응답을 반환하도록 설정합니다.
        mock_instance.post = AsyncMock(
            side_effect=[mock_429_response, mock_success_response]
        )

        client = PerplexityClient(api_key="fake_key")
        # `asyncio.sleep`을 모의 처리하여 테스트 중 실제 대기 시간을 없애고 즉시 실행되도록 합니다.
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await client._make_api_call("재시도 테스트")

        # `post` 메서드가 재시도를 포함하여 총 두 번 호출되었는지 확인합니다.
        assert mock_instance.post.call_count == 2
