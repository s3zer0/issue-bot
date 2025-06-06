"""
키워드 생성기 pytest 테스트 (최종 완성본)
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from openai import RateLimitError # 429 오류를 위한 구체적인 예외

# 의존성 모듈 임포트
from src.keyword_generator import KeywordGenerator

class TestKeywordGeneratorAdvanced:
    """키워드 생성기의 고급 예외 처리 및 엣지 케이스를 테스트합니다."""

    @pytest.mark.asyncio
    @patch('src.keyword_generator.config')
    async def test_generator_handles_rate_limit_error(self, mock_config):
        """[커버리지 향상] API 속도 제한(RateLimitError) 오류 처리 테스트"""
        mock_config.get_max_retry_count.return_value = 1

        # openai 라이브러리의 RateLimitError를 모킹하여 발생
        mock_response = MagicMock()
        mock_response.status_code = 429
        rate_limit_error = RateLimitError("Rate limit exceeded", response=mock_response, body=None)

        with patch('openai.resources.chat.completions.AsyncCompletions.create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = rate_limit_error
            generator = KeywordGenerator(api_key="fake_key")

            with pytest.raises(ValueError, match="API 사용량 한도를 초과했습니다."):
                await generator.generate_keywords("속도 제한 테스트")

    @patch('src.keyword_generator.AsyncOpenAI')
    def test_clean_keywords_with_invalid_input_type(self, mock_openai):
        """[수정] _clean_keywords에 리스트가 아닌 값이 들어왔을 때 반환 값을 검증"""
        generator = KeywordGenerator(api_key="fake_key")

        # 💡 caplog 대신, 함수의 반환 값이 빈 리스트인지를 직접 검증
        assert generator._clean_keywords("이것은 리스트가 아님") == []
        assert generator._clean_keywords({"key": "value"}) == []

    @pytest.mark.asyncio
    @patch('src.keyword_generator.config')
    async def test_generator_handles_empty_response(self, mock_config):
        """[커버리지 향상] LLM 응답 내용은 있으나 비어있는 경우(empty string) 테스트"""
        mock_config.get_max_retry_count.return_value = 1

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""  # 비어있는 응답

        with patch('openai.resources.chat.completions.AsyncCompletions.create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            generator = KeywordGenerator(api_key="fake_key")

            with pytest.raises(ValueError, match="LLM 응답이 비어있습니다"):
                await generator.generate_keywords("빈 응답 테스트")

    # JSON은 유효하나, primary_keywords가 없는 경우의 폴백 테스트
    @patch('src.keyword_generator.AsyncOpenAI')
    def test_parse_response_no_primary_keywords(self, mock_openai):
        """[수정] JSON에 필수 필드가 없을 때, 폴백 결과를 반환하는지 테스트"""
        generator = KeywordGenerator(api_key="fake_key")
        # primary_keywords가 없는 유효한 JSON
        response_content = '{"related_terms": ["c"], "context_keywords": ["d"]}'

        # 💡 [수정] ValueError 대신, 폴백 로직이 실행되는지 검증
        result = generator._parse_response("테스트", response_content, 1.0)

        # 폴백 결과의 특징인 낮은 신뢰도 점수를 확인
        assert result.confidence_score == 0.2
        # 폴백 결과의 기본 키워드가 주제명으로 설정되었는지 확인
        assert result.primary_keywords == ["테스트"]