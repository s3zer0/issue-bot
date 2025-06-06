"""
키워드 생성기 pytest 테스트
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock

# 프로젝트 루트 디렉토리의 src 폴더를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

try:
    from openai import AuthenticationError
except ImportError:
    class AuthenticationError(Exception): pass

# 의존성 모듈 임포트
from src.keyword_generator import (
    KeywordGenerator, KeywordResult, create_keyword_generator, generate_keywords_for_topic
)

class TestKeywordGenerator:
    """키워드 생성기 테스트 클래스"""

    @pytest.mark.unit
    def test_keyword_result_dataclass(self):
        """[수정됨] KeywordResult 데이터클래스 테스트 - synonyms 제거"""
        result = KeywordResult(
            topic="테스트 주제",
            primary_keywords=["키워드1", "키워드2"],
            related_terms=["용어1", "용어2"],
            context_keywords=["맥락1"],
            confidence_score=0.85,
            generation_time=1.5,
            raw_response="테스트 응답"
        )
        assert result.topic == "테스트 주제"
        assert not hasattr(result, 'synonyms')

    @pytest.mark.unit
    @patch('src.keyword_generator.AsyncOpenAI')
    def test_build_prompt(self, mock_openai):
        """[수정됨] 개선된 프롬프트 생성 테스트"""
        generator = create_keyword_generator(api_key="test_key")
        prompt = generator._build_prompt("AI 기술", None, 15)
        assert "AI 기술" in prompt
        assert "primary_keywords" in prompt
        assert "related_terms" in prompt
        assert "context_keywords" in prompt
        assert "synonyms" not in prompt
        assert "단순 번역을 절대 피해주세요" in prompt

    @pytest.mark.unit
    @patch('src.keyword_generator.AsyncOpenAI')
    def test_get_all_keywords(self, mock_openai):
        """[수정됨] 전체 키워드 추출 테스트 - synonyms 제거"""
        generator = create_keyword_generator(api_key="test_key")
        result = KeywordResult(
            topic="테스트",
            primary_keywords=["A", "B"],
            related_terms=["C", "D"],
            context_keywords=["F", "G"],
            confidence_score=0.8,
            generation_time=1.0,
            raw_response="test"
        )
        all_keywords = generator.get_all_keywords(result)
        assert len(all_keywords) == 6
        assert "E" not in all_keywords

    @pytest.mark.integration
    @pytest.mark.asyncio
    @patch('src.keyword_generator.AsyncOpenAI')
    async def test_full_keyword_generation_flow(self, mock_openai):
        """[수정됨] 전체 키워드 생성 플로우 테스트 - AttributeError 수정"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "primary_keywords": ["인공지능", "AI", "머신러닝", "딥러닝"],
            "related_terms": ["신경망", "자연어처리", "컴퓨터비전"],
            "context_keywords": ["기술혁신", "자동화", "데이터과학"],
            "confidence": 0.92
        }
        '''
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        generator = create_keyword_generator(api_key="test_key")
        result = await generator.generate_keywords("AI 기술 발전")

        assert len(result.primary_keywords) == 4
        assert len(result.related_terms) == 3
        # 💡 [수정] result.synonyms 접근 코드 제거
        all_keywords = generator.get_all_keywords(result)
        assert len(all_keywords) == 10