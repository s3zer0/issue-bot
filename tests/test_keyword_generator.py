"""
키워드 생성기 pytest 테스트
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# 프로젝트 루트 디렉토리의 src 폴더를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)


class TestKeywordGenerator:
    """키워드 생성기 테스트 클래스"""

    @pytest.mark.unit
    def test_keyword_generator_import(self):
        """키워드 생성기 모듈 import 테스트"""
        from src.keyword_generator import KeywordGenerator, KeywordResult, create_keyword_generator
        assert KeywordGenerator is not None
        assert KeywordResult is not None
        assert create_keyword_generator is not None

    @pytest.mark.unit
    def test_keyword_result_dataclass(self):
        """KeywordResult 데이터클래스 테스트"""
        from src.keyword_generator import KeywordResult

        result = KeywordResult(
            topic="테스트 주제",
            primary_keywords=["키워드1", "키워드2"],
            related_terms=["용어1", "용어2"],
            synonyms=["동의어1"],
            context_keywords=["맥락1"],
            confidence_score=0.85,
            generation_time=1.5,
            raw_response="테스트 응답"
        )

        assert result.topic == "테스트 주제"
        assert len(result.primary_keywords) == 2
        assert result.confidence_score == 0.85
        assert result.generation_time == 1.5

    @pytest.mark.unit
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'test_api_key'})
    def test_keyword_generator_initialization(self):
        """키워드 생성기 초기화 테스트"""
        with patch('src.keyword_generator.AsyncOpenAI'):
            from src.keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")
            assert generator.api_key == "test_key"
            assert generator.model == "gpt-4o-mini"
            assert generator.max_retries >= 3

    @pytest.mark.unit
    def test_keyword_generator_no_api_key(self):
        """API 키 없을 때 예외 처리 테스트"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('keyword_generator.config') as mock_config:
                mock_config.get_openai_api_key.return_value = None

                from keyword_generator import create_keyword_generator

                with pytest.raises(ValueError, match="OpenAI API 키가 설정되지 않았습니다"):
                    create_keyword_generator()

    @pytest.mark.unit
    def test_clean_keywords(self):
        """키워드 정제 함수 테스트"""
        with patch('keyword_generator.AsyncOpenAI'):
            from keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")

            # 정상적인 키워드 리스트
            keywords = ["AI", "인공지능", "machine learning", " 딥러닝 ", '"자연어처리"']
            cleaned = generator._clean_keywords(keywords)

            assert "AI" in cleaned
            assert "인공지능" in cleaned
            assert "machine learning" in cleaned
            assert "딥러닝" in cleaned
            assert "자연어처리" in cleaned
            assert len(cleaned) == 5

            # 중복 제거 테스트
            keywords_with_duplicates = ["AI", "ai", "AI", "인공지능"]
            cleaned_no_dups = generator._clean_keywords(keywords_with_duplicates)
            assert len(cleaned_no_dups) == 2  # "AI"와 "인공지능"만 남아야 함

            # 빈 값 및 잘못된 형식 처리
            invalid_keywords = ["", " ", None, 123, "valid_keyword"]
            cleaned_invalid = generator._clean_keywords(invalid_keywords)
            assert "valid_keyword" in cleaned_invalid

    @pytest.mark.unit
    def test_build_prompt(self):
        """프롬프트 생성 테스트"""
        with patch('keyword_generator.AsyncOpenAI'):
            from keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")

            # 기본 프롬프트
            prompt = generator._build_prompt("AI 기술", None, 20)
            assert "AI 기술" in prompt
            assert "primary_keywords" in prompt
            assert "related_terms" in prompt
            assert "synonyms" in prompt
            assert "context_keywords" in prompt

            # 맥락 포함 프롬프트
            prompt_with_context = generator._build_prompt("AI 기술", "기업 환경에서의 활용", 20)
            assert "기업 환경에서의 활용" in prompt_with_context

    @pytest.mark.unit
    def test_create_fallback_result(self):
        """폴백 결과 생성 테스트"""
        with patch('keyword_generator.AsyncOpenAI'):
            from keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")

            fallback = generator._create_fallback_result("AI 기술 발전", "test response", 1.0)

            assert fallback.topic == "AI 기술 발전"
            assert "AI 기술 발전" in fallback.primary_keywords
            assert fallback.confidence_score == 0.2  # 낮은 신뢰도
            assert fallback.generation_time == 1.0
            assert fallback.raw_response == "test response"

    @pytest.mark.unit
    def test_parse_response_valid_json(self):
        """유효한 JSON 응답 파싱 테스트"""
        with patch('keyword_generator.AsyncOpenAI'):
            from keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")

            # 유효한 JSON 응답 시뮬레이션
            mock_response = '''
            {
                "primary_keywords": ["AI", "인공지능", "머신러닝"],
                "related_terms": ["딥러닝", "신경망"],
                "synonyms": ["Artificial Intelligence"],
                "context_keywords": ["기술혁신", "자동화"],
                "confidence": 0.9
            }
            '''

            result = generator._parse_response("AI 기술", mock_response, 1.5)

            assert result.topic == "AI 기술"
            assert len(result.primary_keywords) == 3
            assert "AI" in result.primary_keywords
            assert len(result.related_terms) == 2
            assert result.confidence_score == 0.9
            assert result.generation_time == 1.5

    @pytest.mark.unit
    def test_parse_response_invalid_json(self):
        """잘못된 JSON 응답 처리 테스트"""
        with patch('keyword_generator.AsyncOpenAI'):
            from keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")

            # 잘못된 응답
            invalid_response = "이것은 JSON이 아닙니다"

            result = generator._parse_response("테스트 주제", invalid_response, 1.0)

            # 폴백 결과가 반환되어야 함
            assert result.topic == "테스트 주제"
            assert result.confidence_score == 0.2
            assert "테스트 주제" in result.primary_keywords

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generate_keywords_success(self):
        """키워드 생성 성공 테스트 (Mock 사용)"""
        # Mock OpenAI 응답
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "primary_keywords": ["블록체인", "암호화폐", "Bitcoin"],
            "related_terms": ["스마트계약", "DeFi"],
            "synonyms": ["분산원장", "cryptocurrency"],
            "context_keywords": ["핀테크", "디지털자산"],
            "confidence": 0.95
        }
        '''

        with patch('keyword_generator.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")
            result = await generator.generate_keywords("블록체인 기술")

            assert result.topic == "블록체인 기술"
            assert len(result.primary_keywords) > 0
            assert "블록체인" in result.primary_keywords
            assert result.confidence_score == 0.95

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generate_keywords_api_error(self):
        """API 오류 시 키워드 생성 테스트"""
        with patch('keyword_generator.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.side_effect = Exception("API 연결 실패")
            mock_openai.return_value = mock_client

            from keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")

            with pytest.raises(ValueError, match="LLM API 호출 최종 실패"):
                await generator.generate_keywords("테스트 주제")

    @pytest.mark.unit
    def test_get_all_keywords(self):
        """전체 키워드 추출 테스트"""
        with patch('src.keyword_generator.AsyncOpenAI'):
            from src.keyword_generator import create_keyword_generator, KeywordResult

            generator = create_keyword_generator(api_key="test_key")

            result = KeywordResult(
                topic="테스트",
                primary_keywords=["A", "B"],
                related_terms=["C", "D"],
                synonyms=["E"],
                context_keywords=["F", "G"],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test"
            )

            all_keywords = generator.get_all_keywords(result)
            assert len(all_keywords) == 7
            assert all(kw in all_keywords for kw in ["A", "B", "C", "D", "E", "F", "G"])

    @pytest.mark.unit
    def test_format_keywords_summary(self):
        """키워드 요약 포맷팅 테스트"""
        with patch('src.keyword_generator.AsyncOpenAI'):
            from src.keyword_generator import create_keyword_generator, KeywordResult

            generator = create_keyword_generator(api_key="test_key")

            result = KeywordResult(
                topic="AI 기술",
                primary_keywords=["AI", "머신러닝", "딥러닝"],
                related_terms=["신경망", "알고리즘"],
                synonyms=["인공지능"],
                context_keywords=["기술혁신"],
                confidence_score=0.85,
                generation_time=2.0,
                raw_response="test"
            )

            summary = generator.format_keywords_summary(result)

            assert "AI 기술" in summary
            assert "85%" in summary  # 신뢰도 표시
            assert "🎯 **핵심**" in summary
            assert "🔗 **관련**" in summary

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """편의 함수 테스트"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "primary_keywords": ["테스트"],
            "related_terms": ["관련"],
            "synonyms": ["동의어"],
            "context_keywords": ["맥락"],
            "confidence": 0.8
        }
        '''

        with patch('src.keyword_generator.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from src.keyword_generator import generate_keywords_for_topic

            result = await generate_keywords_for_topic("테스트 주제")

            assert result.topic == "테스트 주제"
            assert result.confidence_score == 0.8

    @pytest.mark.unit
    @pytest.mark.parametrize("confidence_input,expected_output", [
        (0.95, 0.95),  # 정상 범위
        (1.2, 1.0),    # 상한 초과 -> 1.0으로 제한
        (-0.1, 0.0),   # 하한 미만 -> 0.0으로 제한
        (0.0, 0.0),    # 경계값
        (1.0, 1.0),    # 경계값
    ])
    def test_confidence_score_validation(self, confidence_input, expected_output):
        """신뢰도 점수 검증 테스트"""
        with patch('src.keyword_generator.AsyncOpenAI'):
            from src.keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")

            mock_response = f'''
            {{
                "primary_keywords": ["test"],
                "related_terms": ["test"],
                "synonyms": ["test"],
                "context_keywords": ["test"],
                "confidence": {confidence_input}
            }}
            '''

            result = generator._parse_response("test", mock_response, 1.0)
            assert result.confidence_score == expected_output

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_topic_validation(self):
        """빈 주제 입력 검증 테스트"""
        with patch('keyword_generator.AsyncOpenAI'):
            from keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")

            with pytest.raises(ValueError, match="주제가 비어있습니다"):
                await generator.generate_keywords("")

    @pytest.mark.unit
    def test_create_keyword_generator_function(self):
        """create_keyword_generator 함수 테스트"""
        with patch('keyword_generator.AsyncOpenAI'):
            from keyword_generator import create_keyword_generator

            # 기본 생성
            generator1 = create_keyword_generator(api_key="test_key")
            assert generator1.model == "gpt-4o-mini"

            # 커스텀 모델 생성
            generator2 = create_keyword_generator(api_key="test_key", model="gpt-4")
            assert generator2.model == "gpt-4"


class TestKeywordGeneratorIntegration:
    """키워드 생성기 통합 테스트"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_keyword_generation_flow(self):
        """전체 키워드 생성 플로우 테스트"""
        # 실제와 유사한 응답 시뮬레이션
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "primary_keywords": ["인공지능", "AI", "머신러닝", "딥러닝"],
            "related_terms": ["신경망", "자연어처리", "컴퓨터비전"],
            "synonyms": ["Artificial Intelligence", "기계학습"],
            "context_keywords": ["기술혁신", "자동화", "데이터과학"],
            "confidence": 0.92
        }
        '''

        with patch('src.keyword_generator.AsyncOpenAI') as mock_openai:
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from src.keyword_generator import create_keyword_generator

            generator = create_keyword_generator(api_key="test_key")
            result = await generator.generate_keywords("AI 기술 발전")

            # 결과 검증
            assert result.topic == "AI 기술 발전"
            assert len(result.primary_keywords) == 4
            assert len(result.related_terms) == 3
            assert len(result.synonyms) == 2
            assert len(result.context_keywords) == 3
            assert result.confidence_score == 0.92
            assert result.generation_time > 0

            # 전체 키워드 수 확인
            all_keywords = generator.get_all_keywords(result)
            assert len(all_keywords) == 12  # 중복 제거된 전체 키워드 수

            # 요약 포맷팅 확인
            summary = generator.format_keywords_summary(result)
            assert "AI 기술 발전" in summary
            assert "92%" in summary

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_json_parsing_with_code_blocks(self):
        """코드 블록이 포함된 JSON 파싱 테스트"""
        mock_response_with_code_blocks = '''
        다음은 요청하신 키워드입니다:
        
        ```json
        {
            "primary_keywords": ["블록체인", "NFT", "웹3"],
            "related_terms": ["스마트계약", "메타버스"],
            "synonyms": ["Web3", "크립토"],
            "context_keywords": ["디지털자산", "탈중앙화"],
            "confidence": 0.88
        }
        ```
        
        이상입니다.
        '''

        with patch('src.keyword_generator.AsyncOpenAI'):
            from src.keyword_generator import create_keyword_generator
            
            generator = create_keyword_generator(api_key="test_key")
            result = generator._parse_response("블록체인 기술", mock_response_with_code_blocks, 1.5)
            
            assert result.topic == "블록체인 기술"
            assert "블록체인" in result.primary_keywords
            assert "NFT" in result.primary_keywords
            assert result.confidence_score == 0.88


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v"])