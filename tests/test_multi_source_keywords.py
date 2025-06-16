"""
멀티 소스 키워드 생성 시스템 테스트.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import os
import json

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.keyword_generation import (
    MultiSourceKeywordManager,
    GPTKeywordExtractor,
    GrokKeywordExtractor,
    PerplexityKeywordExtractor,
    KeywordSimilarityAnalyzer,
    KeywordItem,
    KeywordImportance,
    generate_keywords_for_topic,
    get_keyword_generation_status,
    generate_multi_source_keywords
)
from src.keyword_generation.manager import MultiSourceKeywordResult
from src.keyword_generation.base import KeywordExtractionResult
from src.models import KeywordResult


class TestKeywordSimilarityAnalyzer:
    """키워드 유사도 분석기 테스트."""

    @pytest.fixture
    def analyzer(self):
        """테스트용 유사도 분석기."""
        return KeywordSimilarityAnalyzer(similarity_threshold=0.85)

    def test_exact_match(self, analyzer):
        """완전 일치 테스트."""
        similar = analyzer.find_similar_keywords(
            "artificial intelligence",
            ["AI", "Artificial Intelligence", "machine learning"]
        )

        # 'exact' 타입의 일치 결과를 필터링합니다.
        exact_matches = [res for res in similar if res[1].similarity_type == "exact"]

        # 정확히 일치하는 결과가 하나인지 확인합니다.
        assert len(exact_matches) == 1
        assert exact_matches[0][0].lower() == "artificial intelligence"


    def test_synonym_match(self, analyzer):
        """동의어 매칭 테스트."""
        similar = analyzer.find_similar_keywords(
            "AI",
            ["artificial intelligence", "machine intelligence", "deep learning"]
        )

        assert len(similar) >= 1
        assert any(result[1].similarity_type == "synonym" for result in similar)

    def test_fuzzy_match(self, analyzer):
        """문자열 유사도 매칭 테스트."""
        similar = analyzer.find_similar_keywords(
            "gpt-4",
            ["gpt4", "gpt-3", "gpt_4", "bert"]
        )

        assert len(similar) >= 2  # gpt4와 gpt_4는 매칭되어야 함
        fuzzy_matches = [r for r in similar if r[1].similarity_type == "fuzzy"]
        assert len(fuzzy_matches) > 0

    def test_merge_similar_keywords(self, analyzer):
        """키워드 병합 테스트."""
        keyword_lists = {
            "GPT": [
                KeywordItem("AI", sources=["GPT"], importance=KeywordImportance.HIGH),
                KeywordItem("machine learning", sources=["GPT"], importance=KeywordImportance.NORMAL)
            ],
            "Perplexity": [
                KeywordItem("artificial intelligence", sources=["Perplexity"], importance=KeywordImportance.HIGH),
                KeywordItem("ML", sources=["Perplexity"], importance=KeywordImportance.NORMAL)
            ]
        }

        merged = analyzer.merge_similar_keywords(keyword_lists)

        # AI와 artificial intelligence가 병합되어야 함
        ai_keywords = [kw for kw in merged if "ai" in kw.keyword.lower() or "artificial" in kw.keyword.lower()]
        assert len(ai_keywords) == 1
        assert len(ai_keywords[0].sources) == 2  # 두 소스에서 나왔으므로
        assert ai_keywords[0].importance == KeywordImportance.HIGH


@pytest.mark.asyncio
class TestMultiSourceKeywordManager:
    """멀티 소스 키워드 매니저 테스트."""

    @pytest.fixture
    def mock_extractors(self):
        """모의 추출기들."""
        # GPT 추출기 모의
        gpt_extractor = MagicMock(spec=GPTKeywordExtractor)
        gpt_extractor.name = "GPT"
        gpt_extractor.extract_keywords = AsyncMock()

        # Grok 추출기 모의
        grok_extractor = MagicMock(spec=GrokKeywordExtractor)
        grok_extractor.name = "Grok"
        grok_extractor.extract_keywords = AsyncMock()

        # Perplexity 추출기 모의
        perplexity_extractor = MagicMock(spec=PerplexityKeywordExtractor)
        perplexity_extractor.name = "Perplexity"
        perplexity_extractor.extract_keywords = AsyncMock()

        return [gpt_extractor, grok_extractor, perplexity_extractor]

    async def test_generate_keywords_success(self, mock_extractors):
        """성공적인 키워드 생성 테스트."""
        # 각 추출기의 응답 설정
        mock_extractors[0].extract_keywords.return_value = KeywordExtractionResult(
            keywords=[
                KeywordItem("quantum computing", sources=["GPT"], importance=KeywordImportance.HIGH),
                KeywordItem("qubits", sources=["GPT"], importance=KeywordImportance.NORMAL)
            ],
            source_name="GPT",
            extraction_time=1.0
        )

        mock_extractors[1].extract_keywords.return_value = KeywordExtractionResult(
            keywords=[
                KeywordItem("#quantumcomputing", sources=["Grok"], importance=KeywordImportance.HIGH),
                KeywordItem("quantum computing", sources=["Grok"], importance=KeywordImportance.HIGH)
            ],
            source_name="Grok",
            extraction_time=0.5
        )

        mock_extractors[2].extract_keywords.return_value = KeywordExtractionResult(
            keywords=[
                KeywordItem("quantum algorithms", sources=["Perplexity"], importance=KeywordImportance.NORMAL),
                KeywordItem("qubits", sources=["Perplexity"], importance=KeywordImportance.NORMAL)
            ],
            source_name="Perplexity",
            extraction_time=0.8
        )

        # 매니저 생성 및 실행
        manager = MultiSourceKeywordManager(extractors=mock_extractors)
        result = await manager.generate_keywords("quantum computing")

        # 검증
        assert len(result.keywords) > 0
        # 'total_sources' 대신 'source_results'의 길이를 확인합니다.
        assert len(result.source_results) == 3

        # quantum computing 관련 키워드 (여러 형태 가능)는 2개 소스에서 나왔으므로 HIGH여야 함
        quantum_kw = next((kw for kw in result.keywords if 
                          "quantum computing" in kw.keyword.lower() or 
                          "quantumcomputing" in kw.keyword.lower()), None)
        assert quantum_kw is not None
        assert quantum_kw.importance == KeywordImportance.HIGH
        assert len(quantum_kw.sources) >= 2

        # qubits도 2개 소스에서 나왔으므로 HIGH여야 함
        qubits_kw = next((kw for kw in result.keywords if "qubits" in kw.keyword.lower()), None)
        assert qubits_kw is not None
        assert qubits_kw.importance == KeywordImportance.HIGH

    async def test_partial_failure_handling(self, mock_extractors):
        """일부 추출기 실패 시 처리 테스트."""
        # GPT는 성공
        mock_extractors[0].extract_keywords.return_value = KeywordExtractionResult(
            keywords=[KeywordItem("test keyword", sources=["GPT"])],
            source_name="GPT",
            extraction_time=1.0
        )

        # Grok은 실패
        mock_extractors[1].extract_keywords.side_effect = Exception("API Error")

        # Perplexity는 성공
        mock_extractors[2].extract_keywords.return_value = KeywordExtractionResult(
            keywords=[KeywordItem("test keyword", sources=["Perplexity"])],
            source_name="Perplexity",
            extraction_time=0.8
        )

        manager = MultiSourceKeywordManager(extractors=mock_extractors)
        result = await manager.generate_keywords("test")

        # 3개 소스 모두 결과에 포함되어야 함 (성공/실패 여부 포함)
        assert len(result.source_results) == 3
        assert result.source_results["GPT"].is_success
        assert not result.source_results["Grok"].is_success
        assert result.source_results["Perplexity"].is_success

        # test keyword는 2개 소스에서 나왔으므로 HIGH
        assert len(result.keywords) > 0
        assert result.keywords[0].importance == KeywordImportance.HIGH


class TestSystemIntegration:
    """시스템 통합 테스트."""

    @pytest.mark.asyncio
    async def test_generate_keywords_for_topic_compatibility(self):
        """기존 인터페이스 호환성 테스트."""
        with patch('src.keyword_generation.get_keyword_manager') as mock_get_manager:
            # 모의 매니저 설정
            mock_manager = MagicMock()
            mock_manager.extractors = [MagicMock(name="GPT")]
            mock_manager.generate_keywords = AsyncMock()

            mock_result = MultiSourceKeywordResult(
                keywords=[
                    KeywordItem("test1", sources=["GPT"], importance=KeywordImportance.HIGH),
                    KeywordItem("test2", sources=["GPT"], importance=KeywordImportance.NORMAL)
                ],
                source_results={},
                total_time=1.0,
                merged_count=0,
                high_importance_count=1,
                normal_importance_count=1,
                low_importance_count=0
            )

            mock_manager.generate_keywords.return_value = mock_result
            mock_get_manager.return_value = mock_manager

            # 기존 인터페이스로 호출
            result = await generate_keywords_for_topic("test topic")

            # KeywordResult 타입이어야 함
            assert isinstance(result, KeywordResult)
            assert result.topic == "test topic"
            assert len(result.primary_keywords) > 0

    def test_get_keyword_generation_status(self):
        """상태 확인 함수 테스트."""
        with patch('src.keyword_generation.get_keyword_manager') as mock_get_manager:
            mock_manager = MagicMock()

            # --- FIX: 모의 객체의 'name' 속성을 명시적으로 설정 ---
            gpt_mock = MagicMock()
            gpt_mock.name = "GPT"

            perplexity_mock = MagicMock()
            perplexity_mock.name = "Perplexity"

            mock_manager.extractors = [gpt_mock, perplexity_mock]
            # --- End of FIX ---

            mock_manager.similarity_analyzer = MagicMock(similarity_threshold=0.85)
            mock_get_manager.return_value = mock_manager

            with patch('src.keyword_generation.config') as mock_config:
                mock_config.get_openai_api_key.return_value = "test_key"
                mock_config.get_perplexity_api_key.return_value = "test_key"

                status = get_keyword_generation_status()

                # 'get_keyword_generation_status' 함수가 이름 문자열 리스트를 반환한다고 가정
                active_extractor_names = status['active_extractors']

                assert status['total_extractors'] == 2
                assert 'GPT' in active_extractor_names
                assert 'Perplexity' in active_extractor_names
                assert status['similarity_threshold'] == 0.85
                assert status['available_apis']['gpt'] is True
                assert status['available_apis']['perplexity'] is True


async def test_multi_source_system():
    """멀티 소스 키워드 시스템을 테스트합니다."""
    print("=" * 60)
    print("🧪 멀티 소스 키워드 시스템 테스트")
    print("=" * 60)

    # 1. 시스템 상태 확인
    print("\n[1] 시스템 상태 확인")
    status = get_keyword_generation_status()
    # 상태 확인 시 active_extractors가 모의 객체일 수 있으므로 이름으로 변환
    active_extractor_names = [e.name if isinstance(e, MagicMock) else e for e in status['active_extractors']]
    print(f"활성 추출기: {', '.join(active_extractor_names)}")
    print(f"총 추출기 수: {status['total_extractors']}")
    print(f"유사도 임계값: {status['similarity_threshold']:.0%}")

    # 2. 키워드 생성 테스트
    test_topics = ["quantum computing", "인공지능", "blockchain"]

    for topic in test_topics:
        print(f"\n[2] '{topic}' 키워드 생성 중...")

        try:
            result = await generate_multi_source_keywords(topic)

            print(f"\n✅ 생성 완료!")
            print(f"- 총 키워드: {len(result.keywords)}개")
            print(f"- HIGH: {result.high_importance_count}개")
            print(f"- NORMAL: {result.normal_importance_count}개")
            print(f"- LOW: {result.low_importance_count}개")
            print(f"- 중복 병합: {result.merged_count}개")
            print(f"- 소요 시간: {result.total_time:.2f}초")

            # 상위 키워드 출력
            print("\n🎯 상위 키워드:")
            for i, kw in enumerate(result.keywords[:5], 1):
                sources = ", ".join(kw.sources)
                print(f"{i}. {kw.keyword} ({kw.importance.value}) - 소스: {sources}")

            # JSON 출력 (요구사항대로)
            json_output = []
            for kw in result.keywords[:10]:
                json_output.append({
                    "keyword": kw.keyword,
                    "sources": kw.sources,
                    "importance": kw.importance.value
                })

            print("\n📄 JSON 출력:")
            print(json.dumps(json_output, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"❌ 오류 발생: {e}")

        print("\n" + "-" * 60)

    print("\n✨ 테스트 완료!")


if __name__ == "__main__":
    # 이 파일이 직접 실행될 때 테스트 시스템을 호출합니다.
    # pytest로 실행될 때는 아래 코드가 실행되지 않습니다.
    asyncio.run(test_multi_source_system())
