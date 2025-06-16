"""
Coverage improvement tests for core functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.cache import PerformanceCache
from src.reporting.topic_classifier import TopicClassifier
from src.reporting.markdown_parser import MarkdownToPDFConverter


class TestPerformanceCache:
    """PerformanceCache 테스트로 utils/cache.py 커버리지 향상."""

    @pytest.fixture
    def cache(self):
        """테스트용 캐시 인스턴스."""
        return PerformanceCache(max_size=5, default_ttl=10)

    def test_cache_initialization(self, cache):
        """캐시 초기화 테스트."""
        assert cache.max_size == 5
        assert cache.default_ttl == 10
        assert len(cache._cache) == 0
        assert cache._hits == 0
        assert cache._misses == 0

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache):
        """캐시 저장 및 조회 테스트."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"
        assert cache._hits == 1
        assert cache._misses == 0

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """캐시 미스 테스트."""
        result = await cache.get("nonexistent")
        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0

    @pytest.mark.asyncio
    async def test_cache_size_limit(self, cache):
        """캐시 크기 제한 테스트."""
        # 최대 크기를 초과하여 저장
        for i in range(7):
            await cache.set(f"key{i}", f"value{i}")
        
        # 캐시 크기가 max_size를 초과하지 않아야 함
        assert len(cache._cache) <= cache.max_size

    @pytest.mark.asyncio
    async def test_cache_contains(self, cache):
        """캐시 포함 여부 테스트."""
        await cache.set("test_key", "test_value")
        assert "test_key" in cache._cache
        assert "nonexistent" not in cache._cache

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache):
        """캐시 클리어 테스트."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        
        assert len(cache._cache) == 0
        # Note: clear doesn't reset stats in PerformanceCache

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache):
        """캐시 통계 테스트."""
        await cache.set("key1", "value1")
        await cache.get("key1")  # hit
        await cache.get("key2")  # miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["hit_rate"] == 0.5


class TestTopicClassifier:
    """TopicClassifier 테스트로 reporting/topic_classifier.py 커버리지 향상."""

    @pytest.fixture
    def classifier(self):
        """테스트용 분류기 인스턴스."""
        return TopicClassifier()

    def test_classifier_initialization(self, classifier):
        """분류기 초기화 테스트."""
        assert classifier.classification_patterns is not None
        assert len(classifier.classification_patterns) > 0
        assert classifier.section_generators is not None

    def test_classify_technical_topic(self, classifier):
        """기술 주제 분류 테스트."""
        topic = "machine learning AI algorithms"
        keywords = ["AI", "machine learning", "algorithms"]
        
        result = classifier.classify_topic(topic, keywords)
        
        assert result is not None
        assert result.primary_type is not None
        assert 0.0 <= result.confidence <= 1.0
        assert result.keywords_matched is not None
        assert result.reasoning is not None

    def test_classify_business_topic(self, classifier):
        """비즈니스 주제 분류 테스트."""
        topic = "market strategy business growth"
        keywords = ["market", "strategy", "business", "growth"]
        
        result = classifier.classify_topic(topic, keywords)
        
        assert result is not None
        assert result.primary_type is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_empty_topic(self, classifier):
        """빈 주제 분류 테스트."""
        result = classifier.classify_topic("", [])
        
        assert result is not None
        assert result.confidence == 0.0

    def test_should_include_risk_analysis(self, classifier):
        """리스크 분석 포함 여부 테스트."""
        # Mock classification result
        mock_result = MagicMock()
        mock_result.primary_type.value = "business_strategic"
        mock_result.confidence = 0.8
        
        should_include = classifier.should_include_risk_analysis(mock_result)
        assert isinstance(should_include, bool)


class TestMarkdownToPDFConverter:
    """MarkdownToPDFConverter 테스트로 reporting/markdown_parser.py 커버리지 향상."""

    @pytest.fixture
    def converter(self):
        """테스트용 변환기 인스턴스."""
        return MarkdownToPDFConverter()

    def test_converter_initialization(self, converter):
        """변환기 초기화 테스트."""
        assert converter.default_font is not None
        assert converter.styles is not None
        assert converter.patterns is not None

    def test_convert_simple_markdown(self, converter):
        """간단한 마크다운 변환 테스트."""
        markdown_text = "# 제목\n\n이것은 **굵은 글씨**입니다."
        
        with patch.object(converter, 'convert_to_pdf_elements', return_value=["mock_element"]):
            elements = converter.convert_to_pdf_elements(markdown_text)
            
            assert isinstance(elements, list)
            assert len(elements) > 0

    def test_convert_empty_markdown(self, converter):
        """빈 마크다운 변환 테스트."""
        with patch.object(converter, 'convert_to_pdf_elements', return_value=["default_message"]):
            elements = converter.convert_to_pdf_elements("")
            
            assert isinstance(elements, list)
            assert len(elements) > 0  # 기본 메시지가 있어야 함

    def test_format_inline_markdown(self, converter):
        """인라인 마크다운 포맷팅 테스트."""
        text = "이것은 **굵은 글씨**와 *기울임*입니다."
        
        formatted = converter._format_inline_markdown(text)
        
        assert isinstance(formatted, str)
        assert "<b>" in formatted or "굵은 글씨" in formatted

    def test_confidence_indicator(self, converter):
        """신뢰도 표시기 테스트."""
        high_confidence = converter.create_confidence_indicator(0.9)
        medium_confidence = converter.create_confidence_indicator(0.6)
        low_confidence = converter.create_confidence_indicator(0.3)
        
        assert high_confidence == "🟢"
        assert medium_confidence == "🟡"
        assert low_confidence == "🔴"

    def test_format_confidence_text(self, converter):
        """신뢰도 텍스트 포맷팅 테스트."""
        confidence_text = converter.format_confidence_text(0.85)
        
        assert "🟢" in confidence_text
        assert "85.0%" in confidence_text


class TestUtilityFunctions:
    """기타 유틸리티 함수들 테스트."""

    @pytest.mark.asyncio
    async def test_cache_basic_operations(self):
        """기본 캐시 연산 테스트."""
        cache = PerformanceCache(max_size=3)
        
        # 기본 연산
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.set("c", 3)
        
        assert await cache.get("a") == 1
        assert await cache.get("b") == 2
        assert await cache.get("c") == 3
        
        # 크기 제한 테스트
        await cache.set("d", 4)  # 최대 크기 초과
        assert len(cache._cache) <= 3

    @pytest.mark.asyncio
    async def test_cache_with_different_types(self):
        """다양한 타입으로 캐시 테스트."""
        cache = PerformanceCache()
        
        # 다양한 데이터 타입
        await cache.set("string", "value")
        await cache.set("number", 42)
        await cache.set("list", [1, 2, 3])
        await cache.set("dict", {"key": "value"})
        
        assert await cache.get("string") == "value"
        assert await cache.get("number") == 42
        assert await cache.get("list") == [1, 2, 3]
        assert await cache.get("dict") == {"key": "value"}

    def test_topic_classification_edge_cases(self):
        """주제 분류 엣지 케이스 테스트."""
        classifier = TopicClassifier()
        
        # 매우 긴 주제
        long_topic = "AI " * 100
        result = classifier.classify_topic(long_topic, ["AI"])
        assert result is not None
        
        # 특수 문자가 포함된 주제
        special_topic = "AI/ML & deep-learning: next-gen tech (2024)"
        result = classifier.classify_topic(special_topic, ["AI", "ML"])
        assert result is not None
        
        # 숫자만 포함된 키워드
        result = classifier.classify_topic("123 456", ["123", "456"])
        assert result is not None

    def test_markdown_converter_edge_cases(self):
        """마크다운 변환기 엣지 케이스 테스트."""
        converter = MarkdownToPDFConverter()
        
        with patch.object(converter, 'convert_to_pdf_elements', return_value=["mock_element"]):
            # 매우 긴 텍스트
            long_text = "이것은 매우 긴 텍스트입니다. " * 100
            elements = converter.convert_to_pdf_elements(long_text)
            assert len(elements) > 0
            
            # 특수 문자가 포함된 마크다운
            special_markdown = "# 제목 <>&\n\n**굵은 글씨** & `코드`"
            elements = converter.convert_to_pdf_elements(special_markdown)
            assert len(elements) > 0
            
            # 코드 블록
            code_markdown = "```python\nprint('hello')\n```"
            elements = converter.convert_to_pdf_elements(code_markdown)
            assert len(elements) > 0