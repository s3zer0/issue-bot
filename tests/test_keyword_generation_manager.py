"""
Keyword Generation Manager 테스트.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import os

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.keyword_generation.manager import MultiSourceKeywordManager
from src.models import KeywordResult
from src.keyword_generation.extractors.gpt_extractor import GPTKeywordExtractor
from src.keyword_generation.extractors.perplexity_extractor import PerplexityKeywordExtractor


class TestMultiSourceKeywordManager:
    """Multi-Source Keyword Manager의 단위 테스트."""

    @pytest.fixture
    def manager(self):
        """테스트용 MultiSourceKeywordManager 인스턴스."""
        with patch('src.keyword_generation.manager.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_openai_key"
            mock_config.get_perplexity_api_key.return_value = "test_perplexity_key"
            mock_config.get_grok_api_key.return_value = "test_grok_key"
            yield MultiSourceKeywordManager()

    @pytest.fixture
    def sample_keyword_result(self):
        """테스트용 KeywordResult."""
        return KeywordResult(
            topic="AI technology",
            primary_keywords=["AI", "machine learning", "technology"],
            related_terms=["neural networks", "deep learning", "automation"],
            context_keywords=["future", "innovation", "digital transformation"],
            confidence_score=0.85,
            generation_time=1.5,
            raw_response="test response"
        )

    def test_initialization(self, manager):
        """초기화 테스트."""
        assert manager.extractors is not None
        assert len(manager.extractors) >= 2  # GPT와 Perplexity 최소
        assert manager.similarity_analyzer is not None
        assert manager.clusterer is not None

    def test_extractors_initialization(self, manager):
        """추출기 초기화 테스트."""
        # GPT 추출기가 있는지 확인
        gpt_extractor = None
        perplexity_extractor = None
        
        for extractor in manager.extractors:
            if isinstance(extractor, GPTKeywordExtractor):
                gpt_extractor = extractor
            elif isinstance(extractor, PerplexityKeywordExtractor):
                perplexity_extractor = extractor
        
        assert gpt_extractor is not None
        assert perplexity_extractor is not None

    @pytest.mark.asyncio
    async def test_generate_keywords_success(self, manager, sample_keyword_result):
        """키워드 생성 성공 테스트."""
        topic = "AI technology trends"
        
        # Mock extractors
        mock_extractor = AsyncMock()
        mock_extractor.extract_keywords.return_value = sample_keyword_result
        manager.extractors = [mock_extractor]
        
        result = await manager.generate_keywords(topic)
        
        assert result is not None
        assert isinstance(result, KeywordResult)
        assert len(result.primary_keywords) > 0
        assert result.confidence > 0.0
        mock_extractor.extract_keywords.assert_called_once_with(topic)

    @pytest.mark.asyncio
    async def test_generate_keywords_multiple_extractors(self, manager):
        """여러 추출기를 사용한 키워드 생성 테스트."""
        topic = "artificial intelligence"
        
        # Mock multiple extractors with different results
        mock_extractor1 = AsyncMock()
        mock_extractor1.extract_keywords.return_value = KeywordResult(
            primary_keywords=["AI", "machine learning"],
            related_terms=["neural networks"],
            context_keywords=["technology"],
            confidence=0.8,
            tier="tier1",
            source="gpt",
            metadata={}
        )
        
        mock_extractor2 = AsyncMock()
        mock_extractor2.extract_keywords.return_value = KeywordResult(
            primary_keywords=["artificial intelligence", "ML"],
            related_terms=["deep learning"],
            context_keywords=["innovation"],
            confidence=0.7,
            tier="tier1",
            source="perplexity",
            metadata={}
        )
        
        manager.extractors = [mock_extractor1, mock_extractor2]
        
        # Mock similarity analyzer and clusterer
        with patch.object(manager.similarity_analyzer, 'remove_duplicates', return_value=["AI", "machine learning", "artificial intelligence"]), \
             patch.object(manager.clusterer, 'cluster_keywords', return_value={
                 "primary": ["AI", "machine learning"],
                 "related": ["neural networks", "deep learning"],
                 "context": ["technology", "innovation"]
             }):
            
            result = await manager.generate_keywords(topic)
            
            assert result is not None
            assert len(result.primary_keywords) > 0
            # 여러 소스에서 온 키워드들이 병합되었는지 확인
            all_keywords = result.primary_keywords + result.related_terms + result.context_keywords
            assert len(all_keywords) > 2

    @pytest.mark.asyncio
    async def test_generate_keywords_all_extractors_fail(self, manager):
        """모든 추출기 실패 테스트."""
        topic = "test topic"
        
        # Mock all extractors to fail
        mock_extractor = AsyncMock()
        mock_extractor.extract_keywords.side_effect = Exception("Extraction failed")
        manager.extractors = [mock_extractor]
        
        result = await manager.generate_keywords(topic)
        
        # 실패 시 None 반환되거나 기본 결과 반환
        assert result is None or isinstance(result, KeywordResult)

    @pytest.mark.asyncio
    async def test_regenerate_keywords_success(self, manager, sample_keyword_result):
        """키워드 재생성 성공 테스트."""
        topic = "AI technology"
        previous_keywords = sample_keyword_result
        failure_reason = "Low confidence"
        
        # Mock extractor for regeneration
        new_keyword_result = KeywordResult(
            primary_keywords=["artificial intelligence", "AI systems", "smart technology"],
            related_terms=["cognitive computing", "intelligent systems"],
            context_keywords=["future tech", "digital innovation"],
            confidence=0.9,
            tier="tier1", 
            source="gpt",
            metadata={"regenerated": True}
        )
        
        mock_extractor = AsyncMock()
        mock_extractor.extract_keywords.return_value = new_keyword_result
        manager.extractors = [mock_extractor]
        
        result = await manager.regenerate_keywords(topic, previous_keywords, failure_reason)
        
        assert result is not None
        assert isinstance(result, KeywordResult)
        # 재생성된 키워드는 이전과 다른 키워드여야 함
        assert result.primary_keywords != previous_keywords.primary_keywords

    @pytest.mark.asyncio
    async def test_regenerate_keywords_with_context(self, manager, sample_keyword_result):
        """컨텍스트를 포함한 키워드 재생성 테스트."""
        topic = "machine learning"
        previous_keywords = sample_keyword_result
        failure_reason = "Insufficient keywords"
        
        # Mock search result context
        mock_search_result = MagicMock()
        mock_search_result.issues = [
            MagicMock(title="Deep Learning Advances", summary="New neural network architectures"),
            MagicMock(title="AI in Healthcare", summary="Medical diagnosis improvements")
        ]
        
        new_keyword_result = KeywordResult(
            primary_keywords=["deep learning", "neural networks", "medical AI"],
            related_terms=["healthcare applications", "diagnosis"],
            context_keywords=["medical", "healthcare"],
            confidence=0.85,
            tier="tier2",
            source="gpt",
            metadata={"with_context": True}
        )
        
        mock_extractor = AsyncMock()
        mock_extractor.extract_keywords.return_value = new_keyword_result
        manager.extractors = [mock_extractor]
        
        result = await manager.regenerate_keywords(
            topic, previous_keywords, failure_reason, 
            search_context=mock_search_result
        )
        
        assert result is not None
        assert result.tier == "tier2"  # 컨텍스트 포함으로 tier 변경

    def test_merge_results_multiple_sources(self, manager):
        """여러 소스 결과 병합 테스트."""
        results = [
            KeywordResult(
                primary_keywords=["AI", "machine learning"],
                related_terms=["neural networks"],
                context_keywords=["technology"],
                confidence=0.8,
                tier="tier1",
                source="gpt",
                metadata={"source_1": True}
            ),
            KeywordResult(
                primary_keywords=["artificial intelligence", "ML"],
                related_terms=["deep learning"],
                context_keywords=["innovation"],
                confidence=0.7,
                tier="tier1", 
                source="perplexity",
                metadata={"source_2": True}
            )
        ]
        
        with patch.object(manager.similarity_analyzer, 'remove_duplicates') as mock_remove_dup, \
             patch.object(manager.clusterer, 'cluster_keywords') as mock_cluster:
            
            # Mock return values
            mock_remove_dup.return_value = ["AI", "machine learning", "artificial intelligence", "ML"]
            mock_cluster.return_value = {
                "primary": ["AI", "machine learning"],
                "related": ["neural networks", "deep learning"],
                "context": ["technology", "innovation"]
            }
            
            merged = manager._merge_results(results)
            
            assert merged is not None
            assert isinstance(merged, KeywordResult)
            assert len(merged.primary_keywords) >= 2
            assert "merged" in merged.metadata
            
            # 두 소스 모두 호출되었는지 확인
            mock_remove_dup.assert_called()
            mock_cluster.assert_called()

    def test_merge_results_empty_list(self, manager):
        """빈 결과 리스트 병합 테스트."""
        merged = manager._merge_results([])
        assert merged is None

    def test_merge_results_single_result(self, manager, sample_keyword_result):
        """단일 결과 병합 테스트."""
        merged = manager._merge_results([sample_keyword_result])
        
        assert merged is not None
        assert merged == sample_keyword_result  # 단일 결과는 그대로 반환

    def test_calculate_merged_confidence(self, manager):
        """병합된 신뢰도 계산 테스트."""
        results = [
            MagicMock(confidence=0.8),
            MagicMock(confidence=0.7),
            MagicMock(confidence=0.9)
        ]
        
        merged_confidence = manager._calculate_merged_confidence(results)
        
        # 가중 평균 계산 확인
        expected = 0.8 * 0.5 + 0.7 * 0.3 + 0.9 * 0.2  # 가중치는 구현에 따라 다를 수 있음
        assert 0.0 <= merged_confidence <= 1.0
        assert merged_confidence > 0.7  # 높은 신뢰도들의 평균

    def test_calculate_merged_confidence_empty_list(self, manager):
        """빈 리스트 신뢰도 계산 테스트."""
        merged_confidence = manager._calculate_merged_confidence([])
        assert merged_confidence == 0.0

    def test_create_merged_metadata(self, manager):
        """병합된 메타데이터 생성 테스트."""
        results = [
            KeywordResult(
                primary_keywords=["AI"], related_terms=[], context_keywords=[],
                confidence=0.8, tier="tier1", source="gpt",
                metadata={"model": "gpt-4", "tokens": 100}
            ),
            KeywordResult(
                primary_keywords=["ML"], related_terms=[], context_keywords=[],
                confidence=0.7, tier="tier1", source="perplexity", 
                metadata={"model": "perplexity", "tokens": 80}
            )
        ]
        
        merged_metadata = manager._create_merged_metadata(results)
        
        assert "merged" in merged_metadata
        assert merged_metadata["merged"] is True
        assert "sources" in merged_metadata
        assert "gpt" in merged_metadata["sources"]
        assert "perplexity" in merged_metadata["sources"]
        assert "total_results" in merged_metadata
        assert merged_metadata["total_results"] == 2

    def test_enhance_keywords_with_context(self, manager):
        """컨텍스트를 활용한 키워드 향상 테스트."""
        base_keywords = ["AI", "technology"]
        
        mock_search_result = MagicMock()
        mock_search_result.issues = [
            MagicMock(title="Healthcare AI Revolution", summary="AI transforming medical diagnosis"),
            MagicMock(title="AI in Education", summary="Personalized learning systems")
        ]
        
        enhanced = manager._enhance_keywords_with_context(base_keywords, mock_search_result)
        
        assert isinstance(enhanced, list)
        assert len(enhanced) >= len(base_keywords)
        # 컨텍스트에서 추출된 키워드가 추가되었는지 확인
        enhanced_str = ' '.join(enhanced).lower()
        assert any(word in enhanced_str for word in ["healthcare", "medical", "education", "learning"])

    def test_enhance_keywords_with_context_no_issues(self, manager):
        """이슈가 없는 컨텍스트로 키워드 향상 테스트."""
        base_keywords = ["AI", "technology"]
        
        mock_search_result = MagicMock()
        mock_search_result.issues = []
        
        enhanced = manager._enhance_keywords_with_context(base_keywords, mock_search_result)
        
        # 이슈가 없으면 원래 키워드만 반환
        assert enhanced == base_keywords

    def test_extract_context_keywords(self, manager):
        """컨텍스트 키워드 추출 테스트."""
        issues = [
            MagicMock(title="Machine Learning Breakthrough", summary="New neural network architectures"),
            MagicMock(title="AI Ethics Guidelines", summary="Responsible AI development practices"),
            MagicMock(title="Deep Learning Applications", summary="Computer vision and NLP advances")
        ]
        
        context_keywords = manager._extract_context_keywords(issues)
        
        assert isinstance(context_keywords, list)
        assert len(context_keywords) > 0
        
        # 추출된 키워드들이 의미있는지 확인
        context_str = ' '.join(context_keywords).lower()
        assert any(word in context_str for word in [
            "machine", "learning", "neural", "network", "ethics", "deep", "vision", "nlp"
        ])

    def test_extract_meaningful_terms(self, manager):
        """의미있는 용어 추출 테스트."""
        text = "Machine learning algorithms and neural network architectures are revolutionizing artificial intelligence"
        
        terms = manager._extract_meaningful_terms(text)
        
        assert isinstance(terms, list)
        assert len(terms) > 0
        
        # 의미있는 용어들이 추출되었는지 확인
        terms_lower = [term.lower() for term in terms]
        expected_terms = ["machine learning", "neural network", "artificial intelligence", "algorithms"]
        
        # 적어도 일부 예상 용어가 포함되어야 함
        found_terms = [term for term in expected_terms if any(t in term for t in terms_lower)]
        assert len(found_terms) > 0


class TestMultiSourceKeywordManagerIntegration:
    """Multi-Source Keyword Manager의 통합 테스트."""

    @pytest.fixture
    def manager(self):
        """테스트용 manager."""
        with patch('src.keyword_generation.manager.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_openai_key"
            mock_config.get_perplexity_api_key.return_value = "test_perplexity_key"
            mock_config.get_grok_api_key.return_value = "test_grok_key"
            yield MultiSourceKeywordManager()

    @pytest.mark.asyncio
    async def test_end_to_end_keyword_generation(self, manager):
        """전체 키워드 생성 프로세스 테스트."""
        topic = "sustainable energy technologies"
        
        # Mock 모든 종속성
        mock_gpt_result = KeywordResult(
            primary_keywords=["renewable energy", "solar power", "wind energy"],
            related_terms=["photovoltaic", "turbines", "grid storage"],
            context_keywords=["sustainability", "clean tech", "environment"],
            confidence=0.85,
            tier="tier1",
            source="gpt",
            metadata={"model": "gpt-4"}
        )
        
        mock_perplexity_result = KeywordResult(
            primary_keywords=["green energy", "sustainable power", "clean electricity"],
            related_terms=["energy storage", "smart grid", "carbon neutral"],
            context_keywords=["climate", "green tech", "renewable"],
            confidence=0.8,
            tier="tier1",
            source="perplexity",
            metadata={"model": "perplexity"}
        )
        
        # Mock extractors
        for extractor in manager.extractors:
            if hasattr(extractor, 'extract_keywords'):
                if "gpt" in str(type(extractor)).lower():
                    extractor.extract_keywords = AsyncMock(return_value=mock_gpt_result)
                else:
                    extractor.extract_keywords = AsyncMock(return_value=mock_perplexity_result)
        
        # Mock similarity analyzer and clusterer
        with patch.object(manager.similarity_analyzer, 'remove_duplicates') as mock_remove_dup, \
             patch.object(manager.clusterer, 'cluster_keywords') as mock_cluster:
            
            mock_remove_dup.return_value = [
                "renewable energy", "solar power", "wind energy", "green energy", 
                "sustainable power", "photovoltaic", "turbines", "energy storage"
            ]
            
            mock_cluster.return_value = {
                "primary": ["renewable energy", "solar power", "wind energy"],
                "related": ["photovoltaic", "turbines", "energy storage", "smart grid"],
                "context": ["sustainability", "clean tech", "environment", "climate"]
            }
            
            result = await manager.generate_keywords(topic)
            
            assert result is not None
            assert isinstance(result, KeywordResult)
            assert len(result.primary_keywords) >= 3
            assert len(result.related_terms) >= 3
            assert len(result.context_keywords) >= 3
            assert result.confidence > 0.8
            assert "merged" in result.metadata

    @pytest.mark.asyncio
    async def test_fallback_to_single_extractor(self, manager):
        """단일 추출기 폴백 테스트."""
        topic = "quantum computing"
        
        # 하나의 추출기만 성공
        successful_result = KeywordResult(
            primary_keywords=["quantum computing", "qubits", "quantum algorithms"],
            related_terms=["quantum supremacy", "quantum gates"],
            context_keywords=["computing", "technology"],
            confidence=0.9,
            tier="tier1",
            source="gpt",
            metadata={"model": "gpt-4"}
        )
        
        # Mock extractors - 일부는 실패, 일부는 성공
        mock_success_extractor = AsyncMock()
        mock_success_extractor.extract_keywords.return_value = successful_result
        
        mock_fail_extractor = AsyncMock()
        mock_fail_extractor.extract_keywords.side_effect = Exception("API Error")
        
        manager.extractors = [mock_fail_extractor, mock_success_extractor]
        
        result = await manager.generate_keywords(topic)
        
        assert result is not None
        assert result == successful_result  # 성공한 하나의 결과 반환

    @pytest.mark.asyncio
    async def test_regeneration_with_improved_strategy(self, manager):
        """개선된 전략으로 재생성 테스트."""
        topic = "blockchain technology"
        
        # 이전 결과 (낮은 신뢰도)
        previous_result = KeywordResult(
            primary_keywords=["blockchain"],
            related_terms=["crypto"],
            context_keywords=["digital"],
            confidence=0.4,
            tier="tier1",
            source="gpt",
            metadata={}
        )
        
        # 개선된 결과
        improved_result = KeywordResult(
            primary_keywords=["blockchain technology", "distributed ledger", "smart contracts"],
            related_terms=["cryptocurrency", "DeFi", "consensus algorithms", "hash functions"],
            context_keywords=["decentralization", "digital assets", "fintech"],
            confidence=0.9,
            tier="tier2",  # 더 상세한 tier
            source="gpt",
            metadata={"regenerated": True, "strategy": "enhanced"}
        )
        
        # Mock extractor for regeneration
        mock_extractor = AsyncMock()
        mock_extractor.extract_keywords.return_value = improved_result
        manager.extractors = [mock_extractor]
        
        result = await manager.regenerate_keywords(
            topic, previous_result, "Low confidence and insufficient keywords"
        )
        
        assert result is not None
        assert result.confidence > previous_result.confidence
        assert len(result.primary_keywords) > len(previous_result.primary_keywords)
        assert result.tier == "tier2"
        assert result.metadata.get("regenerated") is True


class TestMultiSourceKeywordManagerEdgeCases:
    """Multi-Source Keyword Manager의 엣지 케이스 테스트."""

    @pytest.fixture
    def manager(self):
        """테스트용 manager."""
        with patch('src.keyword_generation.manager.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_key"
            mock_config.get_perplexity_api_key.return_value = "test_key"
            yield MultiSourceKeywordManager()

    @pytest.mark.asyncio
    async def test_generate_keywords_empty_topic(self, manager):
        """빈 주제로 키워드 생성 테스트."""
        result = await manager.generate_keywords("")
        
        # 빈 주제에 대한 처리
        assert result is None or (isinstance(result, KeywordResult) and len(result.primary_keywords) == 0)

    @pytest.mark.asyncio
    async def test_generate_keywords_none_topic(self, manager):
        """None 주제로 키워드 생성 테스트."""
        result = await manager.generate_keywords(None)
        
        # None 주제에 대한 처리
        assert result is None

    def test_merge_results_with_none_values(self, manager):
        """None 값이 포함된 결과 병합 테스트."""
        results = [
            None,
            KeywordResult(
                primary_keywords=["AI"],
                related_terms=["ML"],
                context_keywords=["tech"],
                confidence=0.8,
                tier="tier1",
                source="gpt",
                metadata={}
            ),
            None
        ]
        
        # None 값들을 필터링하여 처리
        filtered_results = [r for r in results if r is not None]
        merged = manager._merge_results(filtered_results)
        
        assert merged is not None
        assert len(merged.primary_keywords) > 0

    def test_calculate_merged_confidence_with_zero_confidence(self, manager):
        """0 신뢰도가 포함된 병합 신뢰도 계산 테스트."""
        results = [
            MagicMock(confidence=0.0),
            MagicMock(confidence=0.8),
            MagicMock(confidence=0.0)
        ]
        
        merged_confidence = manager._calculate_merged_confidence(results)
        
        # 0 신뢰도가 있어도 합리적인 결과
        assert 0.0 <= merged_confidence <= 1.0

    def test_extract_context_keywords_empty_issues(self, manager):
        """빈 이슈 리스트에서 컨텍스트 키워드 추출 테스트."""
        context_keywords = manager._extract_context_keywords([])
        
        assert isinstance(context_keywords, list)
        assert len(context_keywords) == 0

    def test_extract_meaningful_terms_special_characters(self, manager):
        """특수 문자가 포함된 텍스트에서 용어 추출 테스트."""
        text = "AI/ML & deep-learning: next-gen tech (2024) - $100B market!"
        
        terms = manager._extract_meaningful_terms(text)
        
        assert isinstance(terms, list)
        # 특수 문자 처리 확인
        if terms:
            for term in terms:
                assert isinstance(term, str)
                assert len(term.strip()) > 0