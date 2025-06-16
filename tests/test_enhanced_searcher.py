"""
Enhanced Searcher 환각 탐지 시스템 테스트.
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

from src.hallucination_detection.enhanced_searcher import EnhancedIssueSearcher
from src.hallucination_detection.models import (
    CombinedHallucinationScore, LLMJudgeScore, ConsistencyScore, RePPLScore
)
from src.models import SearchResult, IssueItem, KeywordResult


class TestEnhancedIssueSearcher:
    """Enhanced Issue Searcher의 단위 테스트."""

    @pytest.fixture
    def searcher(self):
        """테스트용 EnhancedIssueSearcher 인스턴스."""
        with patch('src.hallucination_detection.enhanced_searcher.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_key"
            mock_config.get_perplexity_api_key.return_value = "test_key"
            yield EnhancedIssueSearcher()

    @pytest.fixture
    def sample_search_result(self):
        """테스트용 SearchResult 인스턴스."""
        issues = [
            IssueItem(
                title="AI가 모든 인간 직업을 대체할 것",
                summary="2025년까지 모든 직업이 AI로 대체된다",
                source="Tech Blog",
                published_date="2024-01-15",
                relevance_score=0.8,
                category="technology",
                content_snippet="AI는 혁신적이다"
            ),
            IssueItem(
                title="새로운 프로그래밍 언어 발표",
                summary="실제로 발표된 언어에 대한 정보",
                source="Developer News",
                published_date="2024-01-10",
                relevance_score=0.9,
                category="programming",
                content_snippet="새로운 언어가 발표되었다"
            )
        ]
        
        return SearchResult(
            query_keywords=["AI", "technology"],
            issues=issues,
            total_found=2,
            search_time=1.5,
            time_period="2024-01",
            api_calls_used=3,
            confidence_score=0.8,
            raw_responses=["response1", "response2"]
        )

    @pytest.fixture
    def sample_keyword_result(self):
        """테스트용 KeywordResult 인스턴스."""
        return KeywordResult(
            topic="test topic",
            primary_keywords=["AI", "technology"],
            related_terms=["machine learning", "automation"],
            context_keywords=["future", "jobs"],
            confidence_score=0.8,
            generation_time=1.0,
            raw_response="test response"
        )

    def test_initialization(self, searcher):
        """초기화 테스트."""
        assert searcher.threshold_manager is not None
        assert searcher.detectors is not None
        assert len(searcher.detectors) >= 2  # RePPL, Self-Consistency 최소

    def test_get_keyword_signature(self, searcher, sample_keyword_result):
        """키워드 서명 생성 테스트."""
        signature = searcher._get_keyword_signature(sample_keyword_result)
        
        # 모든 키워드가 정렬되어 포함되어야 함
        expected_keywords = ["ai", "automation", "future", "jobs", "machine learning", "technology"]
        assert signature == '|'.join(expected_keywords)

    def test_keyword_signature_deduplication(self, searcher):
        """키워드 중복 제거 테스트."""
        keyword_result = KeywordResult(
            topic="test topic",
            primary_keywords=["AI", "ai", "Technology"],
            related_terms=["AI", "tech"],
            context_keywords=["TECHNOLOGY"],
            confidence_score=0.8,
            generation_time=1.0,
            raw_response="test response"
        )
        
        signature = searcher._get_keyword_signature(keyword_result)
        
        # 중복이 제거되고 소문자로 정규화되어야 함
        expected = "ai|tech|technology"
        assert signature == expected

    @pytest.mark.asyncio
    async def test_regenerate_keywords_safely_max_reached(self, searcher, sample_keyword_result, sample_search_result):
        """최대 재생성 횟수 도달 테스트."""
        keyword_history = set()
        
        new_keywords, updated_count, should_continue = await searcher._regenerate_keywords_safely(
            current_keywords=sample_keyword_result,
            search_result=sample_search_result,
            failure_type="test_failure",
            keyword_history=keyword_history,
            regeneration_count=3,  # 최대값
            max_regenerations=3
        )
        
        assert new_keywords == sample_keyword_result  # 원본 반환
        assert updated_count == 3  # 증가되지 않음
        assert should_continue is False  # 중단

    @pytest.mark.asyncio
    async def test_regenerate_keywords_safely_duplicate_detected(self, searcher, sample_keyword_result, sample_search_result):
        """중복 키워드 감지 테스트."""
        signature = searcher._get_keyword_signature(sample_keyword_result)
        keyword_history = {signature}  # 이미 존재하는 서명
        
        with patch.object(searcher.keyword_manager, 'regenerate_keywords', AsyncMock(return_value=sample_keyword_result)):
            new_keywords, updated_count, should_continue = await searcher._regenerate_keywords_safely(
                current_keywords=sample_keyword_result,
                search_result=sample_search_result,
                failure_type="test_failure",
                keyword_history=keyword_history,
                regeneration_count=1,
                max_regenerations=3
            )
        
        assert new_keywords == sample_keyword_result  # 원본 반환
        assert updated_count == 2  # 증가됨
        assert should_continue is False  # 중복으로 인한 중단

    @pytest.mark.asyncio
    async def test_run_reppl_detector(self, searcher):
        """RePPL 탐지기 실행 테스트."""
        text = "AI는 2024년에 모든 직업을 대체할 것입니다."
        
        with patch.object(searcher.detectors['RePPL'], 'analyze_text', AsyncMock()) as mock_analyze:
            mock_analyze.return_value = RePPLScore(
                confidence=0.7,
                repetition_score=0.8,
                perplexity_score=0.6,
                entropy_score=0.7,
                analysis_details={}
            )
            
            result = await searcher._run_reppl_detector(text)
            
            assert result['status'] == 'success'
            assert result['confidence'] == 0.7
            assert isinstance(result, RePPLScore)

    @pytest.mark.asyncio
    async def test_run_reppl_detector_failure(self, searcher):
        """RePPL 탐지기 실패 처리 테스트."""
        text = "테스트 텍스트"
        
        with patch.object(searcher.detectors['RePPL'], 'analyze_text', AsyncMock(side_effect=Exception("API Error"))):
            result = await searcher._run_reppl_detector(text)
            
            assert result['status'] == 'error'
            assert result['confidence'] == 0.0

    @pytest.mark.asyncio
    async def test_run_consistency_detector_short_text(self, searcher):
        """짧은 텍스트 일관성 검사 최적화 테스트."""
        short_text = "AI는 좋다."  # 100자 미만
        
        result = await searcher._run_consistency_detector(short_text)
        
        # 최적화된 일관성 점수 반환
        assert isinstance(result, ConsistencyScore)
        assert result.confidence == 0.6  # 수정된 값
        assert result.analysis_details["optimized"] is True

    @pytest.mark.asyncio
    async def test_run_consistency_detector_long_text(self, searcher):
        """긴 텍스트 일관성 검사 테스트."""
        long_text = "AI 기술은 빠르게 발전하고 있으며, 많은 산업에 영향을 미치고 있습니다. " * 5
        
        with patch.object(searcher.detectors['Self-Consistency'], 'analyze_text', AsyncMock()) as mock_analyze:
            mock_analyze.return_value = ConsistencyScore(
                confidence=0.8,
                consistency_rate=0.85,
                num_queries=3,
                num_consistent=2,
                variations=[],
                common_elements=[],
                divergent_elements=[],
                analysis_details={}
            )
            
            result = await searcher._run_consistency_detector(long_text)
            
            assert isinstance(result, ConsistencyScore)
            assert result.confidence == 0.8

    @pytest.mark.asyncio
    async def test_finalize_issue_validation_no_scores(self, searcher):
        """점수 없는 이슈 검증 테스트."""
        issue = IssueItem(
            title="테스트 이슈",
            summary="테스트 요약",
            source="Test Source",
            published_date="2024-01-01",
            relevance_score=0.8,
            category="test",
            content_snippet="테스트"
        )
        
        result = searcher._finalize_issue_validation(issue, {})
        
        assert result is None  # 점수가 없으면 None 반환

    @pytest.mark.asyncio
    async def test_finalize_issue_validation_success(self, searcher):
        """성공적인 이슈 검증 테스트."""
        issue = IssueItem(
            title="테스트 이슈",
            summary="테스트 요약",
            source="Test Source",
            published_date="2024-01-01",
            relevance_score=0.8,
            category="test",
            content_snippet="테스트"
        )
        
        scores = {
            'RePPL': RePPLScore(confidence=0.7, repetition_score=0.8, perplexity_score=0.6, entropy_score=0.7),
            'Self-Consistency': ConsistencyScore(
                confidence=0.8, consistency_rate=0.8, num_queries=3, num_consistent=2,
                variations=[], common_elements=[], divergent_elements=[], analysis_details={}
            )
        }
        
        result = searcher._finalize_issue_validation(issue, scores)
        
        assert result is not None
        assert hasattr(result, 'hallucination_confidence')
        assert hasattr(result, 'hallucination_analysis')
        assert result.hallucination_confidence > 0.6  # 높은 신뢰도

    @pytest.mark.asyncio
    async def test_finalize_issue_validation_low_confidence(self, searcher):
        """낮은 신뢰도 이슈 검증 테스트."""
        issue = IssueItem(
            title="의심스러운 이슈",
            summary="확인되지 않은 주장",
            source="Unknown Source",
            published_date="2024-01-01",
            relevance_score=0.3,
            category="suspicious",
            content_snippet="의심스러운 내용"
        )
        
        scores = {
            'RePPL': RePPLScore(confidence=0.3, repetition_score=0.4, perplexity_score=0.2, entropy_score=0.3),
            'Self-Consistency': ConsistencyScore(
                confidence=0.2, consistency_rate=0.2, num_queries=3, num_consistent=0,
                variations=["inconsistent"], common_elements=[], divergent_elements=["different"], analysis_details={}
            )
        }
        
        result = searcher._finalize_issue_validation(issue, scores)
        
        # 낮은 신뢰도로 인해 필터링될 수 있음
        if result is not None:
            assert result.hallucination_confidence < 0.5
        # 또는 None (임계값 미달로 필터링)

    def test_create_optimized_consistency_score(self, searcher):
        """최적화된 일관성 점수 생성 테스트."""
        score = searcher._create_optimized_consistency_score()
        
        assert isinstance(score, ConsistencyScore)
        assert score.confidence == 0.6  # 수정된 값
        assert score.consistency_rate == 0.6
        assert score.num_queries == 1
        assert score.num_consistent == 1
        assert score.analysis_details["optimized"] is True
        assert score.analysis_details["reason"] == "짧은 텍스트 최적화 처리"


class TestEnhancedSearcherIntegration:
    """Enhanced Searcher의 통합 테스트."""

    @pytest.fixture
    def searcher(self):
        """테스트용 searcher."""
        with patch('src.hallucination_detection.enhanced_searcher.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_key"
            mock_config.get_perplexity_api_key.return_value = "test_key"
            yield EnhancedIssueSearcher()

    @pytest.mark.asyncio
    async def test_search_with_validation_high_confidence(self, searcher):
        """높은 신뢰도 검색 및 검증 테스트."""
        with patch.object(searcher, '_run_reppl_detector', AsyncMock()) as mock_reppl, \
             patch.object(searcher, '_run_consistency_detector', AsyncMock()) as mock_consistency, \
             patch.object(searcher.keyword_manager, 'generate_keywords', AsyncMock()) as mock_keywords, \
             patch.object(searcher.issue_searcher, 'search_issues', AsyncMock()) as mock_search:
            
            # Mock 응답 설정
            mock_keywords.return_value = KeywordResult(
                topic="AI technology trends",
                primary_keywords=["AI", "technology"],
                related_terms=["machine learning"],
                context_keywords=["future"],
                confidence_score=0.9,
                generation_time=1.0,
                raw_response="test response"
            )
            
            mock_issues = [
                IssueItem(
                    title="신뢰할 수 있는 AI 뉴스",
                    summary="검증된 정보",
                    source="Reliable Source",
                    published_date="2024-01-15",
                    relevance_score=0.9,
                    category="technology",
                    content_snippet="신뢰할 수 있는 내용"
                )
            ]
            
            mock_search.return_value = SearchResult(
                query_keywords=["AI", "technology"],
                issues=mock_issues,
                total_found=1,
                search_time=1.0,
                time_period="2024-01",
                api_calls_used=2,
                confidence_score=0.9,
                raw_responses=["response1"]
            )
            
            mock_reppl.return_value = RePPLScore(
                confidence=0.8, repetition_score=0.8, perplexity_score=0.8, entropy_score=0.8
            )
            
            mock_consistency.return_value = ConsistencyScore(
                confidence=0.9, consistency_rate=0.9, num_queries=3, num_consistent=3,
                variations=[], common_elements=[], divergent_elements=[], analysis_details={}
            )
            
            result = await searcher.search_with_validation("AI technology trends", max_regenerations=2)
            
            assert result is not None
            assert result.total_found >= 1
            # 높은 신뢰도 이슈가 포함되어야 함
            if result.issues:
                assert hasattr(result.issues[0], 'hallucination_confidence')

    @pytest.mark.asyncio
    async def test_search_with_validation_keyword_regeneration(self, searcher):
        """키워드 재생성이 필요한 경우 테스트."""
        regeneration_count = 0
        
        async def mock_search_side_effect(*args, **kwargs):
            nonlocal regeneration_count
            regeneration_count += 1
            
            if regeneration_count == 1:
                # 첫 번째 검색: 낮은 신뢰도 결과
                return SearchResult(
                    query_keywords=["AI"],
                    issues=[
                        IssueItem(
                            title="의심스러운 주장",
                            summary="검증되지 않은 정보",
                            source="Unknown",
                            published_date="2024-01-01",
                            relevance_score=0.3,
                            category="suspicious",
                            content_snippet="의심스러운 내용"
                        )
                    ],
                    total_found=1,
                    search_time=1.0,
                    time_period="2024-01",
                    api_calls_used=1,
                    confidence_score=0.3,
                    raw_responses=["low_confidence_response"]
                )
            else:
                # 재생성 후: 더 나은 결과
                return SearchResult(
                    query_keywords=["AI", "technology", "trends"],
                    issues=[
                        IssueItem(
                            title="신뢰할 수 있는 정보",
                            summary="검증된 내용",
                            source="Reliable Source",
                            published_date="2024-01-15",
                            relevance_score=0.9,
                            category="technology",
                            content_snippet="신뢰할 수 있는 내용"
                        )
                    ],
                    total_found=1,
                    search_time=1.2,
                    time_period="2024-01",
                    api_calls_used=2,
                    confidence_score=0.9,
                    raw_responses=["reliable_response"]
                )
        
        with patch.object(searcher.keyword_manager, 'generate_keywords', AsyncMock()) as mock_keywords, \
             patch.object(searcher.keyword_manager, 'regenerate_keywords', AsyncMock()) as mock_regen, \
             patch.object(searcher.issue_searcher, 'search_issues', AsyncMock(side_effect=mock_search_side_effect)) as mock_search, \
             patch.object(searcher, '_run_reppl_detector', AsyncMock()) as mock_reppl, \
             patch.object(searcher, '_run_consistency_detector', AsyncMock()) as mock_consistency:
            
            # 초기 키워드
            mock_keywords.return_value = KeywordResult(
                topic="AI",
                primary_keywords=["AI"],
                related_terms=[],
                context_keywords=[],
                confidence_score=0.5,
                generation_time=1.0,
                raw_response="test response"
            )
            
            # 재생성된 키워드
            mock_regen.return_value = KeywordResult(
                topic="AI technology trends",
                primary_keywords=["AI", "technology", "trends"],
                related_terms=["machine learning"],
                context_keywords=["future", "innovation"],
                confidence_score=0.9,
                generation_time=1.0,
                raw_response="test response"
            )
            
            # 낮은 신뢰도로 시작, 높은 신뢰도로 개선
            mock_reppl.side_effect = [
                RePPLScore(confidence=0.3, repetition_score=0.3, perplexity_score=0.3, entropy_score=0.3),
                RePPLScore(confidence=0.8, repetition_score=0.8, perplexity_score=0.8, entropy_score=0.8)
            ]
            
            mock_consistency.side_effect = [
                ConsistencyScore(confidence=0.2, consistency_rate=0.2, num_queries=3, num_consistent=0, 
                               variations=[], common_elements=[], divergent_elements=[], analysis_details={}),
                ConsistencyScore(confidence=0.9, consistency_rate=0.9, num_queries=3, num_consistent=3,
                               variations=[], common_elements=[], divergent_elements=[], analysis_details={})
            ]
            
            result = await searcher.search_with_validation("AI", max_regenerations=2)
            
            assert result is not None
            assert regeneration_count == 2  # 재생성이 발생했는지 확인
            # 재생성 키워드가 호출되었는지 확인
            assert mock_regen.called


class TestEnhancedSearcherEdgeCases:
    """Enhanced Searcher의 엣지 케이스 테스트."""

    @pytest.fixture
    def searcher(self):
        """테스트용 searcher."""
        with patch('src.hallucination_detection.enhanced_searcher.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_key"
            mock_config.get_perplexity_api_key.return_value = "test_key"
            yield EnhancedIssueSearcher()

    @pytest.mark.asyncio
    async def test_search_with_no_issues_found(self, searcher):
        """이슈가 발견되지 않은 경우 테스트."""
        with patch.object(searcher.keyword_manager, 'generate_keywords', AsyncMock()) as mock_keywords, \
             patch.object(searcher.issue_searcher, 'search_issues', AsyncMock()) as mock_search:
            
            mock_keywords.return_value = KeywordResult(
                topic="nonexistent topic",
                primary_keywords=["nonexistent"],
                related_terms=[],
                context_keywords=[],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test response"
            )
            
            mock_search.return_value = SearchResult(
                query_keywords=["nonexistent"],
                issues=[],
                total_found=0,
                search_time=0.5,
                time_period="2024-01",
                api_calls_used=1,
                confidence_score=0.0,
                raw_responses=[]
            )
            
            result = await searcher.search_with_validation("nonexistent topic")
            
            assert result is not None
            assert result.total_found == 0
            assert len(result.issues) == 0

    @pytest.mark.asyncio
    async def test_search_with_detector_failures(self, searcher):
        """모든 탐지기가 실패하는 경우 테스트."""
        with patch.object(searcher.keyword_manager, 'generate_keywords', AsyncMock()) as mock_keywords, \
             patch.object(searcher.issue_searcher, 'search_issues', AsyncMock()) as mock_search, \
             patch.object(searcher, '_run_reppl_detector', AsyncMock(side_effect=Exception("RePPL Failed"))), \
             patch.object(searcher, '_run_consistency_detector', AsyncMock(side_effect=Exception("Consistency Failed"))):
            
            mock_keywords.return_value = KeywordResult(
                topic="test topic",
                primary_keywords=["test"],
                related_terms=[],
                context_keywords=[],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test response"
            )
            
            mock_search.return_value = SearchResult(
                query_keywords=["test"],
                issues=[
                    IssueItem(
                        title="테스트 이슈",
                        summary="테스트 요약",
                        source="Test Source",
                        published_date="2024-01-01",
                        relevance_score=0.8,
                        category="test",
                        content_snippet="테스트"
                    )
                ],
                total_found=1,
                search_time=1.0,
                time_period="2024-01",
                api_calls_used=1,
                confidence_score=0.8,
                raw_responses=["test_response"]
            )
            
            result = await searcher.search_with_validation("test topic")
            
            # 탐지기 실패로 인해 이슈가 필터링될 수 있음
            assert result is not None
            # 결과의 이슈 수는 0일 수 있음 (모든 탐지기 실패로 인한 필터링)

    def test_empty_keyword_signature(self, searcher):
        """빈 키워드로 서명 생성 테스트."""
        empty_result = KeywordResult(
            topic="empty topic",
            primary_keywords=[],
            related_terms=[],
            context_keywords=[],
            confidence_score=0.0,
            generation_time=1.0,
            raw_response="test response"
        )
        
        signature = searcher._get_keyword_signature(empty_result)
        assert signature == ""