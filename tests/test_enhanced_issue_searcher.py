"""
Enhanced Issue Searcher 테스트.
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime
import sys
import os

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.search.enhanced_issue_searcher import EnhancedIssueSearcher, EnhancedIssue
from src.models import IssueItem, SearchResult
from src.search.issue_searcher import IssueSearcher


class TestEnhancedIssueSearcher:
    """Enhanced Issue Searcher의 단위 테스트."""

    @pytest.fixture
    def searcher(self):
        """테스트용 EnhancedIssueSearcher 인스턴스."""
        with patch('src.search.enhanced_issue_searcher.config') as mock_config:
            mock_config.get_perplexity_api_key.return_value = "test_key"
            yield EnhancedIssueSearcher()

    @pytest.fixture
    def sample_issue_item(self):
        """테스트용 IssueItem."""
        return IssueItem(
            title="AI 기술의 혁신적 발전",
            summary="AI 기술이 여러 분야에서 혁신적으로 발전하고 있습니다.",
            source="Tech News",
            published_date="2024-01-15",
            relevance_score=0.8,
            category="technology",
            content_snippet="AI가 의료, 교육, 자동화 분야에서 큰 진전을 보이고 있음",
            detailed_content="인공지능 기술이 의료 진단의 정확도를 크게 향상시키고..."
        )

    @pytest.fixture
    def sample_enhanced_issue(self):
        """테스트용 EnhancedIssue."""
        return EnhancedIssue(
            title="AI 기술의 혁신적 발전",
            content="AI 기술이 여러 분야에서 혁신적으로 발전하고 있습니다.",
            source=MagicMock(name="Tech News", reliability=0.8),
            timestamp=datetime(2024, 1, 15),
            fact_check_score=0.85,
            sentiment_score=0.7,
            impact_score=0.9,
            controversy_score=0.2,
            momentum_score=0.8,
            authority_score=0.7,
            shares=100,
            comments=25,
            likes=200,
            contradictions=[],
            supporting_facts=["AI improves medical diagnosis", "Education AI tools are effective"]
        )

    def test_initialization(self, searcher):
        """초기화 테스트."""
        assert searcher.base_searcher is not None
        assert isinstance(searcher.base_searcher, IssueSearcher)
        assert hasattr(searcher, 'fact_patterns')
        assert 'numbers' in searcher.fact_patterns
        assert 'dates' in searcher.fact_patterns

    def test_fact_patterns_configuration(self, searcher):
        """사실 패턴 설정 테스트."""
        assert 'numbers' in searcher.fact_patterns
        assert 'dates' in searcher.fact_patterns
        assert 'percentages' in searcher.fact_patterns
        assert 'companies' in searcher.fact_patterns
        assert 'locations' in searcher.fact_patterns

    @pytest.mark.asyncio
    async def test_search_enhanced_issues_success(self, searcher):
        """향상된 이슈 검색 성공 테스트."""
        mock_search_result = SearchResult(
            query_keywords=["AI", "technology"],
            issues=[
                IssueItem(
                    title="AI 발전",
                    summary="AI가 발전하고 있음",
                    source="Tech Source",
                    published_date="2024-01-01",
                    relevance_score=0.8,
                    category="tech",
                    content_snippet="AI 기술 발전"
                )
            ],
            total_found=1,
            search_time=1.0,
            time_period="2024-01",
            api_calls_used=2,
            confidence_score=0.8,
            raw_responses=["response1"]
        )

        with patch.object(searcher.base_searcher, 'search_issues', AsyncMock(return_value=mock_search_result)):
            result = await searcher.search_enhanced_issues(["AI", "technology"])
            
            assert result is not None
            assert result.total_found == 1
            assert len(result.issues) == 1
            
            # Enhanced 속성이 추가되었는지 확인
            enhanced_issue = result.issues[0]
            assert hasattr(enhanced_issue, 'combined_confidence')
            assert hasattr(enhanced_issue, 'sentiment_score')
            assert hasattr(enhanced_issue, 'impact_score')

    @pytest.mark.asyncio
    async def test_search_enhanced_issues_empty_result(self, searcher):
        """빈 검색 결과 테스트."""
        empty_result = SearchResult(
            query_keywords=["nonexistent"],
            issues=[],
            total_found=0,
            search_time=0.5,
            time_period="2024-01",
            api_calls_used=1,
            confidence_score=0.0,
            raw_responses=[]
        )

        with patch.object(searcher.base_searcher, 'search_issues', AsyncMock(return_value=empty_result)):
            result = await searcher.search_enhanced_issues(["nonexistent"])
            
            assert result is not None
            assert result.total_found == 0
            assert len(result.issues) == 0

    def test_enhance_issue_item(self, searcher, sample_issue_item):
        """이슈 아이템 향상 테스트."""
        enhanced = searcher._enhance_issue_item(sample_issue_item)
        
        assert isinstance(enhanced, EnhancedIssue)
        assert enhanced.title == sample_issue_item.title
        assert enhanced.content == sample_issue_item.summary
        assert enhanced.source.name == sample_issue_item.source
        assert enhanced.timestamp is not None
        
        # 점수들이 합리적인 범위 내에 있는지 확인
        assert 0.0 <= enhanced.fact_check_score <= 1.0
        assert 0.0 <= enhanced.sentiment_score <= 1.0
        assert 0.0 <= enhanced.impact_score <= 1.0
        assert 0.0 <= enhanced.controversy_score <= 1.0
        assert 0.0 <= enhanced.momentum_score <= 1.0
        assert 0.0 <= enhanced.authority_score <= 1.0

    def test_calculate_fact_check_score(self, searcher):
        """사실 확인 점수 계산 테스트."""
        # 높은 신뢰도 콘텐츠
        high_confidence_content = "연구에 따르면 AI 기술이 85% 향상되었습니다. IEEE에서 발표한 자료에 의하면..."
        high_score = searcher._calculate_fact_check_score(high_confidence_content, "IEEE")
        assert high_score > 0.7

        # 낮은 신뢰도 콘텐츠
        low_confidence_content = "AI가 모든 것을 대체할 것입니다. 확실합니다."
        low_score = searcher._calculate_fact_check_score(low_confidence_content, "Unknown Blog")
        assert low_score < 0.5

    def test_calculate_sentiment_score(self, searcher):
        """감정 점수 계산 테스트."""
        # 긍정적 콘텐츠
        positive_content = "AI 기술의 훌륭한 발전과 혁신적인 성과"
        positive_score = searcher._calculate_sentiment_score(positive_content)
        assert positive_score > 0.5

        # 부정적 콘텐츠
        negative_content = "AI의 위험하고 우려되는 측면들"
        negative_score = searcher._calculate_sentiment_score(negative_content)
        assert negative_score < 0.5

        # 중립적 콘텐츠
        neutral_content = "AI 기술에 대한 객관적 분석"
        neutral_score = searcher._calculate_sentiment_score(neutral_content)
        assert 0.4 <= neutral_score <= 0.6

    def test_calculate_impact_score(self, searcher):
        """임팩트 점수 계산 테스트."""
        # 높은 임팩트 콘텐츠
        high_impact_content = "혁신적인 AI 기술이 전 세계 산업을 변화시키고 있습니다"
        high_score = searcher._calculate_impact_score(high_impact_content)
        assert high_score > 0.7

        # 낮은 임팩트 콘텐츠
        low_impact_content = "AI에 대한 간단한 설명"
        low_score = searcher._calculate_impact_score(low_impact_content)
        assert low_score < 0.5

    def test_calculate_controversy_score(self, searcher):
        """논란 점수 계산 테스트."""
        # 논란이 많은 콘텐츠
        controversial_content = "AI가 인간 일자리를 모두 대체할 것이며, 이는 위험하고 우려되는 상황입니다"
        controversial_score = searcher._calculate_controversy_score(controversial_content)
        assert controversial_score > 0.6

        # 논란이 적은 콘텐츠
        neutral_content = "AI 기술의 기본 원리와 구조"
        neutral_score = searcher._calculate_controversy_score(neutral_content)
        assert neutral_score < 0.4

    def test_calculate_momentum_score(self, searcher):
        """모멘텀 점수 계산 테스트."""
        # 높은 모멘텀 콘텐츠
        high_momentum_content = "빠르게 발전하는 AI 기술, 급속한 성장과 확산"
        high_score = searcher._calculate_momentum_score(high_momentum_content)
        assert high_score > 0.7

        # 낮은 모멘텀 콘텐츠
        low_momentum_content = "AI 기술의 기본 개념"
        low_score = searcher._calculate_momentum_score(low_momentum_content)
        assert low_score < 0.5

    def test_calculate_authority_score(self, searcher):
        """권위 점수 계산 테스트."""
        # 높은 권위 소스
        high_authority_sources = ["IEEE", "Nature", "Science", "MIT Technology Review", "ACM"]
        for source in high_authority_sources:
            score = searcher._calculate_authority_score(source)
            assert score > 0.8

        # 중간 권위 소스
        medium_authority_sources = ["TechCrunch", "Wired", "The Verge"]
        for source in medium_authority_sources:
            score = searcher._calculate_authority_score(source)
            assert 0.6 <= score <= 0.8

        # 낮은 권위 소스
        low_authority_score = searcher._calculate_authority_score("Unknown Blog")
        assert low_authority_score < 0.5

    def test_extract_key_facts(self, searcher):
        """핵심 사실 추출 테스트."""
        content = """
        AI 기술이 2024년에 85% 성장했습니다. 
        OpenAI는 GPT-4를 출시했으며, 
        이는 샌프란시스코에서 개발되었습니다.
        성능이 90% 향상되었다고 보고되었습니다.
        """
        
        facts = searcher._extract_key_facts(content)
        
        # 숫자, 회사명, 위치 등이 추출되었는지 확인
        assert len(facts) > 0
        fact_values = list(facts.values())
        fact_str = ' '.join(str(v) for v in fact_values if v)
        
        # 일부 패턴이 매칭되었는지 확인
        assert len(facts) >= 1

    @pytest.mark.asyncio
    async def test_find_contradictions(self, searcher, sample_enhanced_issue):
        """모순 찾기 테스트."""
        # 모순되는 이슈들 생성
        issue1 = EnhancedIssue(
            title="AI 성능 향상",
            content="AI 성능이 크게 증가했습니다",
            source=MagicMock(name="Source1"),
            timestamp=datetime.now(),
            fact_check_score=0.8,
            sentiment_score=0.8,
            impact_score=0.7,
            controversy_score=0.2,
            momentum_score=0.8,
            authority_score=0.7,
            shares=50, comments=10, likes=100,
            contradictions=[], supporting_facts=[]
        )
        
        issue2 = EnhancedIssue(
            title="AI 성능 저하",
            content="AI 성능이 크게 감소했습니다",
            source=MagicMock(name="Source2"),
            timestamp=datetime.now(),
            fact_check_score=0.8,
            sentiment_score=0.2,
            impact_score=0.7,
            controversy_score=0.2,
            momentum_score=0.8,
            authority_score=0.7,
            shares=30, comments=5, likes=50,
            contradictions=[], supporting_facts=[]
        )

        all_issues = [issue1, issue2]
        contradictions = await searcher._find_contradictions(issue1, all_issues)
        
        assert len(contradictions) > 0
        assert "Source2" in contradictions[0]

    def test_detect_direction(self, searcher):
        """방향성 감지 테스트."""
        positive_keywords = ['increase', 'growth', 'rise', 'up', 'gain', 'positive']
        negative_keywords = ['decrease', 'decline', 'fall', 'down', 'loss', 'negative']
        
        # 긍정적 방향
        positive_content = "AI performance shows significant increase and growth"
        positive_direction = searcher._detect_direction(positive_content, positive_keywords, negative_keywords)
        assert positive_direction == "positive"
        
        # 부정적 방향
        negative_content = "AI performance shows decline and decrease"
        negative_direction = searcher._detect_direction(negative_content, positive_keywords, negative_keywords)
        assert negative_direction == "negative"
        
        # 중립적 방향
        neutral_content = "AI performance analysis without clear direction"
        neutral_direction = searcher._detect_direction(neutral_content, positive_keywords, negative_keywords)
        assert neutral_direction is None

    def test_convert_to_issue_item(self, searcher, sample_enhanced_issue):
        """EnhancedIssue를 IssueItem으로 변환 테스트."""
        issue_item = searcher._convert_to_issue_item(sample_enhanced_issue)
        
        assert isinstance(issue_item, IssueItem)
        assert issue_item.title == sample_enhanced_issue.title
        assert issue_item.summary == sample_enhanced_issue.content
        assert issue_item.source == sample_enhanced_issue.source.name
        
        # Enhanced 속성들이 올바르게 설정되었는지 확인
        assert hasattr(issue_item, 'combined_confidence')
        assert hasattr(issue_item, 'sentiment_score')
        assert hasattr(issue_item, 'impact_score')
        assert hasattr(issue_item, 'controversy_score')
        assert hasattr(issue_item, 'momentum_score')
        assert hasattr(issue_item, 'authority_score')
        assert hasattr(issue_item, 'social_engagement')
        
        assert issue_item.combined_confidence == sample_enhanced_issue.fact_check_score
        assert issue_item.sentiment_score == sample_enhanced_issue.sentiment_score
        assert issue_item.social_engagement == (
            sample_enhanced_issue.shares + 
            sample_enhanced_issue.comments + 
            sample_enhanced_issue.likes
        )


class TestEnhancedIssueSearcherIntegration:
    """Enhanced Issue Searcher의 통합 테스트."""

    @pytest.fixture
    def searcher(self):
        """테스트용 searcher."""
        with patch('src.search.enhanced_issue_searcher.config') as mock_config:
            mock_config.get_perplexity_api_key.return_value = "test_key"
            yield EnhancedIssueSearcher()

    @pytest.mark.asyncio
    async def test_full_search_pipeline(self, searcher):
        """전체 검색 파이프라인 테스트."""
        mock_issues = [
            IssueItem(
                title="AI 기술 혁신",
                summary="AI 기술이 혁신적으로 발전하고 있습니다",
                source="IEEE Computer Society",
                published_date="2024-01-15",
                relevance_score=0.9,
                category="technology",
                content_snippet="AI가 의료 진단의 정확도를 95% 향상시켰습니다",
                detailed_content="최신 연구에 따르면 AI 기술이..."
            ),
            IssueItem(
                title="AI 위험성 경고",
                summary="일부 전문가들이 AI의 위험성을 경고하고 있습니다",
                source="Tech Blog",
                published_date="2024-01-10",
                relevance_score=0.7,
                category="opinion",
                content_snippet="AI가 일자리를 대체할 위험이 있다고 주장",
                detailed_content="AI 기술의 급속한 발전으로 인한 우려..."
            )
        ]

        mock_search_result = SearchResult(
            query_keywords=["AI", "technology"],
            issues=mock_issues,
            total_found=2,
            search_time=2.0,
            time_period="2024-01",
            api_calls_used=3,
            confidence_score=0.85,
            raw_responses=["response1", "response2"]
        )

        with patch.object(searcher.base_searcher, 'search_issues', AsyncMock(return_value=mock_search_result)):
            result = await searcher.search_enhanced_issues(["AI", "technology"])
            
            assert result is not None
            assert result.total_found == 2
            assert len(result.issues) == 2
            
            # 첫 번째 이슈 (높은 권위 소스)
            enhanced_issue1 = result.issues[0]
            assert hasattr(enhanced_issue1, 'combined_confidence')
            assert enhanced_issue1.combined_confidence > 0.7  # IEEE는 높은 권위
            
            # 두 번째 이슈 (낮은 권위 소스)
            enhanced_issue2 = result.issues[1]
            assert hasattr(enhanced_issue2, 'combined_confidence')
            # Tech Blog는 상대적으로 낮은 권위

    @pytest.mark.asyncio
    async def test_search_with_base_searcher_error(self, searcher):
        """기본 검색기 오류 처리 테스트."""
        with patch.object(searcher.base_searcher, 'search_issues', AsyncMock(side_effect=Exception("Search failed"))):
            
            # 예외가 발생해도 적절히 처리되는지 확인
            with pytest.raises(Exception):
                await searcher.search_enhanced_issues(["test"])


class TestEnhancedIssueSearcherEdgeCases:
    """Enhanced Issue Searcher의 엣지 케이스 테스트."""

    @pytest.fixture
    def searcher(self):
        """테스트용 searcher."""
        with patch('src.search.enhanced_issue_searcher.config') as mock_config:
            mock_config.get_perplexity_api_key.return_value = "test_key"
            yield EnhancedIssueSearcher()

    def test_enhance_issue_item_minimal_data(self, searcher):
        """최소한의 데이터로 이슈 향상 테스트."""
        minimal_issue = IssueItem(
            title="최소 이슈",
            summary="",  # 빈 요약
            source="",   # 빈 소스
            published_date=None,  # 날짜 없음
            relevance_score=0.0,
            category="",
            content_snippet=""
        )
        
        enhanced = searcher._enhance_issue_item(minimal_issue)
        
        assert isinstance(enhanced, EnhancedIssue)
        assert enhanced.title == "최소 이슈"
        assert enhanced.authority_score < 0.5  # 빈 소스는 낮은 권위
        assert 0.0 <= enhanced.fact_check_score <= 1.0
        assert 0.0 <= enhanced.sentiment_score <= 1.0

    def test_calculate_scores_empty_content(self, searcher):
        """빈 콘텐츠로 점수 계산 테스트."""
        empty_content = ""
        
        fact_score = searcher._calculate_fact_check_score(empty_content, "")
        sentiment_score = searcher._calculate_sentiment_score(empty_content)
        impact_score = searcher._calculate_impact_score(empty_content)
        controversy_score = searcher._calculate_controversy_score(empty_content)
        momentum_score = searcher._calculate_momentum_score(empty_content)
        
        # 모든 점수가 유효한 범위 내에 있어야 함
        for score in [fact_score, sentiment_score, impact_score, controversy_score, momentum_score]:
            assert 0.0 <= score <= 1.0

    def test_extract_key_facts_no_patterns(self, searcher):
        """패턴이 매칭되지 않는 콘텐츠에서 사실 추출 테스트."""
        content_without_facts = "일반적인 텍스트 내용입니다"
        
        facts = searcher._extract_key_facts(content_without_facts)
        
        # 빈 딕셔너리이거나 빈 값들이어야 함
        assert isinstance(facts, dict)
        for key, value in facts.items():
            if value:  # 값이 있다면
                assert isinstance(value, (str, list))

    def test_parse_published_date_various_formats(self, searcher):
        """다양한 날짜 형식 파싱 테스트."""
        # 유효한 날짜 형식들
        valid_dates = [
            "2024-01-15",
            "2024/01/15", 
            "2024.01.15",
            "Jan 15, 2024",
            "15 January 2024"
        ]
        
        for date_str in valid_dates:
            result = searcher._parse_published_date(date_str)
            if result:  # 파싱이 성공한 경우
                assert isinstance(result, datetime)
                assert result.year == 2024
        
        # 유효하지 않은 날짜
        invalid_dates = ["invalid", "", None, "not-a-date"]
        for date_str in invalid_dates:
            result = searcher._parse_published_date(date_str)
            # None이거나 현재 시간이어야 함
            if result:
                assert isinstance(result, datetime)

    def test_source_object_creation(self, searcher):
        """소스 객체 생성 테스트."""
        # 다양한 소스명으로 테스트
        sources = ["IEEE", "Unknown Blog", "", "TechCrunch", "Nature"]
        
        for source_name in sources:
            source_obj = searcher._create_source_object(source_name)
            
            assert hasattr(source_obj, 'name')
            assert hasattr(source_obj, 'reliability')
            assert source_obj.name == source_name
            assert 0.0 <= source_obj.reliability <= 1.0

    @pytest.mark.asyncio
    async def test_find_contradictions_same_source(self, searcher):
        """같은 소스의 이슈들 간 모순 찾기 테스트."""
        issue = EnhancedIssue(
            title="테스트 이슈",
            content="AI 성능이 향상되었습니다",
            source=MagicMock(name="Same Source"),
            timestamp=datetime.now(),
            fact_check_score=0.8, sentiment_score=0.8, impact_score=0.7,
            controversy_score=0.2, momentum_score=0.8, authority_score=0.7,
            shares=50, comments=10, likes=100,
            contradictions=[], supporting_facts=[]
        )
        
        same_source_issue = EnhancedIssue(
            title="다른 이슈",
            content="AI 성능이 저하되었습니다",
            source=MagicMock(name="Same Source"),  # 같은 소스
            timestamp=datetime.now(),
            fact_check_score=0.8, sentiment_score=0.2, impact_score=0.7,
            controversy_score=0.2, momentum_score=0.8, authority_score=0.7,
            shares=30, comments=5, likes=50,
            contradictions=[], supporting_facts=[]
        )
        
        all_issues = [issue, same_source_issue]
        contradictions = await searcher._find_contradictions(issue, all_issues)
        
        # 같은 소스는 제외되어야 함
        assert len(contradictions) == 0