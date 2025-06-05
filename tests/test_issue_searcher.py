"""
이슈 검색기 pytest 테스트 - 완전 수정된 버전
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json

# 프로젝트 루트 디렉토리의 src 폴더를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)


class TestPerplexityClient:
    """Perplexity API 클라이언트 테스트 클래스"""

    @pytest.mark.unit
    def test_perplexity_client_import(self):
        """Perplexity 클라이언트 import 테스트"""
        from src.issue_searcher import PerplexityClient, IssueItem, SearchResult
        assert PerplexityClient is not None
        assert IssueItem is not None
        assert SearchResult is not None

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_perplexity_key'})
    def test_perplexity_client_initialization(self):
        """Perplexity 클라이언트 초기화 테스트"""
        from src.issue_searcher import PerplexityClient

        client = PerplexityClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.model == "llama-3.1-sonar-small-128k-online"
        assert client.base_url == "https://api.perplexity.ai/chat/completions"
        assert "Authorization" in client.headers
        assert "Bearer test_key" in client.headers["Authorization"]

    @pytest.mark.unit
    def test_perplexity_client_no_api_key(self):
        """API 키 없을 때 예외 처리 테스트"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('src.issue_searcher.config') as mock_config:
                mock_config.get_perplexity_api_key.return_value = None

                from src.issue_searcher import PerplexityClient

                with pytest.raises(ValueError, match="Perplexity API 키가 설정되지 않았습니다"):
                    PerplexityClient()


class TestIssueItem:
    """IssueItem 데이터클래스 테스트"""

    @pytest.mark.unit
    def test_issue_item_creation(self):
        """IssueItem 생성 테스트"""
        from src.issue_searcher import IssueItem

        issue = IssueItem(
            title="테스트 이슈",
            summary="테스트 요약",
            source="테스트 소스",
            published_date="2024-01-15",
            relevance_score=0.85,
            category="news",
            content_snippet="테스트 내용"
        )

        assert issue.title == "테스트 이슈"
        assert issue.summary == "테스트 요약"
        assert issue.source == "테스트 소스"
        assert issue.published_date == "2024-01-15"
        assert issue.relevance_score == 0.85
        assert issue.category == "news"
        assert issue.content_snippet == "테스트 내용"


class TestSearchResult:
    """SearchResult 데이터클래스 테스트"""

    @pytest.mark.unit
    def test_search_result_creation(self):
        """SearchResult 생성 테스트"""
        from src.issue_searcher import SearchResult, IssueItem

        issue = IssueItem(
            title="테스트", summary="테스트", source="테스트",
            published_date=None, relevance_score=0.5,
            category="news", content_snippet="테스트"
        )

        result = SearchResult(
            query_keywords=["AI", "기술"],
            total_found=1,
            issues=[issue],
            search_time=2.5,
            api_calls_used=1,
            confidence_score=0.8,
            time_period="최근 1주일",
            raw_responses=["test response"]
        )

        assert result.query_keywords == ["AI", "기술"]
        assert result.total_found == 1
        assert len(result.issues) == 1
        assert result.search_time == 2.5
        assert result.api_calls_used == 1
        assert result.confidence_score == 0.8
        assert result.time_period == "최근 1주일"


class TestIssueSearcher:
    """IssueSearcher 테스트 클래스"""

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_perplexity_key'})
    def test_issue_searcher_initialization(self):
        """IssueSearcher 초기화 테스트"""
        from src.issue_searcher import IssueSearcher

        searcher = IssueSearcher(api_key="test_key")
        assert searcher.client.api_key == "test_key"
        assert searcher.max_keywords_per_search == 5
        assert searcher.max_results_per_search == 10

    @pytest.mark.unit
    def test_prepare_search_keywords(self):
        """검색 키워드 준비 테스트"""
        from src.issue_searcher import IssueSearcher
        from src.keyword_generator import KeywordResult

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            keyword_result = KeywordResult(
                topic="AI 기술",
                primary_keywords=["AI", "인공지능", "머신러닝", "딥러닝"],
                related_terms=["신경망", "알고리즘", "빅데이터"],
                synonyms=["Artificial Intelligence"],
                context_keywords=["기술혁신"],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test"
            )

            keywords = searcher._prepare_search_keywords(keyword_result)

            # 최대 5개 키워드, 핵심 키워드 우선
            assert len(keywords) <= 5
            assert "AI" in keywords
            assert "인공지능" in keywords
            assert "머신러닝" in keywords

    @pytest.mark.unit
    def test_parse_issue_section(self):
        """이슈 섹션 파싱 테스트"""
        from src.issue_searcher import IssueSearcher

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            section = """AI 기술 혁신 가속화
**요약**: 최근 AI 기술이 빠르게 발전하고 있습니다.
**출처**: Tech News
**일자**: 2024-01-15
**카테고리**: news"""

            issue = searcher._parse_issue_section(section, 1)

            assert issue is not None
            assert issue.title == "AI 기술 혁신 가속화"
            assert "AI 기술이 빠르게 발전" in issue.summary
            assert issue.source == "Tech News"
            assert issue.published_date == "2024-01-15"
            assert issue.category == "news"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_issues_from_keywords_success(self):
        """키워드 기반 이슈 검색 성공 테스트"""
        # Mock API 응답
        mock_api_response = {
            "choices": [{
                "message": {
                    "content": """**제목**: AI 기술 발전
**요약**: 인공지능 기술이 빠르게 발전하고 있습니다.
**출처**: Tech News
**일자**: 2024-01-15
**카테고리**: news

**제목**: 머신러닝 혁신
**요약**: 새로운 머신러닝 알고리즘이 개발되었습니다.
**출처**: AI Journal
**일자**: 2024-01-14
**카테고리**: academic"""
                }
            }]
        }

        with patch('src.issue_searcher.PerplexityClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.search_issues.return_value = mock_api_response
            mock_client_class.return_value = mock_client

            from src.issue_searcher import IssueSearcher
            from src.keyword_generator import KeywordResult

            keyword_result = KeywordResult(
                topic="AI 기술",
                primary_keywords=["AI", "인공지능"],
                related_terms=["머신러닝"],
                synonyms=["기계학습"],
                context_keywords=["기술혁신"],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test"
            )

            searcher = IssueSearcher(api_key="test_key")
            result = await searcher.search_issues_from_keywords(keyword_result)

            assert result.total_found == 2
            assert len(result.issues) == 2
            assert result.confidence_score > 0.0
            assert result.search_time > 0.0
            assert "AI" in result.query_keywords

            # 첫 번째 이슈 확인
            first_issue = result.issues[0]
            assert first_issue.title == "AI 기술 발전"
            assert "인공지능" in first_issue.summary
            assert first_issue.source == "Tech News"

            # raw_responses 검증
            assert len(result.raw_responses) == 1
            assert isinstance(result.raw_responses[0], str)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_issues_from_keywords_failure(self):
        """키워드 기반 이슈 검색 실패 테스트"""
        with patch('src.issue_searcher.PerplexityClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.search_issues.side_effect = Exception("API 오류")
            mock_client_class.return_value = mock_client

            from src.issue_searcher import IssueSearcher
            from src.keyword_generator import KeywordResult

            keyword_result = KeywordResult(
                topic="테스트",
                primary_keywords=["테스트"],
                related_terms=[],
                synonyms=[],
                context_keywords=[],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test"
            )

            searcher = IssueSearcher(api_key="test_key")
            result = await searcher.search_issues_from_keywords(keyword_result)

            # 폴백 결과 확인
            assert result.total_found == 0
            assert len(result.issues) == 0
            assert result.confidence_score == 0.1
            assert result.raw_responses == ["검색 실패로 인한 응답 없음"]

    @pytest.mark.unit
    def test_format_search_summary_success(self):
        """검색 결과 요약 포맷팅 테스트 (성공)"""
        from src.issue_searcher import IssueSearcher, SearchResult, IssueItem

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            issues = [
                IssueItem(
                    title="AI 기술 발전",
                    summary="인공지능 기술이 빠르게 발전하고 있습니다.",
                    source="Tech News",
                    published_date="2024-01-15",
                    relevance_score=0.9,
                    category="news",
                    content_snippet="test"
                )
            ]

            result = SearchResult(
                query_keywords=["AI", "기술"],
                total_found=1,
                issues=issues,
                search_time=2.5,
                api_calls_used=1,
                confidence_score=0.85,
                time_period="최근 1주일",
                raw_responses=["test"]
            )

            summary = searcher.format_search_summary(result)

            assert "이슈 검색 완료" in summary
            assert "1개 이슈 발견" in summary
            assert "85%" in summary
            assert "AI 기술 발전" in summary


class TestConvenienceFunctions:
    """편의 함수 테스트"""

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_key'})
    def test_create_issue_searcher(self):
        """create_issue_searcher 함수 테스트"""
        from src.issue_searcher import create_issue_searcher

        searcher = create_issue_searcher(api_key="test_key")
        assert searcher is not None
        assert searcher.client.api_key == "test_key"


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v"])