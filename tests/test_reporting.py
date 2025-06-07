# tests/test_reporting.py
import pytest
from src.reporting import format_search_summary, format_detailed_issue_report
from src.models import SearchResult, IssueItem, KeywordResult

@pytest.fixture
def sample_search_result():
    """테스트용 SearchResult 픽스처"""
    issue1 = IssueItem(title="이슈 1", summary="요약 1", source="출처 1", relevance_score=0.8, published_date="2024-01-01", category="news", content_snippet="...")
    issue2 = IssueItem(title="이슈 2", summary="요약 2", source="출처 2", relevance_score=0.6, published_date="2024-01-02", category="blog", content_snippet="...", detailed_content="상세 내용", background_context="배경 정보")
    return SearchResult(
        query_keywords=["테스트"], total_found=2, issues=[issue1, issue2],
        search_time=1.0, api_calls_used=2, confidence_score=0.7,
        time_period="최근 1주일", raw_responses=[], detailed_issues_count=1
    )

class TestReporting:
    """보고서 생성 함수들을 테스트합니다."""

    def test_format_search_summary(self, sample_search_result):
        """검색 결과 요약 포맷팅을 테스트합니다."""
        summary = format_search_summary(sample_search_result)
        assert "이슈 검색 완료" in summary
        assert "이슈 1" in summary
        assert "이슈 2" in summary
        assert "관련도: 80%" in summary

    def test_format_detailed_issue_report(self):
        """상세 이슈 보고서 포맷팅을 테스트합니다."""
        issue = IssueItem(
            title="상세 이슈", summary="요약", source="출처", relevance_score=0.9,
            published_date="2024-01-03", category="news", content_snippet="...",
            detailed_content="이것은 상세 내용입니다.",
            background_context="이것은 배경 정보입니다.",
            detail_confidence=0.85
        )
        report = format_detailed_issue_report(issue)
        assert "# 📋 상세 이슈" in report
        assert "## 📖 상세 내용" in report
        assert "이것은 상세 내용입니다." in report
        assert "## 🔗 배경 정보" in report
        assert "이것은 배경 정보입니다." in report