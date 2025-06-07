import pytest
import sys
import os

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.reporting import format_search_summary, format_detailed_issue_report, create_detailed_report_from_search_result
from src.models import SearchResult, IssueItem

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
        assert "검증된 이슈 발견" in summary
        assert "이슈 1" in summary
        assert "이슈 2" in summary
        assert "관련도: 80%" in summary

    def test_create_detailed_report_from_search_result(self, sample_search_result):
        """상세 보고서 생성 함수를 테스트합니다."""
        report = create_detailed_report_from_search_result(sample_search_result)
        assert "# 🔍 종합 이슈 분석 보고서" in report
        assert "키워드: 테스트" in report
        assert "## 📋 이슈 2" in report # 상세 내용이 있는 이슈2만 포함되어야 함
        assert "이슈 1" not in report # 상세 내용이 없는 이슈1은 포함되지 않아야 함