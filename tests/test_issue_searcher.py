"""
이슈 검색기 pytest 테스트 - 4단계 세부 정보 수집 테스트 포함 (수정됨)
"""

import pytest
import sys
import os
from unittest.mock import patch, AsyncMock

# 프로젝트 루트 디렉토리의 src 폴더를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# 의존성 모듈 임포트
from src.issue_searcher import IssueSearcher
from src.reporting import create_detailed_report_from_search_result, format_detailed_issue_report
from src.models import KeywordResult, IssueItem, SearchResult


@pytest.fixture
def sample_keyword_result():
    """테스트용 KeywordResult 픽스처"""
    return KeywordResult(
        topic="AI 기술",
        primary_keywords=["AI", "인공지능"],
        related_terms=["머신러닝", "딥러닝"],
        context_keywords=["기술혁신"],
        confidence_score=0.8,
        generation_time=1.0,
        raw_response="test"
    )

class TestIssueSearcher:
    """IssueSearcher 테스트 클래스"""

    @pytest.mark.unit
    @patch('src.issue_searcher.PerplexityClient')
    def test_parse_issue_section(self, mock_client):
        """새로운 API 형식에 맞는 섹션 파싱 테스트"""
        searcher = IssueSearcher(api_key="test_key")
        # API의 새로운 형식 '## **...**'에 맞춰 테스트 데이터 수정
        section = "## **AI 혁신**\n**요약**: 내용입니다.\n**출처**: 출처\n**일자**: 2024-01-01"
        issue = searcher._parse_issue_section(section)
        assert issue is not None, "이슈 객체가 생성되어야 합니다."
        assert issue.title == "AI 혁신"
        assert issue.summary == "내용입니다."

    @pytest.mark.integration
    @pytest.mark.asyncio
    @patch('src.issue_searcher.PerplexityClient')
    async def test_search_issues_with_details(self, mock_client_class, sample_keyword_result):
        """ 세부 정보 포함 이슈 검색 통합 테스트"""
        mock_client = AsyncMock()

        mock_client.search_issues.return_value = {
            "choices": [{
                "message": {
                    "content": "## **AI 기술 혁신**\n**요약**: AI가 발전합니다.\n**출처**: Tech Journal"
                }
            }]
        }

        mock_client.collect_detailed_information.return_value = {
            "choices": [{
                "message": {
                    "content": """### 1. 핵심 기술 분석 (Core Technical Analysis)
    - **작동 원리**: AI의 기본 원리 설명

    ### 2. 배경 및 맥락 (Background Context)
    - **역사적 발전**: AI의 역사와 발전 과정
    - **문제 정의**: AI가 해결하려는 문제들

    ### 3. 심층 영향 분석 (Deep Impact Analysis)
    - **기술적 영향**: 산업에 미치는 영향"""
                }
            }]
        }

        mock_client_class.return_value = mock_client

        searcher = IssueSearcher(api_key="test_key")

        result = await searcher.search_issues_from_keywords(
            sample_keyword_result,
            time_period="최근 1주일",
            collect_details=True
        )

        assert result.total_found >= 1, "최소 1개 이상의 이슈를 찾아야 합니다."
        assert result.detailed_issues_count >= 1, "최소 1개 이상의 세부 정보가 수집되어야 합니다."

        first_issue = result.issues[0]
        assert "AI 기술 혁신" in first_issue.title

        # background_context가 None이 아닌지 먼저 확인
        assert first_issue.background_context is not None, "배경 정보가 수집되어야 합니다."
        assert "AI의 역사" in first_issue.background_context, "배경 정보가 올바르게 파싱되어야 합니다."


class TestConvenienceFunctions:
    """편의 함수 테스트"""

    @pytest.mark.unit
    def test_create_detailed_report_from_search_result(self):
        """상세 보고서 생성 편의 함수 테스트"""
        issue = IssueItem(
            title="AI 기술 혁신", summary="AI가 발전합니다.", source="Tech News",
            published_date="2024-01-15", relevance_score=0.9, category="news",
            content_snippet="...", detailed_content="상세 내용입니다.", detail_confidence=0.85
        )
        search_result = SearchResult(
            query_keywords=["AI"], total_found=1, issues=[issue], search_time=3.0,
            api_calls_used=3, confidence_score=0.8, time_period="최근 1주일",
            raw_responses=["..."], detailed_issues_count=1
        )

        report = create_detailed_report_from_search_result(search_result)

        # 보고서의 실제 내용과 형식을 검증하도록 변경
        assert "# 🔍 종합 이슈 분석 보고서" in report
        assert "키워드: AI" in report
        assert "## 📖 상세 내용" in report

    @pytest.mark.unit
    def test_format_detailed_issue_report(self):
        """개별 이슈 상세 보고서 포맷팅 테스트"""
        issue = IssueItem(
            title="Tesla 신모델", summary="요약 내용.", source="Tesla Blog", published_date="2024-01-15",
            relevance_score=0.9, category="news", content_snippet="...",
            detailed_content="상세 내용.", detail_confidence=0.88,
            background_context="전기차 시장 발전 배경"
        )
        report = format_detailed_issue_report(issue)
        assert "Tesla 신모델" in report
        assert "## 📖 상세 내용" in report
        assert "## 🔗 배경 정보" in report
        assert "전기차 시장 발전 배경" in report