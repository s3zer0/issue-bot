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

class TestIssueSearcherImproved:
    """개선된 IssueSearcher 테스트"""

    @pytest.mark.unit
    def test_extract_field_multiple_names(self):
        """다양한 필드명으로 값을 추출하는 테스트"""
        searcher = IssueSearcher(api_key="test_key")
        text = """
        **출처**: TechCrunch
        **Date**: 2024-01-15
        **카테고리**: 뉴스
        """

        # 한글/영문 모두 추출 가능해야 함
        assert searcher._extract_field(text, ['출처', 'Source']) == "TechCrunch"
        assert searcher._extract_field(text, ['발행일', 'Date']) == "2024-01-15"
        assert searcher._extract_field(text, ['카테고리', 'Category']) == "뉴스"

    @pytest.mark.unit
    def test_clean_source(self):
        """출처 정리 로직 테스트"""
        searcher = IssueSearcher(api_key="test_key")

        # URL에서 도메인 추출
        assert searcher._clean_source("https://techcrunch.com/2024/01/15/article") == "techcrunch.com"
        assert searcher._clean_source("https://www.reuters.com/article") == "reuters.com"

        # Unknown 처리
        assert searcher._clean_source("Unknown") == "Unknown"
        assert searcher._clean_source("N/A") == "Unknown"
        assert searcher._clean_source(None) == "Unknown"

        # 일반 텍스트는 그대로
        assert searcher._clean_source("TechCrunch") == "TechCrunch"

    @pytest.mark.unit
    def test_parse_date(self):
        """날짜 파싱 로직 테스트"""
        searcher = IssueSearcher(api_key="test_key")

        # 다양한 형식 파싱
        assert searcher._parse_date("2024-01-15") == "2024-01-15"
        assert searcher._parse_date("2024/01/15") == "2024-01-15"
        assert searcher._parse_date("2024.01.15") == "2024-01-15"
        assert searcher._parse_date("2024년 1월 15일") == "2024-01-15"

        # 유효하지 않은 날짜
        assert searcher._parse_date("N/A") is None
        assert searcher._parse_date("unknown") is None
        assert searcher._parse_date(None) is None

    @pytest.mark.unit
    @patch('src.issue_searcher.PerplexityClient')
    def test_parse_issue_section_improved(self, mock_client):
        """개선된 API 응답 파싱 테스트"""
        searcher = IssueSearcher(api_key="test_key")

        # 더 현실적인 API 응답
        section = """## **iOS 19 대규모 UI 개편 예정**
**요약**: 애플이 WWDC 2025에서 iOS 19를 공개할 예정이며, 2013년 iOS 7 이후 최대 규모의 UI 개편이 예상됩니다.
**출처**: https://techcrunch.com/2024/12/20/ios-19-ui-redesign
**발행일**: 2024-12-20
**카테고리**: 뉴스
**기술적 핵심**: visionOS 스타일의 둥근 버튼과 반투명 UI 디자인 적용
**중요도**: Critical
**관련 키워드**: iOS, Swift, UI/UX"""

        issue = searcher._parse_issue_section(section)

        assert issue is not None
        assert issue.title == "iOS 19 대규모 UI 개편 예정"
        assert issue.source == "techcrunch.com"
        assert issue.published_date == "2024-12-20"
        assert issue.category == "뉴스"
        assert hasattr(issue, 'technical_core')
        assert hasattr(issue, 'importance')
        assert getattr(issue, 'importance') == "Critical"

    @pytest.mark.unit
    def test_calculate_relevance_scores_improved(self):
        """개선된 관련도 점수 계산 테스트"""
        searcher = IssueSearcher(api_key="test_key")

        keyword_result = KeywordResult(
            topic="iOS Development",
            primary_keywords=["iOS", "Swift", "SwiftUI"],
            related_terms=["iPhone", "Xcode"],
            context_keywords=["Mobile", "Apple"],
            confidence_score=0.9,
            generation_time=1.0,
            raw_response=""
        )

        # 높은 관련도 이슈
        high_relevance_issue = IssueItem(
            title="iOS 19 SwiftUI 새로운 기능",
            summary="Swift와 SwiftUI의 혁신적인 업데이트가 iOS 19에 포함됩니다.",
            source="apple.com",
            published_date="2024-12-20",
            relevance_score=0.5,
            category="news",
            content_snippet="..."
        )
        setattr(high_relevance_issue, 'importance', 'Critical')

        # 낮은 관련도 이슈
        low_relevance_issue = IssueItem(
            title="Android 개발 동향",
            summary="구글이 새로운 Android 버전을 발표했습니다.",
            source="Unknown",
            published_date=None,
            relevance_score=0.5,
            category="news",
            content_snippet="..."
        )

        issues = [high_relevance_issue, low_relevance_issue]
        scored_issues = searcher._calculate_relevance_scores(issues, keyword_result)

        # 첫 번째 이슈가 훨씬 높은 점수를 받아야 함
        assert scored_issues[0].relevance_score > 0.7
        assert scored_issues[1].relevance_score < 0.3
