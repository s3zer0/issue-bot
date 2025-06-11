"""
보고서 생성 모듈에 대한 테스트 (PDF 지원 포함).

기존 마크다운 보고서 테스트와 함께 PDF 보고서 생성 기능도 테스트합니다.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
import tempfile

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.reporting.reporting import format_search_summary, format_detailed_issue_report, create_detailed_report_from_search_result
from src.models import SearchResult, IssueItem
from src.hallucination_detection.models import CombinedHallucinationScore


@pytest.fixture
def sample_search_result():
    """테스트용 SearchResult 픽스처"""
    issue1 = IssueItem(
        title="이슈 1",
        summary="요약 1",
        source="출처 1",
        relevance_score=0.8,
        published_date="2024-01-01",
        category="news",
        content_snippet="..."
    )
    # 환각 탐지 신뢰도 추가
    issue1.combined_confidence = 0.85
    # 환각 탐지 속성 추가 (threshold_manager가 찾는 속성)
    issue1.hallucination_confidence = 0.85

    issue2 = IssueItem(
        title="이슈 2",
        summary="요약 2",
        source="출처 2",
        relevance_score=0.6,
        published_date="2024-01-02",
        category="blog",
        content_snippet="...",
        detailed_content="상세 내용",
        background_context="배경 정보"
    )
    issue2.combined_confidence = 0.45
    # 환각 탐지 속성 추가 (threshold_manager가 찾는 속성)
    issue2.hallucination_confidence = 0.45

    result = SearchResult(
        query_keywords=["테스트"],
        total_found=2,
        issues=[issue1, issue2],
        search_time=1.0,
        api_calls_used=2,
        confidence_score=0.7,
        time_period="최근 1주일",
        raw_responses=[],
        detailed_issues_count=1
    )

    # 신뢰도 분포 추가
    result.confidence_distribution = {
        'high': 1,
        'moderate': 0,
        'low': 1
    }

    return result


class TestReporting:
    """기존 마크다운 보고서 생성 함수들을 테스트합니다."""

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
        assert "## 📋 이슈 2" in report  # 상세 내용이 있는 이슈2만 포함되어야 함
        assert "이슈 1" not in report  # 상세 내용이 없는 이슈1은 포함되지 않아야 함


class TestEnhancedReportingWithPDF:
    """향상된 보고서 생성 기능 (PDF 포함) 테스트"""

    @pytest.mark.asyncio
    async def test_enhanced_report_generator_initialization(self):
        """향상된 보고서 생성기 초기화 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        generator = EnhancedReportGenerator()
        assert generator.threshold_manager is not None
        assert generator.pdf_generator is not None

    @pytest.mark.asyncio
    async def test_generate_discord_embed(self, sample_search_result):
        """Discord 임베드 생성 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        generator = EnhancedReportGenerator()
        embed = generator.generate_discord_embed(sample_search_result)

        assert embed.title.startswith("🔍 이슈 모니터링 결과")
        assert "테스트" in embed.title
        assert len(embed.fields) > 0

        # 신뢰도 필드 확인
        confidence_field = next((f for f in embed.fields if "전체 신뢰도" in f.name), None)
        assert confidence_field is not None

    @pytest.mark.asyncio
    async def test_generate_detailed_markdown_report(self, sample_search_result):
        """상세 마크다운 보고서 생성 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        generator = EnhancedReportGenerator()
        report = generator.generate_detailed_report(sample_search_result)

        # 보고서 구조 확인
        assert "# 🔍 AI 이슈 모니터링 종합 보고서" in report
        assert "## 📋 핵심 요약" in report
        assert "## 🟢 높은 신뢰도 이슈" in report
        assert "## 🔴 낮은 신뢰도 이슈" in report
        assert "## 🛡️ 환각 탐지 분석" in report

        # 신뢰도 분포 정보 확인
        assert "높은 신뢰도: 1개" in report
        assert "낮은 신뢰도: 1개" in report

    @pytest.mark.asyncio
    async def test_save_markdown_report_to_file(self, sample_search_result):
        """마크다운 보고서 파일 저장 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        generator = EnhancedReportGenerator()

        with patch('builtins.open', mock_open()) as mock_file:
            with patch('os.makedirs'):
                file_path = generator.save_report_to_file("테스트 보고서 내용", "테스트주제")

                assert "report_테스트주제_" in file_path
                assert file_path.endswith("_validated.md")
                mock_file.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.pdf_report_generator.PDFReportGenerator.generate_report')
    async def test_generate_reports_with_pdf(self, mock_pdf_generate, sample_search_result):
        """PDF 포함 전체 보고서 생성 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        mock_pdf_generate.return_value = "reports/test.pdf"

        generator = EnhancedReportGenerator()

        with patch('builtins.open', mock_open()) as mock_file:
            with patch('os.makedirs'):
                embed, md_path, pdf_path = await generator.generate_reports(
                    sample_search_result,
                    "테스트주제",
                    generate_pdf=True
                )

                assert embed is not None
                assert "report_테스트주제_" in md_path
                assert pdf_path == "reports/test.pdf"
                mock_pdf_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_reports_without_pdf(self, sample_search_result):
        """PDF 없이 보고서 생성 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        generator = EnhancedReportGenerator()

        with patch('builtins.open', mock_open()) as mock_file:
            with patch('os.makedirs'):
                embed, md_path, pdf_path = await generator.generate_reports(
                    sample_search_result,
                    "테스트주제",
                    generate_pdf=False
                )

                assert embed is not None
                assert "report_테스트주제_" in md_path
                assert pdf_path is None

    @pytest.mark.asyncio
    async def test_pdf_generation_error_handling(self, sample_search_result):
        """PDF 생성 오류 처리 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        generator = EnhancedReportGenerator()

        with patch('src.pdf_report_generator.PDFReportGenerator.generate_report',
                   side_effect=Exception("PDF 생성 오류")):
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('os.makedirs'):
                    # PDF 생성 실패해도 마크다운은 생성되어야 함
                    embed, md_path, pdf_path = await generator.generate_reports(
                        sample_search_result,
                        "테스트주제",
                        generate_pdf=True
                    )

                    assert embed is not None
                    assert md_path is not None
                    assert pdf_path is None  # PDF 생성 실패

    def test_calculate_average_confidence(self, sample_search_result):
        """평균 신뢰도 계산 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        generator = EnhancedReportGenerator()
        avg_confidence = generator._calculate_average_confidence(sample_search_result.issues)

        # (0.85 + 0.45) / 2 = 0.65
        assert abs(avg_confidence - 0.65) < 0.01

    def test_format_issues_for_embed(self, sample_search_result):
        """임베드용 이슈 포맷팅 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        generator = EnhancedReportGenerator()
        formatted = generator._format_issues_for_embed(sample_search_result.issues[:1])

        assert "• **이슈 1**" in formatted
        assert "신뢰도: 85.0%" in formatted
        assert "출처: 출처 1" in formatted

    def test_create_detailed_issues_section(self, sample_search_result):
        """상세 이슈 섹션 생성 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import EnhancedReportGenerator

        generator = EnhancedReportGenerator()
        section = generator._create_detailed_issues_section(
            sample_search_result.issues,
            include_all=True
        )

        # 생성된 섹션이 한글을 포함하고 있는지 확인
        assert "### 이슈 1" in section
        assert "### 이슈 2" in section
        # "종합 신뢰도: 85.0%"가 포함되어 있는지 확인
        assert "85.0%" in section
        assert "45.0%" in section


class TestHelperFunctions:
    """헬퍼 함수 테스트"""

    @pytest.mark.asyncio
    @patch('src.hallucination_detection.enhanced_reporting_with_pdf.EnhancedReportGenerator')
    async def test_generate_all_reports_helper(self, mock_generator_class, sample_search_result):
        """generate_all_reports 헬퍼 함수 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import generate_all_reports

        mock_generator = AsyncMock()
        mock_generator.generate_reports.return_value = (
            MagicMock(),  # embed
            "reports/test.md",
            "reports/test.pdf"
        )
        mock_generator_class.return_value = mock_generator

        embed, md_path, pdf_path = await generate_all_reports(
            sample_search_result,
            "테스트",
            generate_pdf=True
        )

        assert md_path == "reports/test.md"
        assert pdf_path == "reports/test.pdf"
        mock_generator.generate_reports.assert_called_once_with(
            sample_search_result,
            "테스트",
            True
        )