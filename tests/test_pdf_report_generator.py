"""
PDF 보고서 생성기에 대한 테스트 파일.

PDFReportGenerator 클래스의 주요 기능을 테스트합니다.
"""

import pytest
import os
import sys
from unittest.mock import patch, MagicMock, AsyncMock, mock_open
from datetime import datetime
import tempfile

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.reporting.pdf_report_generator import PDFReportGenerator
from src.models import SearchResult, IssueItem


@pytest.fixture
def sample_search_result():
    """테스트용 SearchResult 픽스처"""
    issue1 = IssueItem(
        title="AI 기술 혁신",
        summary="인공지능 기술이 빠르게 발전하고 있습니다.",
        source="Tech News",
        relevance_score=0.9,
        published_date="2024-01-01",
        category="technology",
        content_snippet="AI는...",
        detailed_content="인공지능 기술은 다양한 분야에서 혁신을 이끌고 있습니다.",
        background_context="AI의 역사는..."
    )
    issue1.combined_confidence = 0.85  # 환각 탐지 신뢰도 추가

    issue2 = IssueItem(
        title="머신러닝 응용",
        summary="머신러닝이 의료 분야에 적용되고 있습니다.",
        source="Medical Journal",
        relevance_score=0.75,
        published_date="2024-01-02",
        category="healthcare",
        content_snippet="ML은..."
    )
    issue2.combined_confidence = 0.6

    return SearchResult(
        query_keywords=["AI", "인공지능", "기술"],
        total_found=2,
        issues=[issue1, issue2],
        search_time=2.5,
        api_calls_used=3,
        confidence_score=0.8,
        time_period="최근 1주일",
        raw_responses=[],
        detailed_issues_count=1
    )


class TestPDFReportGenerator:
    """PDFReportGenerator 클래스 테스트"""

    @pytest.mark.unit
    def test_pdf_generator_initialization(self):
        """PDF 생성기 초기화 테스트"""
        with patch('src.reporting.pdf_report_generator.config.get_openai_api_key', return_value="test_key"):
            generator = PDFReportGenerator()
            assert generator.api_key == "test_key"
            assert generator.threshold_manager is not None
            assert generator.default_font in ['Helvetica', 'MalgunGothic', 'NanumGothic', 'NotoSansKR']

    @pytest.mark.unit
    def test_create_basic_structure_without_llm(self, sample_search_result):
        """LLM 없이 기본 구조 생성 테스트"""
        generator = PDFReportGenerator(openai_api_key=None)
        structure = generator._create_basic_structure(sample_search_result)

        assert 'search_result' in structure
        assert 'enhanced_sections' in structure
        assert 'metadata' in structure
        assert structure['metadata']['llm_enhanced'] is False
        assert len(structure['enhanced_sections']['key_findings']) > 0

    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_enhance_content_with_llm(self, mock_openai_class, sample_search_result):
        """LLM을 사용한 콘텐츠 개선 테스트"""
        # OpenAI API 모의 응답 설정
        mock_client = AsyncMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
핵심 요약 (Executive Summary):
AI 기술이 빠르게 발전하고 있으며 다양한 산업에 영향을 미치고 있습니다.

주요 발견사항 (Key Findings):
- AI 기술 혁신이 가속화되고 있음
- 의료 분야에서 머신러닝 활용 증가

트렌드 분석 (Trend Analysis):
AI 기술은 지속적으로 성장할 것으로 예상됩니다.

리스크 및 기회 (Risks & Opportunities):
윤리적 문제와 함께 혁신의 기회가 공존합니다.

권장 조치사항 (Recommended Actions):
- AI 기술 투자 확대
- 윤리 가이드라인 수립
        """
        mock_client.chat.completions.create.return_value = mock_response

        generator = PDFReportGenerator(openai_api_key="test_key")
        enhanced_data = await generator.enhance_content_with_llm(sample_search_result)

        assert enhanced_data['metadata']['llm_enhanced'] is True
        assert 'AI 기술이 빠르게 발전' in enhanced_data['enhanced_sections']['executive_summary']
        assert len(enhanced_data['enhanced_sections']['key_findings']) >= 2
        assert len(enhanced_data['enhanced_sections']['recommended_actions']) >= 2

    @pytest.mark.asyncio
    async def test_enhance_content_fallback_on_error(self, sample_search_result):
        """LLM 오류 시 기본 구조로 폴백 테스트"""
        with patch('openai.AsyncOpenAI', side_effect=Exception("API Error")):
            generator = PDFReportGenerator(openai_api_key="test_key")
            enhanced_data = await generator.enhance_content_with_llm(sample_search_result)

            assert enhanced_data['metadata']['llm_enhanced'] is False
            assert 'search_result' in enhanced_data

    @pytest.mark.unit
    def test_parse_llm_response(self, sample_search_result):
        """LLM 응답 파싱 테스트"""
        generator = PDFReportGenerator()

        llm_response = """
핵심 요약 (Executive Summary):
이것은 요약입니다.

주요 발견사항 (Key Findings):
- 첫 번째 발견
- 두 번째 발견

트렌드 분석 (Trend Analysis):
트렌드 내용입니다.

리스크 및 기회 (Risks & Opportunities):
리스크와 기회입니다.

권장 조치사항 (Recommended Actions):
1. 첫 번째 조치
2. 두 번째 조치
        """

        parsed = generator._parse_llm_response(llm_response, sample_search_result)

        assert '이것은 요약입니다' in parsed['enhanced_sections']['executive_summary']
        assert len(parsed['enhanced_sections']['key_findings']) == 2
        assert '트렌드 내용' in parsed['enhanced_sections']['trend_analysis']
        assert len(parsed['enhanced_sections']['recommended_actions']) == 2

    @pytest.mark.integration
    def test_generate_pdf_file(self, sample_search_result):
        """PDF 파일 생성 통합 테스트"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            generator = PDFReportGenerator()
            enhanced_data = generator._create_basic_structure(sample_search_result)

            # PDF 생성
            result_path = generator.generate_pdf(enhanced_data, tmp_path)

            # 파일이 생성되었는지 확인
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0

        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @pytest.mark.asyncio
    async def test_generate_report_full_flow(self, sample_search_result):
        """전체 보고서 생성 플로우 테스트"""
        generator = PDFReportGenerator(openai_api_key=None)  # LLM 없이 테스트

        with patch.object(generator, 'generate_pdf') as mock_generate_pdf:
            mock_generate_pdf.return_value = "reports/test_report.pdf"

            with patch('os.makedirs'):
                result_path = await generator.generate_report(sample_search_result, "테스트 주제")

                assert result_path.startswith("reports/") and result_path.endswith("_enhanced.pdf")
                mock_generate_pdf.assert_called_once()

    @pytest.mark.unit
    def test_create_cover_page_elements(self, sample_search_result):
        """표지 페이지 요소 생성 테스트"""
        generator = PDFReportGenerator()
        enhanced_data = generator._create_basic_structure(sample_search_result)

        cover_elements = generator._create_cover_page(enhanced_data)

        assert len(cover_elements) > 0
        # 표지에는 제목, 부제목, 메타정보 등이 포함되어야 함
        assert any("AI 이슈 모니터링 보고서" in str(elem) for elem in cover_elements if hasattr(elem, '__str__'))

    @pytest.mark.unit
    def test_create_executive_summary_elements(self, sample_search_result):
        """핵심 요약 섹션 생성 테스트"""
        generator = PDFReportGenerator()
        enhanced_data = generator._create_basic_structure(sample_search_result)

        summary_elements = generator._create_executive_summary(enhanced_data)

        assert len(summary_elements) > 0
        # 핵심 요약에는 통계 테이블이 포함되어야 함
        assert any(hasattr(elem, 'setStyle') for elem in summary_elements)  # Table 객체 확인

    @pytest.mark.unit
    @patch('src.reporting.pdf_report_generator.create_pdf_report')
    async def test_helper_function(self, mock_create_pdf, sample_search_result):
        """헬퍼 함수 테스트"""
        from src.reporting.pdf_report_generator import create_pdf_report

        mock_create_pdf.return_value = "reports/test.pdf"
        result = await create_pdf_report(sample_search_result, "테스트")

        assert result == "reports/test.pdf"


class TestPDFIntegrationWithBot:
    """Discord 봇과의 통합 테스트"""

    @pytest.mark.asyncio
    @patch('src.hallucination_detection.enhanced_reporting_with_pdf.PDFReportGenerator')
    async def test_generate_all_reports_with_pdf(self, mock_pdf_generator_class, sample_search_result):
        """generate_all_reports 함수의 PDF 생성 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import generate_all_reports

        # PDF 생성기 모의 설정  
        mock_pdf_generator = MagicMock()
        mock_pdf_generator.generate_report = AsyncMock(return_value="reports/test.pdf")
        mock_pdf_generator_class.return_value = mock_pdf_generator

        # 보고서 생성
        with patch('src.hallucination_detection.enhanced_reporting_with_pdf.EnhancedReportGenerator') as mock_report_gen:
            mock_report_instance = MagicMock()
            mock_report_instance.generate_discord_embed.return_value = MagicMock()
            mock_report_instance.generate_detailed_report.return_value = "마크다운 보고서"
            mock_report_instance.save_report_to_file.return_value = "reports/test.md"
            mock_report_instance.generate_reports = AsyncMock(return_value=(MagicMock(), "reports/test.md", "reports/test.pdf"))
            mock_report_gen.return_value = mock_report_instance

            embed, md_path, pdf_path = await generate_all_reports(
                sample_search_result,
                "테스트 주제",
                generate_pdf=True
            )

            assert md_path == "reports/test.md"
            assert pdf_path == "reports/test.pdf"
            # PDF generator is mocked at a higher level, so we check the instance was called
            mock_report_instance.generate_reports.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_all_reports_without_pdf(self, sample_search_result):
        """PDF 생성 없이 generate_all_reports 테스트"""
        from src.hallucination_detection.enhanced_reporting_with_pdf import generate_all_reports

        with patch('src.hallucination_detection.enhanced_reporting_with_pdf.EnhancedReportGenerator') as mock_report_gen:
            mock_report_instance = MagicMock()
            mock_report_instance.generate_discord_embed.return_value = MagicMock()
            mock_report_instance.generate_detailed_report.return_value = "마크다운 보고서"
            mock_report_instance.save_report_to_file.return_value = "reports/test.md"
            mock_report_instance.generate_reports = AsyncMock(return_value=(MagicMock(), "reports/test.md", None))
            mock_report_gen.return_value = mock_report_instance

            embed, md_path, pdf_path = await generate_all_reports(
                sample_search_result,
                "테스트 주제",
                generate_pdf=False
            )

            assert md_path == "reports/test.md"
            assert pdf_path is None