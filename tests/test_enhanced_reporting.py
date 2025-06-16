"""
Enhanced Reporting 시스템 테스트.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import os

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.hallucination_detection.enhanced_reporting import EnhancedReportGenerator
from src.hallucination_detection.enhanced_reporting_with_pdf import generate_all_reports
from src.hallucination_detection.threshold_manager import ThresholdManager, ConfidenceLevel
from src.hallucination_detection.models import CombinedHallucinationScore, ConsistencyScore, RePPLScore
from src.models import SearchResult, IssueItem


class TestEnhancedReportGenerator:
    """Enhanced Report Generator의 단위 테스트."""

    @pytest.fixture
    def threshold_manager(self):
        """테스트용 ThresholdManager."""
        return ThresholdManager()

    @pytest.fixture
    def report_generator(self, threshold_manager):
        """테스트용 EnhancedReportGenerator."""
        return EnhancedReportGenerator(threshold_manager)

    @pytest.fixture
    def sample_issues(self):
        """테스트용 이슈 리스트."""
        issues = []
        
        # 높은 신뢰도 이슈
        high_confidence_issue = IssueItem(
            title="검증된 AI 기술 발전",
            summary="실제 검증된 AI 기술의 발전에 대한 정보",
            source="IEEE Computer Society",
            published_date="2024-01-15",
            relevance_score=0.9,
            category="technology",
            content_snippet="AI 기술이 실제로 발전하고 있음",
            detailed_content="상세한 기술 발전 내용..."
        )
        # 높은 신뢰도 설정
        setattr(high_confidence_issue, 'hallucination_confidence', 0.85)
        setattr(high_confidence_issue, 'combined_confidence', 0.85)
        issues.append(high_confidence_issue)
        
        # 중간 신뢰도 이슈
        moderate_confidence_issue = IssueItem(
            title="AI의 미래 전망",
            summary="AI의 미래에 대한 일반적인 전망",
            source="Tech Blog",
            published_date="2024-01-10",
            relevance_score=0.7,
            category="technology",
            content_snippet="AI가 미래에 영향을 미칠 것으로 예상"
        )
        setattr(moderate_confidence_issue, 'hallucination_confidence', 0.65)
        setattr(moderate_confidence_issue, 'combined_confidence', 0.65)
        issues.append(moderate_confidence_issue)
        
        # 낮은 신뢰도 이슈
        low_confidence_issue = IssueItem(
            title="AI가 모든 것을 대체할 것",
            summary="과장된 AI 대체 주장",
            source="Unknown Blog",
            published_date="2024-01-05",
            relevance_score=0.4,
            category="speculation",
            content_snippet="AI가 모든 인간 활동을 대체할 것이라는 주장"
        )
        setattr(low_confidence_issue, 'hallucination_confidence', 0.25)
        setattr(low_confidence_issue, 'combined_confidence', 0.25)
        issues.append(low_confidence_issue)
        
        return issues

    @pytest.fixture
    def sample_search_result(self, sample_issues):
        """테스트용 SearchResult."""
        return SearchResult(
            query_keywords=["AI", "technology", "future"],
            issues=sample_issues,
            total_found=3,
            search_time=2.5,
            time_period="2024년 1월",
            api_calls_used=5
        )

    def test_initialization(self, report_generator, threshold_manager):
        """초기화 테스트."""
        assert report_generator.threshold_manager == threshold_manager
        assert report_generator.pdf_generator is not None

    def test_calculate_average_confidence(self, report_generator, sample_issues):
        """평균 신뢰도 계산 테스트."""
        avg_confidence = report_generator._calculate_average_confidence(sample_issues)
        
        # (0.85 + 0.65 + 0.25) / 3 = 0.583...
        expected = (0.85 + 0.65 + 0.25) / 3
        assert abs(avg_confidence - expected) < 0.01

    def test_calculate_average_confidence_empty_list(self, report_generator):
        """빈 리스트의 평균 신뢰도 계산 테스트."""
        avg_confidence = report_generator._calculate_average_confidence([])
        assert avg_confidence == 0.0

    def test_calculate_average_confidence_no_confidence_attr(self, report_generator):
        """신뢰도 속성이 없는 이슈들의 테스트."""
        issues_without_confidence = [
            IssueItem(
                title="테스트 이슈",
                summary="신뢰도 속성 없음",
                source="Test",
                published_date="2024-01-01",
                relevance_score=0.5,
                category="test",
                content_snippet="테스트"
            )
        ]
        
        avg_confidence = report_generator._calculate_average_confidence(issues_without_confidence)
        assert avg_confidence == 0.5  # 기본값

    def test_create_summary_description(self, report_generator):
        """요약 설명 생성 테스트."""
        description = report_generator._create_summary_description(1, 1, 1)
        
        assert "총 **3개**의 이슈가 발견되었습니다" in description
        assert "🟢 높은 신뢰도: **1개**" in description
        assert "🟡 중간 신뢰도: **1개**" in description
        assert "🔴 낮은 신뢰도: **1개**" in description

    def test_create_summary_description_no_issues(self, report_generator):
        """이슈가 없는 경우 요약 설명 테스트."""
        description = report_generator._create_summary_description(0, 0, 0)
        assert description == "검색된 이슈가 없습니다."

    def test_format_confidence_field(self, report_generator):
        """신뢰도 필드 포맷팅 테스트."""
        confidence_summary = {
            'level': ConfidenceLevel.HIGH,
            'level_text': '높음',
            'score': 0.85,
            'recommendation': '대체로 신뢰할 수 있습니다.'
        }
        
        formatted = report_generator._format_confidence_field(confidence_summary)
        
        assert "🟢" in formatted  # 높은 신뢰도 이모지
        assert "**높음**" in formatted
        assert "(85.0%)" in formatted
        assert "대체로 신뢰할 수 있습니다." in formatted

    def test_get_confidence_emoji(self, report_generator):
        """신뢰도 이모지 테스트."""
        assert report_generator._get_confidence_emoji(ConfidenceLevel.VERY_HIGH) == "🟢"
        assert report_generator._get_confidence_emoji(ConfidenceLevel.HIGH) == "🟢"
        assert report_generator._get_confidence_emoji(ConfidenceLevel.MODERATE) == "🟡"
        assert report_generator._get_confidence_emoji(ConfidenceLevel.LOW) == "🟠"
        assert report_generator._get_confidence_emoji(ConfidenceLevel.VERY_LOW) == "🔴"
        assert report_generator._get_confidence_emoji(None) == "⚪"

    def test_format_issues_for_embed(self, report_generator, sample_issues):
        """임베드용 이슈 포맷팅 테스트."""
        formatted = report_generator._format_issues_for_embed(sample_issues[:2])
        
        assert "**검증된 AI 기술 발전**" in formatted
        assert "신뢰도: 85.0%" in formatted
        assert "출처: IEEE Computer Society" in formatted
        assert "**AI의 미래 전망**" in formatted
        assert "신뢰도: 65.0%" in formatted

    def test_create_report_header(self, report_generator, sample_search_result):
        """보고서 헤더 생성 테스트."""
        header = report_generator._create_report_header(sample_search_result)
        
        assert "# 🔍 AI 이슈 모니터링 종합 보고서" in header
        assert "AI, technology, future" in header
        assert "2024년 1월" in header
        assert "**총 이슈 수**: 3개" in header

    def test_create_executive_summary(self, report_generator, sample_search_result, sample_issues):
        """경영진 요약 생성 테스트."""
        high, moderate, low = [sample_issues[0]], [sample_issues[1]], [sample_issues[2]]
        
        summary = report_generator._create_executive_summary(sample_search_result, high, moderate, low)
        
        assert "## 📋 핵심 요약" in summary
        assert "### 신뢰도 분포" in summary
        assert "**높은 신뢰도**: 1개 (33.3%)" in summary
        assert "**중간 신뢰도**: 1개 (33.3%)" in summary
        assert "**낮은 신뢰도**: 1개 (33.3%)" in summary
        assert "### 주요 발견사항" in summary
        assert "1. **검증된 AI 기술 발전**" in summary

    def test_create_detailed_issues_section(self, report_generator, sample_issues):
        """상세 이슈 섹션 생성 테스트."""
        section = report_generator._create_detailed_issues_section(sample_issues[:2], include_all=True)
        
        assert "### 검증된 AI 기술 발전" in section
        assert "**출처**: IEEE Computer Society" in section
        assert "**종합 신뢰도**: 85.0%" in section
        assert "**요약**: 실제 검증된 AI 기술의 발전에 대한 정보" in section
        assert "---" in section  # 구분자

    def test_create_detailed_issues_section_limit(self, report_generator, sample_issues):
        """이슈 제한 테스트 (5개 초과)."""
        # 6개 이슈 생성
        many_issues = sample_issues * 2  # 3 * 2 = 6개
        
        section = report_generator._create_detailed_issues_section(many_issues, include_all=False)
        
        # 5개만 표시되고 나머지 언급
        assert "외 1개의 이슈가 더 있습니다" in section

    def test_format_detailed_issue(self, report_generator, sample_issues):
        """개별 이슈 상세 포맷팅 테스트."""
        issue = sample_issues[0]  # 높은 신뢰도 이슈
        
        # 환각 점수 추가
        hallucination_score = CombinedHallucinationScore(
            individual_scores={
                'RePPL': RePPLScore(confidence=0.8, repetition_score=0.8, perplexity_score=0.8, entropy_score=0.8),
                'Self-Consistency': ConsistencyScore(
                    confidence=0.9, consistency_rate=0.9, num_queries=3, num_consistent=3,
                    variations=[], common_elements=[], divergent_elements=[], analysis_details={}
                )
            },
            weights={'RePPL': 0.5, 'Self-Consistency': 0.5},
            final_confidence=0.85
        )
        setattr(issue, 'hallucination_score', hallucination_score)
        
        formatted = report_generator._format_detailed_issue(issue)
        
        assert "### 검증된 AI 기술 발전" in formatted
        assert "**종합 신뢰도**: 85.0%" in formatted
        assert "**세부 신뢰도 점수**:" in formatted
        assert "- RePPL: 80.0%" in formatted
        assert "- 자기 일관성: 90.0%" in formatted
        assert "**상세 내용**:" in formatted

    def test_create_hallucination_analysis_summary(self, report_generator, sample_search_result):
        """환각 탐지 분석 요약 생성 테스트."""
        summary = report_generator._create_hallucination_analysis_summary(sample_search_result)
        
        assert "## 🛡️ 환각 탐지 분석" in summary
        assert "### 탐지 시스템" in summary
        assert "RePPL (Relevant Paraphrased Prompt with Logit)" in summary
        assert "자기 일관성 검사 (Self-Consistency Check)" in summary
        assert "LLM-as-Judge" in summary

    def test_create_report_footer(self, report_generator, sample_search_result):
        """보고서 푸터 생성 테스트."""
        footer = report_generator._create_report_footer(sample_search_result)
        
        assert "## 📌 메타데이터" in footer
        assert "**검색 소요 시간**: 2.5초" in footer
        assert "**API 호출 횟수**: 5회" in footer
        assert "**검색 기간**: 2024년 1월" in footer
        assert "AI 환각 탐지 시스템에 의해 검증되었습니다" in footer

    def test_generate_discord_embed(self, report_generator, sample_search_result):
        """Discord 임베드 생성 테스트."""
        embed = report_generator.generate_discord_embed(sample_search_result)
        
        assert "🔍 이슈 모니터링 결과: AI, technology, future" in embed.title
        assert embed.color is not None
        assert len(embed.fields) >= 3  # 신뢰도, 높은 신뢰도 이슈, 중간 신뢰도 이슈 등

    def test_generate_detailed_report(self, report_generator, sample_search_result):
        """상세 마크다운 보고서 생성 테스트."""
        report = report_generator.generate_detailed_report(sample_search_result)
        
        assert "# 🔍 AI 이슈 모니터링 종합 보고서" in report
        assert "## 📋 핵심 요약" in report
        assert "## 🟢 높은 신뢰도 이슈" in report
        assert "## 🟡 중간 신뢰도 이슈" in report
        assert "## 🔴 낮은 신뢰도 이슈" in report
        assert "## 🛡️ 환각 탐지 분석" in report
        assert "## 📌 메타데이터" in report

    @patch('os.makedirs')
    @patch('builtins.open', create=True)
    def test_save_report_to_file(self, mock_open, mock_makedirs, report_generator):
        """보고서 파일 저장 테스트."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        report_content = "# 테스트 보고서\n\n테스트 내용"
        topic = "AI 기술"
        
        file_path = report_generator.save_report_to_file(report_content, topic)
        
        assert "reports" in file_path
        assert "AI 기술" in file_path
        assert "_validated.md" in file_path
        mock_makedirs.assert_called_once_with("reports", exist_ok=True)
        mock_file.write.assert_called_once_with(report_content)


class TestEnhancedReportingWithPDF:
    """PDF 포함 향상된 보고서 생성 테스트."""

    @pytest.fixture
    def sample_search_result(self):
        """테스트용 SearchResult."""
        issues = [
            IssueItem(
                title="테스트 이슈",
                summary="테스트 요약",
                source="Test Source",
                published_date="2024-01-01",
                relevance_score=0.8,
                category="test",
                content_snippet="테스트 내용"
            )
        ]
        setattr(issues[0], 'hallucination_confidence', 0.7)
        
        return SearchResult(
            query_keywords=["test"],
            issues=issues,
            total_found=1,
            search_time=1.0,
            time_period="2024-01",
            api_calls_used=2
        )

    @pytest.mark.asyncio
    async def test_generate_all_reports_success(self, sample_search_result):
        """모든 보고서 생성 성공 테스트."""
        with patch('src.hallucination_detection.enhanced_reporting_with_pdf.EnhancedReportGenerator') as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator_class.return_value = mock_generator
            
            # Mock Discord embed
            mock_embed = MagicMock()
            mock_embed.title = "테스트 임베드"
            mock_generator.generate_discord_embed.return_value = mock_embed
            
            # Mock detailed report
            mock_generator.generate_detailed_report.return_value = "# 테스트 보고서"
            mock_generator.save_report_to_file.return_value = "/path/to/report.md"
            
            # Mock PDF generation
            with patch('src.hallucination_detection.enhanced_reporting_with_pdf.PDFReportGenerator') as mock_pdf_class:
                mock_pdf_generator = MagicMock()
                mock_pdf_class.return_value = mock_pdf_generator
                mock_pdf_generator.generate_report = AsyncMock(return_value="/path/to/report.pdf")
                
                embed, markdown_path, pdf_path = await generate_all_reports(
                    sample_search_result, "test topic", generate_pdf=True
                )
                
                assert embed == mock_embed
                assert markdown_path == "/path/to/report.md"
                assert pdf_path == "/path/to/report.pdf"

    @pytest.mark.asyncio
    async def test_generate_all_reports_pdf_disabled(self, sample_search_result):
        """PDF 생성 비활성화 테스트."""
        with patch('src.hallucination_detection.enhanced_reporting_with_pdf.EnhancedReportGenerator') as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator_class.return_value = mock_generator
            
            mock_embed = MagicMock()
            mock_generator.generate_discord_embed.return_value = mock_embed
            mock_generator.generate_detailed_report.return_value = "# 테스트 보고서"
            mock_generator.save_report_to_file.return_value = "/path/to/report.md"
            
            embed, markdown_path, pdf_path = await generate_all_reports(
                sample_search_result, "test topic", generate_pdf=False
            )
            
            assert embed == mock_embed
            assert markdown_path == "/path/to/report.md"
            assert pdf_path is None

    @pytest.mark.asyncio
    async def test_generate_all_reports_pdf_error(self, sample_search_result):
        """PDF 생성 실패 테스트."""
        with patch('src.hallucination_detection.enhanced_reporting_with_pdf.EnhancedReportGenerator') as mock_generator_class:
            mock_generator = MagicMock()
            mock_generator_class.return_value = mock_generator
            
            mock_embed = MagicMock()
            mock_generator.generate_discord_embed.return_value = mock_embed
            mock_generator.generate_detailed_report.return_value = "# 테스트 보고서"
            mock_generator.save_report_to_file.return_value = "/path/to/report.md"
            
            # PDF 생성 실패 시뮬레이션
            with patch('src.hallucination_detection.enhanced_reporting_with_pdf.PDFReportGenerator') as mock_pdf_class:
                mock_pdf_generator = MagicMock()
                mock_pdf_class.return_value = mock_pdf_generator
                mock_pdf_generator.generate_report = AsyncMock(side_effect=Exception("PDF generation failed"))
                
                embed, markdown_path, pdf_path = await generate_all_reports(
                    sample_search_result, "test topic", generate_pdf=True
                )
                
                assert embed == mock_embed
                assert markdown_path == "/path/to/report.md"
                assert pdf_path is None  # PDF 실패로 인해 None


class TestEnhancedReportingEdgeCases:
    """Enhanced Reporting의 엣지 케이스 테스트."""

    @pytest.fixture
    def report_generator(self):
        """테스트용 report generator."""
        return EnhancedReportGenerator()

    def test_low_confidence_summary_creation(self, report_generator):
        """낮은 신뢰도 이슈 요약 생성 테스트."""
        low_issues = []
        for i in range(12):  # 10개 초과
            issue = IssueItem(
                title=f"낮은 신뢰도 이슈 {i}",
                summary=f"의심스러운 주장 {i}",
                source="Unknown",
                published_date="2024-01-01",
                relevance_score=0.2,
                category="suspicious",
                content_snippet=f"의심스러운 내용 {i}"
            )
            setattr(issue, 'combined_confidence', 0.2 + i * 0.01)  # 0.2~0.31
            low_issues.append(issue)
        
        summary = report_generator._create_low_confidence_summary(low_issues)
        
        assert "다음 12개의 이슈는 신뢰도가 낮아" in summary
        assert "낮은 신뢰도 이슈 0" in summary
        assert "외 2개의 낮은 신뢰도 이슈" in summary  # 10개 초과 분

    def test_format_detailed_issue_without_hallucination_score(self, report_generator):
        """환각 점수가 없는 이슈 포맷팅 테스트."""
        issue = IssueItem(
            title="환각 점수 없음",
            summary="환각 점수가 설정되지 않은 이슈",
            source="Test",
            published_date="2024-01-01",
            relevance_score=0.5,
            category="test",
            content_snippet="테스트"
        )
        setattr(issue, 'combined_confidence', 0.6)
        
        formatted = report_generator._format_detailed_issue(issue)
        
        assert "### 환각 점수 없음" in formatted
        assert "**종합 신뢰도**: 60.0%" in formatted
        # 세부 신뢰도 점수 섹션이 없어야 함
        assert "**세부 신뢰도 점수**:" not in formatted

    def test_create_executive_summary_no_high_confidence(self, report_generator):
        """높은 신뢰도 이슈가 없는 경우 테스트."""
        issues = [
            IssueItem(
                title="중간 신뢰도 이슈",
                summary="중간 정도의 신뢰도",
                source="Test",
                published_date="2024-01-01",
                relevance_score=0.6,
                category="test",
                content_snippet="테스트"
            )
        ]
        setattr(issues[0], 'combined_confidence', 0.6)
        
        search_result = SearchResult(
            query_keywords=["test"],
            issues=issues,
            total_found=1,
            search_time=1.0,
            time_period="2024-01",
            api_calls_used=1
        )
        
        summary = report_generator._create_executive_summary(search_result, [], issues, [])
        
        assert "## 📋 핵심 요약" in summary
        assert "**높은 신뢰도**: 0개 (0.0%)" in summary
        assert "**중간 신뢰도**: 1개 (100.0%)" in summary
        # 주요 발견사항 섹션이 없어야 함 (높은 신뢰도 이슈가 없으므로)
        assert "### 주요 발견사항" not in summary