"""
환각 탐지 결과가 통합된 향상된 보고서 생성 모듈 (PDF 지원 추가).

신뢰도 등급별로 이슈를 분류하고, 환각 탐지 결과를 명확히 표시하여
사용자가 정보의 신뢰성을 쉽게 판단할 수 있도록 합니다.
마크다운과 PDF 두 가지 형식의 보고서를 생성할 수 있습니다.
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import discord
from loguru import logger
import asyncio

from src.models import SearchResult, IssueItem
from src.hallucination_detection.models import CombinedHallucinationScore
from src.hallucination_detection.threshold_manager import (
    ThresholdManager, ConfidenceLevel
)
from src.reporting.pdf_report_generator import PDFReportGenerator  # 새로 추가한 PDF 생성 모듈


class EnhancedReportGenerator:
    """환각 탐지 결과가 통합된 보고서 생성기 (PDF 지원)."""

    def __init__(self, threshold_manager: Optional[ThresholdManager] = None):
        """
        보고서 생성기를 초기화합니다.

        Args:
            threshold_manager: 임계값 관리자 (없으면 기본값 사용)
        """
        self.threshold_manager = threshold_manager or ThresholdManager()
        self.pdf_generator = PDFReportGenerator()  # PDF 생성기 초기화
        logger.info("향상된 보고서 생성기 초기화 완료 (PDF 지원)")

    async def generate_reports(
        self,
        search_result: SearchResult,
        topic: str,
        generate_pdf: bool = True
    ) -> Tuple[discord.Embed, str, Optional[str]]:
        """
        Discord 임베드, 마크다운 파일, PDF 파일을 모두 생성합니다.

        Args:
            search_result: 검색 결과
            topic: 보고서 주제
            generate_pdf: PDF 생성 여부

        Returns:
            Tuple[discord.Embed, str, Optional[str]]:
                - Discord 임베드
                - 마크다운 파일 경로
                - PDF 파일 경로 (generate_pdf가 True인 경우)
        """
        # 1. Discord 임베드 생성
        embed = self.generate_discord_embed(search_result)

        # 2. 마크다운 보고서 생성 및 저장
        markdown_report = self.generate_detailed_report(search_result)
        markdown_path = self.save_report_to_file(markdown_report, topic)

        # 3. PDF 보고서 생성 (옵션)
        pdf_path = None
        if generate_pdf:
            try:
                pdf_path = await self.pdf_generator.generate_report(search_result, topic)
                logger.info(f"PDF 보고서 생성 완료: {pdf_path}")
            except Exception as e:
                logger.error(f"PDF 생성 실패, 마크다운만 제공: {e}")

        return embed, markdown_path, pdf_path

    def generate_discord_embed(self, search_result: SearchResult) -> discord.Embed:
        """
        Discord 임베드 형식의 요약 보고서를 생성합니다.

        Args:
            search_result: 검색 결과

        Returns:
            discord.Embed: 포맷팅된 임베드
        """
        # 신뢰도별 이슈 분류
        high, moderate, low = self.threshold_manager.filter_issues_by_confidence(
            search_result.issues
        )

        # 전체 평균 신뢰도 계산
        avg_confidence = self._calculate_average_confidence(search_result.issues)
        confidence_summary = self.threshold_manager.get_confidence_summary(avg_confidence)

        # 임베드 생성
        embed = discord.Embed(
            title=f"🔍 이슈 모니터링 결과: {', '.join(search_result.query_keywords[:3])}",
            description=self._create_summary_description(len(high), len(moderate), len(low)),
            color=confidence_summary['color'],
            timestamp=datetime.now()
        )

        # 신뢰도 요약 필드
        embed.add_field(
            name="📊 전체 신뢰도",
            value=self._format_confidence_field(confidence_summary),
            inline=False
        )

        # 높은 신뢰도 이슈 (최대 3개)
        if high:
            embed.add_field(
                name="✅ 높은 신뢰도 이슈",
                value=self._format_issues_for_embed(high[:3]),
                inline=False
            )

        # 중간 신뢰도 이슈 (최대 2개)
        if moderate:
            embed.add_field(
                name="⚠️ 중간 신뢰도 이슈",
                value=self._format_issues_for_embed(moderate[:2]),
                inline=False
            )

        # 낮은 신뢰도 이슈 요약
        if low and self.threshold_manager.thresholds.include_low_confidence:
            embed.add_field(
                name="❌ 낮은 신뢰도 이슈",
                value=f"{len(low)}개의 이슈가 신뢰도 기준을 충족하지 못했습니다.",
                inline=False
            )

        # 보고서 파일 안내
        embed.add_field(
            name="📄 상세 보고서",
            value="마크다운과 PDF 형식의 상세 보고서가 첨부되었습니다.",
            inline=False
        )

        # 메타데이터
        embed.set_footer(
            text=f"검색 시간: {search_result.search_time:.1f}초 | "
                 f"API 호출: {search_result.api_calls_used}회 | "
                 f"기간: {search_result.time_period}"
        )

        return embed

    def generate_detailed_report(self, search_result: SearchResult) -> str:
        """
        상세 마크다운 보고서를 생성합니다.

        Args:
            search_result: 검색 결과

        Returns:
            str: 마크다운 형식의 상세 보고서
        """
        # 신뢰도별 이슈 분류
        high, moderate, low = self.threshold_manager.filter_issues_by_confidence(
            search_result.issues
        )

        report = self._create_report_header(search_result)
        report += self._create_executive_summary(search_result, high, moderate, low)

        # 높은 신뢰도 이슈 상세
        if high:
            report += "\n## 🟢 높은 신뢰도 이슈\n\n"
            report += self._create_detailed_issues_section(high, include_all=True)

        # 중간 신뢰도 이슈 상세
        if moderate:
            report += "\n## 🟡 중간 신뢰도 이슈\n\n"
            report += self._create_detailed_issues_section(moderate, include_all=False)

        # 낮은 신뢰도 이슈 (옵션에 따라)
        if low and self.threshold_manager.thresholds.include_low_confidence:
            report += "\n## 🔴 낮은 신뢰도 이슈 (참고용)\n\n"
            report += self._create_low_confidence_summary(low)

        # 환각 탐지 분석 요약
        report += self._create_hallucination_analysis_summary(search_result)

        # 보고서 메타데이터
        report += self._create_report_footer(search_result)

        return report

    def save_report_to_file(self, report_content: str, topic: str) -> str:
        """
        보고서를 마크다운 파일로 저장합니다.

        Args:
            report_content: 저장할 보고서 내용
            topic: 보고서 주제

        Returns:
            str: 저장된 파일 경로
        """
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)

        # 신뢰도 정보를 파일명에 포함
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"report_{safe_topic}_{timestamp}_validated.md"

        file_path = os.path.join(reports_dir, filename)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"마크다운 보고서 저장 완료: {file_path}")
        return file_path

    def _calculate_average_confidence(self, issues: List[IssueItem]) -> float:
        """이슈들의 평균 신뢰도를 계산합니다."""
        if not issues:
            return 0.0

        confidences = []
        for issue in issues:
            if hasattr(issue, 'combined_confidence') and issue.combined_confidence is not None:
                confidences.append(issue.combined_confidence)
            elif hasattr(issue, 'hallucination_score') and hasattr(issue.hallucination_score, 'combined_confidence'):
                confidences.append(issue.hallucination_score.combined_confidence)

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _create_summary_description(self, high: int, moderate: int, low: int) -> str:
        """임베드 설명 텍스트를 생성합니다."""
        total = high + moderate + low
        if total == 0:
            return "검색된 이슈가 없습니다."

        desc = f"총 **{total}개**의 이슈가 발견되었습니다.\n\n"
        desc += f"🟢 높은 신뢰도: **{high}개**\n"
        desc += f"🟡 중간 신뢰도: **{moderate}개**\n"
        desc += f"🔴 낮은 신뢰도: **{low}개**"

        return desc

    def _format_confidence_field(self, confidence_summary: Dict[str, Any]) -> str:
        """신뢰도 필드 텍스트를 포맷팅합니다."""
        # emoji 키가 없으므로 level을 사용하여 emoji를 가져옴
        level = confidence_summary.get('level')
        emoji = self._get_confidence_emoji(level) if level else "⚪"

        return (
            f"{emoji} **{confidence_summary['level_text']}** "
            f"({confidence_summary['score']:.1%})\n"
            f"{confidence_summary['recommendation']}"
        )

    def _get_confidence_emoji(self, level) -> str:
        """신뢰도 레벨에 따른 이모지를 반환합니다."""
        from src.hallucination_detection.threshold_manager import ConfidenceLevel

        emojis = {
            ConfidenceLevel.VERY_HIGH: "🟢",
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MODERATE: "🟡",
            ConfidenceLevel.LOW: "🟠",
            ConfidenceLevel.VERY_LOW: "🔴"
        }
        return emojis.get(level, "⚪")

    def _format_issues_for_embed(self, issues: List[IssueItem]) -> str:
        """임베드용 이슈 목록을 포맷팅합니다."""
        formatted_issues = []

        for issue in issues:
            confidence = getattr(issue, 'combined_confidence', 0.5)
            formatted_issues.append(
                f"• **{issue.title}**\n"
                f"  신뢰도: {confidence:.1%} | 출처: {issue.source}"
            )

        return "\n\n".join(formatted_issues)

    def _create_report_header(self, search_result: SearchResult) -> str:
        """보고서 헤더를 생성합니다."""
        return f"""# 🔍 AI 이슈 모니터링 종합 보고서

**생성일시**: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}  
**검색 키워드**: {', '.join(search_result.query_keywords)}  
**검색 기간**: {search_result.time_period}  
**총 이슈 수**: {search_result.total_found}개

---

"""

    def _create_executive_summary(
            self,
            search_result: SearchResult,
            high: List[IssueItem],
            moderate: List[IssueItem],
            low: List[IssueItem]
    ) -> str:
        """경영진 요약(Executive Summary)을 생성합니다."""
        summary = "## 📋 핵심 요약\n\n"

        # 신뢰도 분포
        total = len(search_result.issues)
        if total > 0:
            summary += f"### 신뢰도 분포\n"
            summary += f"- **높은 신뢰도**: {len(high)}개 ({len(high)/total*100:.1f}%)\n"
            summary += f"- **중간 신뢰도**: {len(moderate)}개 ({len(moderate)/total*100:.1f}%)\n"
            summary += f"- **낮은 신뢰도**: {len(low)}개 ({len(low)/total*100:.1f}%)\n\n"

        # 주요 발견사항
        if high:
            summary += "### 주요 발견사항\n"
            for i, issue in enumerate(high[:3], 1):
                summary += f"{i}. **{issue.title}**\n"
                summary += f"   - {issue.summary[:100]}...\n"
            summary += "\n"

        return summary

    def _create_detailed_issues_section(
            self,
            issues: List[IssueItem],
            include_all: bool = False
    ) -> str:
        """상세 이슈 섹션을 생성합니다."""
        section = ""
        issues_to_show = issues if include_all else issues[:5]

        for issue in issues_to_show:
            section += self._format_detailed_issue(issue)
            section += "\n---\n\n"

        if not include_all and len(issues) > 5:
            section += f"*... 외 {len(issues) - 5}개의 이슈가 더 있습니다.*\n\n"

        return section

    def _format_detailed_issue(self, issue: IssueItem) -> str:
        """개별 이슈의 상세 정보를 포맷팅합니다."""
        # 신뢰도 정보 추출
        confidence = getattr(issue, 'combined_confidence', 0.5)
        hallucination_score = getattr(issue, 'hallucination_score', None)

        formatted = f"### {issue.title}\n\n"
        formatted += f"**출처**: {issue.source} | **발행일**: {issue.published_date or 'N/A'}\n"
        formatted += f"**종합 신뢰도**: {confidence:.1%}\n\n"

        # 개별 탐지기 점수 (있는 경우)
        if hallucination_score and isinstance(hallucination_score, CombinedHallucinationScore):
            formatted += "**세부 신뢰도 점수**:\n"
            if hallucination_score.reppl_score:
                formatted += f"- RePPL: {hallucination_score.reppl_score.confidence:.1%}\n"
            if hallucination_score.consistency_score:
                formatted += f"- 자기 일관성: {hallucination_score.consistency_score.confidence:.1%}\n"
            if hallucination_score.llm_judge_score:
                formatted += f"- LLM Judge: {hallucination_score.llm_judge_score.confidence:.1%}\n"
            formatted += "\n"

        # 요약
        formatted += f"**요약**: {issue.summary}\n\n"

        # 상세 내용 (있는 경우)
        if issue.detailed_content:
            formatted += "**상세 내용**:\n"
            formatted += f"{issue.detailed_content[:500]}...\n\n"

        # 배경 정보 (있는 경우)
        if issue.background_context:
            formatted += "**배경 정보**:\n"
            formatted += f"{issue.background_context}\n\n"

        return formatted

    def _create_low_confidence_summary(self, issues: List[IssueItem]) -> str:
        """낮은 신뢰도 이슈 요약을 생성합니다."""
        summary = f"다음 {len(issues)}개의 이슈는 신뢰도가 낮아 참고용으로만 제공됩니다:\n\n"

        for issue in issues[:10]:  # 최대 10개만 표시
            confidence = getattr(issue, 'combined_confidence', 0.3)
            summary += f"- {issue.title} (신뢰도: {confidence:.1%})\n"

        if len(issues) > 10:
            summary += f"\n*... 외 {len(issues) - 10}개의 낮은 신뢰도 이슈*\n"

        return summary + "\n"

    def _create_hallucination_analysis_summary(self, search_result: SearchResult) -> str:
        """환각 탐지 분석 요약을 생성합니다."""
        summary = "\n## 🛡️ 환각 탐지 분석\n\n"

        summary += "### 탐지 시스템\n"
        summary += "본 보고서의 모든 이슈는 3단계 환각 탐지 시스템을 통해 검증되었습니다:\n\n"
        summary += "1. **RePPL (Relevant Paraphrased Prompt with Logit)**\n"
        summary += "   - 의미적 일관성과 관련성을 평가\n"
        summary += "2. **자기 일관성 검사 (Self-Consistency Check)**\n"
        summary += "   - 동일 질의에 대한 응답의 일관성 검증\n"
        summary += "3. **LLM-as-Judge**\n"
        summary += "   - 별도 LLM을 통한 교차 검증\n\n"

        # 재시도 정보 (있는 경우)
        if hasattr(search_result, 'retry_count') and search_result.retry_count > 0:
            summary += f"### 품질 보증\n"
            summary += f"신뢰도 향상을 위해 **{search_result.retry_count}회**의 추가 검색이 수행되었습니다.\n\n"

        return summary

    def _create_report_footer(self, search_result: SearchResult) -> str:
        """보고서 푸터를 생성합니다."""
        footer = "\n---\n\n"
        footer += "## 📌 메타데이터\n\n"
        footer += f"- **검색 소요 시간**: {search_result.search_time:.1f}초\n"
        footer += f"- **API 호출 횟수**: {search_result.api_calls_used}회\n"
        footer += f"- **검색 기간**: {search_result.time_period}\n"

        if hasattr(search_result, 'detailed_issues_count'):
            footer += f"- **상세 분석 이슈**: {search_result.detailed_issues_count}개\n"

        footer += f"\n*이 보고서는 AI 환각 탐지 시스템에 의해 검증되었습니다.*\n"

        return footer

    def _create_hallucination_analysis_summary(self, search_result: SearchResult) -> str:
        """환각 탐지 분석 요약을 생성합니다."""
        summary = "\n## 🛡️ 환각 탐지 분석\n\n"

        summary += "### 탐지 시스템\n"
        summary += "본 보고서의 모든 이슈는 3단계 환각 탐지 시스템을 통해 검증되었습니다:\n\n"
        summary += "1. **RePPL (Relevant Paraphrased Prompt with Logit)**\n"
        summary += "   - 의미적 일관성과 관련성을 평가\n"
        summary += "2. **자기 일관성 검사 (Self-Consistency Check)**\n"
        summary += "   - 동일 질의에 대한 응답의 일관성 검증\n"
        summary += "3. **LLM-as-Judge**\n"
        summary += "   - 별도 LLM을 통한 교차 검증\n\n"

        # 재시도 정보 (있는 경우)
        if hasattr(search_result, 'retry_count') and search_result.retry_count > 0:
            summary += f"### 품질 보증\n"
            summary += f"신뢰도 향상을 위해 **{search_result.retry_count}회**의 추가 검색이 수행되었습니다.\n\n"

        return summary

    def _create_report_footer(self, search_result: SearchResult) -> str:
        """보고서 푸터를 생성합니다."""
        footer = "\n---\n\n"
        footer += "## 📌 메타데이터\n\n"
        footer += f"- **검색 소요 시간**: {search_result.search_time:.1f}초\n"
        footer += f"- **API 호출 횟수**: {search_result.api_calls_used}회\n"
        footer += f"- **검색 기간**: {search_result.time_period}\n"

        if hasattr(search_result, 'detailed_issues_count'):
            footer += f"- **상세 분석 이슈**: {search_result.detailed_issues_count}개\n"

        footer += f"\n*이 보고서는 AI 환각 탐지 시스템에 의해 검증되었습니다.*\n"

        return footer

    def _calculate_average_confidence(self, issues: List[IssueItem]) -> float:
        """이슈들의 평균 신뢰도를 계산합니다."""
        if not issues:
            return 0.0

        confidences = []
        for issue in issues:
            if hasattr(issue, 'combined_confidence') and issue.combined_confidence is not None:
                confidences.append(issue.combined_confidence)
            elif hasattr(issue, 'hallucination_score') and hasattr(issue.hallucination_score, 'combined_confidence'):
                confidences.append(issue.hallucination_score.combined_confidence)

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _create_summary_description(self, high: int, moderate: int, low: int) -> str:
        """임베드 설명 텍스트를 생성합니다."""
        total = high + moderate + low
        if total == 0:
            return "검색된 이슈가 없습니다."

        desc = f"총 **{total}개**의 이슈가 발견되었습니다.\n\n"
        desc += f"🟢 높은 신뢰도: **{high}개**\n"
        desc += f"🟡 중간 신뢰도: **{moderate}개**\n"
        desc += f"🔴 낮은 신뢰도: **{low}개**"

        return desc

    def _format_issues_for_embed(self, issues: List[IssueItem]) -> str:
        """임베드용 이슈 목록을 포맷팅합니다."""
        formatted_issues = []

        for issue in issues:
            confidence = getattr(issue, 'combined_confidence', 0.5)
            formatted_issues.append(
                f"• **{issue.title}**\n"
                f"  신뢰도: {confidence:.1%} | 출처: {issue.source}"
            )

        return "\n\n".join(formatted_issues)


# 기존 reporting.py와의 호환성을 위한 래퍼 함수들
def format_search_summary_enhanced(result: SearchResult) -> str:
    """기존 format_search_summary의 향상된 버전."""
    generator = EnhancedReportGenerator()
    embed = generator.generate_discord_embed(result)
    return embed.description + "\n\n" + embed.fields[0].value


def create_detailed_report_enhanced(search_result: SearchResult) -> str:
    """기존 create_detailed_report의 향상된 버전."""
    generator = EnhancedReportGenerator()
    return generator.generate_detailed_report(search_result)


def save_report_to_file_enhanced(report_content: str, topic: str) -> str:
    """기존 save_report_to_file의 향상된 버전."""
    generator = EnhancedReportGenerator()
    return generator.save_report_to_file(report_content, topic)


async def generate_all_reports(
    search_result: SearchResult,
    topic: str,
    generate_pdf: bool = True
) -> Tuple[discord.Embed, str, Optional[str]]:
    """
    모든 형식의 보고서를 생성하는 통합 함수.

    Args:
        search_result: 검색 결과
        topic: 보고서 주제
        generate_pdf: PDF 생성 여부

    Returns:
        Tuple[discord.Embed, str, Optional[str]]:
            Discord 임베드, 마크다운 파일 경로, PDF 파일 경로
    """
    generator = EnhancedReportGenerator()
    return await generator.generate_reports(search_result, topic, generate_pdf)