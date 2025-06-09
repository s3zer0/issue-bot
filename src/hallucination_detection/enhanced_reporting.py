"""
환각 탐지 결과가 통합된 향상된 보고서 생성 모듈.

신뢰도 등급별로 이슈를 분류하고, 환각 탐지 결과를 명확히 표시하여
사용자가 정보의 신뢰성을 쉽게 판단할 수 있도록 합니다.
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import discord
from loguru import logger

from src.models import SearchResult, IssueItem
from src.hallucination_detection.models import CombinedHallucinationScore
from src.hallucination_detection.threshold_manager import (
    ThresholdManager, ConfidenceLevel
)


class EnhancedReportGenerator:
    """환각 탐지 결과가 통합된 보고서 생성기."""

    def __init__(self, threshold_manager: Optional[ThresholdManager] = None):
        """
        보고서 생성기를 초기화합니다.

        Args:
            threshold_manager: 임계값 관리자 (없으면 기본값 사용)
        """
        self.threshold_manager = threshold_manager or ThresholdManager()
        logger.info("향상된 보고서 생성기 초기화 완료")

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
        avg_confidence = self._calculate_average_confidence(search_result.issues)
        confidence_summary = self.threshold_manager.get_confidence_summary(avg_confidence)

        summary = f"""## 📋 핵심 요약 (Executive Summary)

### 신뢰도 평가
- **전체 신뢰도**: {confidence_summary['score']:.1%} ({confidence_summary['level_text']})
- **평가 결과**: {confidence_summary['recommendation']}

### 이슈 분포
- ✅ **높은 신뢰도**: {len(high)}개 ({len(high) / len(search_result.issues) * 100:.1f}%)
- ⚠️ **중간 신뢰도**: {len(moderate)}개 ({len(moderate) / len(search_result.issues) * 100:.1f}%)
- ❌ **낮은 신뢰도**: {len(low)}개 ({len(low) / len(search_result.issues) * 100:.1f}%)

### 주요 발견사항
"""

        # 상위 3개 높은 신뢰도 이슈 요약
        for i, issue in enumerate(high[:3], 1):
            confidence = getattr(issue, 'hallucination_confidence', 0.0)
            summary += f"{i}. **{issue.title}** (신뢰도: {confidence:.1%})\n"
            summary += f"   - {issue.summary[:100]}...\n\n"

        summary += "\n---\n"
        return summary

    def _create_detailed_issues_section(
            self,
            issues: List[IssueItem],
            include_all: bool = True
    ) -> str:
        """이슈 상세 섹션을 생성합니다."""
        section = ""

        # 포함할 이슈 수 결정
        issues_to_include = issues if include_all else issues[:5]

        for i, issue in enumerate(issues_to_include, 1):
            section += self._format_single_issue(issue, i)

        if not include_all and len(issues) > 5:
            section += f"\n*... 외 {len(issues) - 5}개 이슈*\n"

        return section

    def _format_single_issue(self, issue: IssueItem, index: int) -> str:
        """단일 이슈를 포맷팅합니다."""
        confidence = getattr(issue, 'hallucination_confidence', 0.0)
        hallucination_analysis = getattr(issue, 'hallucination_analysis', None)

        # 기본 정보
        formatted = f"""### {index}. {issue.title}

**신뢰도**: {self._create_confidence_badge(confidence)}  
**출처**: {issue.source} | **발행일**: {issue.published_date or 'N/A'}  
**카테고리**: {issue.category} | **관련도**: {issue.relevance_score:.1%}

#### 요약
{issue.summary}

"""

        # 상세 내용 (신뢰도가 임계값 이상인 경우)
        if (issue.detailed_content and
                self.threshold_manager.should_include_detailed_analysis(confidence)):
            formatted += f"""#### 상세 내용
{issue.detailed_content[:500]}...

"""

        # 환각 탐지 분석 결과
        if hallucination_analysis and isinstance(hallucination_analysis, CombinedHallucinationScore):
            formatted += self._format_hallucination_analysis(hallucination_analysis)

        formatted += "\n---\n"
        return formatted

    def _format_hallucination_analysis(self, analysis: CombinedHallucinationScore) -> str:
        """환각 탐지 분석 결과를 포맷팅합니다."""
        result = "#### 🔍 환각 탐지 상세 분석\n\n"

        # 개별 탐지기 결과
        for method, score in analysis.individual_scores.items():
            result += f"- **{method}**: {score.confidence:.1%}"

            # 각 탐지기별 특수 정보
            if hasattr(score, 'get_summary'):
                result += f" - {score.get_summary()}"

            result += "\n"

        # 문제점 요약 (LLM Judge의 경우)
        llm_judge = analysis.individual_scores.get('LLM-Judge')
        if llm_judge and hasattr(llm_judge, 'problematic_areas') and llm_judge.problematic_areas:
            result += "\n**발견된 문제점**:\n"
            for area in llm_judge.problematic_areas[:3]:
                result += f"- \"{area['text'][:50]}...\": {area['issue']}\n"

        result += "\n"
        return result

    def _create_low_confidence_summary(self, low_issues: List[IssueItem]) -> str:
        """낮은 신뢰도 이슈 요약을 생성합니다."""
        summary = f"총 {len(low_issues)}개의 이슈가 신뢰도 기준을 충족하지 못했습니다.\n\n"

        summary += "**제외된 이슈 목록**:\n"
        for issue in low_issues[:10]:
            confidence = getattr(issue, 'hallucination_confidence', 0.0)
            summary += f"- {issue.title} (신뢰도: {confidence:.1%})\n"

        if len(low_issues) > 10:
            summary += f"\n*... 외 {len(low_issues) - 10}개*\n"

        return summary

    def _create_hallucination_analysis_summary(self, search_result: SearchResult) -> str:
        """전체 환각 탐지 분석 요약을 생성합니다."""
        summary = "\n## 🛡️ 환각 탐지 시스템 분석 요약\n\n"

        # 탐지 방법별 통계
        method_stats = self._calculate_method_statistics(search_result.issues)

        summary += "### 탐지 방법별 평균 신뢰도\n"
        for method, avg_score in method_stats.items():
            summary += f"- **{method}**: {avg_score:.1%}\n"

        # 전체 시스템 성능
        summary += f"\n### 시스템 성능 지표\n"
        summary += f"- **평균 처리 시간**: {search_result.search_time:.1f}초\n"
        summary += f"- **API 호출 횟수**: {search_result.api_calls_used}회\n"
        summary += f"- **상세 분석 이슈**: {search_result.detailed_issues_count}개\n"

        return summary + "\n---\n"

    def _create_report_footer(self, search_result: SearchResult) -> str:
        """보고서 푸터를 생성합니다."""
        return f"""
## 📌 보고서 정보

- **생성 시스템**: AI 기반 이슈 모니터링 봇 v1.0
- **환각 탐지**: 3단계 교차 검증 (RePPL, Self-Consistency, LLM-as-Judge)
- **신뢰도 임계값**: {self.threshold_manager.thresholds.min_confidence_threshold:.1%}
- **보고서 ID**: {datetime.now().strftime('%Y%m%d_%H%M%S')}

---
*이 보고서는 AI 시스템에 의해 자동 생성되었습니다. 중요한 의사결정에는 추가 검증을 권장합니다.*
"""

    # 유틸리티 메서드들
    def _calculate_average_confidence(self, issues: List[IssueItem]) -> float:
        """이슈들의 평균 신뢰도를 계산합니다."""
        if not issues:
            return 0.0

        total_confidence = sum(
            getattr(issue, 'hallucination_confidence', 0.0)
            for issue in issues
        )
        return total_confidence / len(issues)

    def _create_summary_description(self, high: int, moderate: int, low: int) -> str:
        """임베드용 요약 설명을 생성합니다."""
        total = high + moderate + low
        if total == 0:
            return "검색 결과가 없습니다."

        return (
            f"총 {total}개의 이슈를 발견했습니다.\n"
            f"• 높은 신뢰도: {high}개\n"
            f"• 중간 신뢰도: {moderate}개\n"
            f"• 낮은 신뢰도: {low}개"
        )

    def _format_confidence_field(self, confidence_summary: Dict[str, Any]) -> str:
        """신뢰도 필드를 포맷팅합니다."""
        emoji = self._get_confidence_emoji(confidence_summary['level'])
        return (
            f"{emoji} **{confidence_summary['score']:.1%}** "
            f"({confidence_summary['level_text']})\n"
            f"{confidence_summary['recommendation']}"
        )

    def _format_issues_for_embed(self, issues: List[IssueItem]) -> str:
        """임베드용 이슈 목록을 포맷팅합니다."""
        formatted = ""
        for issue in issues:
            confidence = getattr(issue, 'hallucination_confidence', 0.0)
            formatted += f"• **{issue.title[:50]}{'...' if len(issue.title) > 50 else ''}**\n"
            formatted += f"  신뢰도: {confidence:.1%} | {issue.source}\n\n"
        return formatted.strip()

    def _create_confidence_badge(self, confidence: float) -> str:
        """신뢰도 배지를 생성합니다."""
        level = self.threshold_manager.classify_confidence(confidence)
        emoji = self._get_confidence_emoji(level)
        return f"{emoji} {confidence:.1%} ({level.value})"

    def _get_confidence_emoji(self, level: ConfidenceLevel) -> str:
        """신뢰도 레벨에 따른 이모지를 반환합니다."""
        emojis = {
            ConfidenceLevel.VERY_HIGH: "🟢",
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MODERATE: "🟡",
            ConfidenceLevel.LOW: "🟠",
            ConfidenceLevel.VERY_LOW: "🔴"
        }
        return emojis.get(level, "⚪")

    def _calculate_method_statistics(self, issues: List[IssueItem]) -> Dict[str, float]:
        """탐지 방법별 통계를 계산합니다."""
        method_scores = {
            'RePPL': [],
            'Self-Consistency': [],
            'LLM-Judge': []
        }

        for issue in issues:
            analysis = getattr(issue, 'hallucination_analysis', None)
            if analysis and isinstance(analysis, CombinedHallucinationScore):
                for method, score in analysis.individual_scores.items():
                    if method in method_scores:
                        method_scores[method].append(score.confidence)

        # 평균 계산
        return {
            method: sum(scores) / len(scores) if scores else 0.0
            for method, scores in method_scores.items()
        }

    def save_report_to_file(self, report_content: str, topic: str) -> str:
        """
        보고서를 파일로 저장합니다.

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

        logger.info(f"보고서 저장 완료: {file_path}")
        return file_path


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