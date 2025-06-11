"""
Discord-Specific Adaptive Report Formatter

Formats adaptive reports for Discord delivery with embeds and file attachments.
Handles Discord's character limits and formatting constraints while maintaining
the adaptive nature of the reports.
"""

import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import discord
from discord import Embed, File
from loguru import logger

from src.reporting.adaptive_report_generator import (
    AdaptiveReportStructure, 
    TopicCategory, 
    AudienceType,
    ContentComplexity
)
from src.models import SearchResult


class DiscordEmbedBuilder:
    """Build Discord embeds with adaptive formatting."""
    
    def __init__(self):
        self.max_embed_length = 4096
        self.max_field_value = 1024
        self.max_fields = 25
    
    def build_adaptive_embed(
        self, 
        adaptive_report: AdaptiveReportStructure,
        search_result: SearchResult
    ) -> Embed:
        """Build main Discord embed from adaptive report."""
        classification = adaptive_report.metadata.get('classification')
        
        # Determine embed color based on category
        color = self._get_embed_color(classification.primary_category if classification else TopicCategory.GENERAL)
        
        # Create embed with adaptive title
        embed = Embed(
            title=adaptive_report.title[:256],  # Discord title limit
            color=color,
            timestamp=datetime.now()
        )
        
        # Add executive summary
        summary = self._truncate_text(adaptive_report.executive_summary, 2048)
        if summary:
            embed.description = summary
        
        # Add key metrics
        self._add_metrics_field(embed, adaptive_report, search_result)
        
        # Add priority sections as fields
        priority_sections = sorted(adaptive_report.sections, key=lambda x: x.priority)[:8]  # Limit sections
        
        for section in priority_sections:
            field_value = self._format_section_for_embed(section, classification)
            if field_value:
                embed.add_field(
                    name=section.title[:256],
                    value=field_value[:self.max_field_value],
                    inline=len(field_value) < 300  # Short content inline
                )
        
        # Add recommendations if space allows
        if len(embed.fields) < self.max_fields - 1:
            self._add_recommendations_field(embed, adaptive_report)
        
        # Add confidence indicator
        self._add_confidence_field(embed, adaptive_report)
        
        # Set footer with metadata
        footer_text = self._generate_footer_text(classification, search_result)
        embed.set_footer(text=footer_text[:2048])
        
        return embed
    
    def _get_embed_color(self, category: TopicCategory) -> int:
        """Get Discord embed color for topic category."""
        colors = {
            TopicCategory.TECHNOLOGY: 0x007ACC,      # Blue
            TopicCategory.BUSINESS: 0x28A745,       # Green
            TopicCategory.SCIENTIFIC: 0x6F42C1,     # Purple
            TopicCategory.SOCIAL_POLITICAL: 0xFD7E14,  # Orange
            TopicCategory.HEALTHCARE: 0xDC3545,     # Red
            TopicCategory.FINANCE: 0xFFD700,        # Gold
            TopicCategory.ENVIRONMENT: 0x198754,    # Forest Green
            TopicCategory.ENTERTAINMENT: 0xE83E8C,  # Pink
            TopicCategory.GENERAL: 0x6C757D         # Gray
        }
        return colors.get(category, 0x6C757D)
    
    def _add_metrics_field(
        self, 
        embed: Embed, 
        adaptive_report: AdaptiveReportStructure,
        search_result: SearchResult
    ) -> None:
        """Add key metrics field."""
        classification = adaptive_report.metadata.get('classification')
        confidence_range = adaptive_report.metadata.get('confidence_range', {})
        
        metrics = []
        metrics.append(f"📊 **이슈 수**: {search_result.total_found}개")
        
        if confidence_range.get('avg'):
            avg_conf = confidence_range['avg']
            conf_emoji = "🟢" if avg_conf > 0.8 else "🟡" if avg_conf > 0.6 else "🔴"
            metrics.append(f"{conf_emoji} **평균 신뢰도**: {avg_conf:.1%}")
        
        if classification:
            urgency_emoji = "🚨" if classification.time_sensitivity > 0.8 else "⚠️" if classification.time_sensitivity > 0.6 else "ℹ️"
            metrics.append(f"{urgency_emoji} **시급성**: {self._format_urgency(classification.time_sensitivity)}")
        
        if metrics:
            embed.add_field(
                name="📈 핵심 지표",
                value="\n".join(metrics),
                inline=True
            )
    
    def _format_section_for_embed(
        self, 
        section,
        classification
    ) -> str:
        """Format section content for Discord embed field."""
        content = section.content
        
        # Adaptive formatting based on audience and complexity
        if classification and classification.audience_type == AudienceType.EXECUTIVE:
            # Executive summary style - bullet points
            content = self._format_as_bullets(content)
        elif classification and classification.complexity_level == ContentComplexity.BASIC:
            # Simplified formatting
            content = self._simplify_content(content)
        
        # Add visualization indicators
        if section.visualization_type:
            viz_emoji = {
                'chart': '📊',
                'table': '📋',
                'timeline': '📅',
                'network': '🔗'
            }.get(section.visualization_type, '📈')
            content = f"{viz_emoji} {content}"
        
        return self._truncate_text(content, self.max_field_value - 50)
    
    def _add_recommendations_field(self, embed: Embed, adaptive_report: AdaptiveReportStructure) -> None:
        """Add recommendations field if space permits."""
        if not adaptive_report.recommendations:
            return
        
        # Format top 3 recommendations
        top_recommendations = adaptive_report.recommendations[:3]
        rec_text = "\n".join([f"• {rec}" for rec in top_recommendations])
        
        if len(rec_text) <= self.max_field_value:
            embed.add_field(
                name="💡 권장사항",
                value=rec_text,
                inline=False
            )
    
    def _add_confidence_field(self, embed: Embed, adaptive_report: AdaptiveReportStructure) -> None:
        """Add confidence indicator field."""
        confidence_range = adaptive_report.metadata.get('confidence_range', {})
        
        if not confidence_range:
            return
        
        avg_conf = confidence_range.get('avg', 0)
        min_conf = confidence_range.get('min', 0)
        max_conf = confidence_range.get('max', 0)
        
        # Create confidence visualization
        conf_text = f"평균: {avg_conf:.1%} (범위: {min_conf:.1%} - {max_conf:.1%})"
        
        # Add confidence bar visualization
        bar_length = 10
        filled = int(avg_conf * bar_length)
        conf_bar = "█" * filled + "░" * (bar_length - filled)
        
        embed.add_field(
            name="🎯 신뢰도",
            value=f"{conf_bar}\n{conf_text}",
            inline=True
        )
    
    def _generate_footer_text(self, classification, search_result: SearchResult) -> str:
        """Generate footer text with metadata."""
        parts = []
        
        if classification:
            parts.append(f"카테고리: {classification.primary_category.value}")
            parts.append(f"대상: {classification.audience_type.value}")
        
        parts.append(f"키워드: {', '.join(search_result.keywords[:3])}")
        parts.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        return " | ".join(parts)
    
    # Utility methods
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to fit Discord limits."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def _format_urgency(self, time_sensitivity: float) -> str:
        """Format time sensitivity as urgency level."""
        if time_sensitivity > 0.8:
            return "매우 높음"
        elif time_sensitivity > 0.6:
            return "높음"
        elif time_sensitivity > 0.4:
            return "보통"
        else:
            return "낮음"
    
    def _format_as_bullets(self, content: str) -> str:
        """Format content as bullet points for executives."""
        lines = content.split('\n')
        bullets = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('•') and not line.startswith('-'):
                if len(line) > 100:  # Break long lines
                    line = line[:97] + "..."
                bullets.append(f"• {line}")
            elif line:
                bullets.append(line)
        return '\n'.join(bullets[:5])  # Limit bullets
    
    def _simplify_content(self, content: str) -> str:
        """Simplify content for basic complexity level."""
        # Remove technical jargon and simplify language
        # This is a simplified implementation
        simplified = content.replace('구현', '실행').replace('프레임워크', '시스템')
        return simplified


class DiscordFileGenerator:
    """Generate file attachments for Discord."""
    
    def __init__(self):
        self.max_file_size = 25 * 1024 * 1024  # 25MB Discord limit
    
    def generate_detailed_report_file(
        self, 
        adaptive_report: AdaptiveReportStructure,
        search_result: SearchResult
    ) -> Optional[File]:
        """Generate detailed markdown report file."""
        try:
            # Generate full markdown content
            markdown_content = self._generate_full_markdown(adaptive_report, search_result)
            
            # Create filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_safe = "".join(c for c in search_result.topic if c.isalnum() or c in "._-")[:30]
            filename = f"report_{topic_safe}_{timestamp}_adaptive.md"
            
            # Save to reports directory
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            file_path = os.path.join(reports_dir, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            # Check file size
            if os.path.getsize(file_path) > self.max_file_size:
                logger.warning(f"Report file too large: {os.path.getsize(file_path)} bytes")
                return None
            
            # Create Discord file object
            return File(file_path, filename=filename)
            
        except Exception as e:
            logger.error(f"Failed to generate report file: {e}")
            return None
    
    def _generate_full_markdown(
        self, 
        adaptive_report: AdaptiveReportStructure,
        search_result: SearchResult
    ) -> str:
        """Generate complete markdown report."""
        classification = adaptive_report.metadata.get('classification')
        
        # Header
        content = f"# {adaptive_report.title}\n\n"
        
        # Metadata section
        content += self._generate_metadata_section(adaptive_report, search_result, classification)
        
        # Executive Summary
        content += f"## 📋 개요\n\n{adaptive_report.executive_summary}\n\n"
        
        # All sections
        for section in sorted(adaptive_report.sections, key=lambda x: x.priority):
            content += f"## {section.title}\n\n"
            content += f"{section.content}\n\n"
            
            # Add visualization placeholder if specified
            if section.visualization_type:
                content += f"*[{section.visualization_type} 시각화 영역]*\n\n"
        
        # Recommendations
        if adaptive_report.recommendations:
            content += "## 💡 권장사항\n\n"
            for i, rec in enumerate(adaptive_report.recommendations, 1):
                content += f"{i}. {rec}\n"
            content += "\n"
        
        # Detailed issues
        content += self._generate_detailed_issues_section(search_result)
        
        # Confidence and verification
        content += self._generate_verification_appendix(search_result)
        
        # Footer
        content += self._generate_footer(classification)
        
        return content
    
    def _generate_metadata_section(
        self, 
        adaptive_report: AdaptiveReportStructure,
        search_result: SearchResult,
        classification
    ) -> str:
        """Generate metadata section."""
        content = "## 📊 보고서 정보\n\n"
        
        content += f"- **주제**: {search_result.topic}\n"
        content += f"- **검색 키워드**: {', '.join(search_result.keywords)}\n"
        content += f"- **분석 기간**: {search_result.period}\n"
        content += f"- **총 이슈 수**: {search_result.total_found}개\n"
        
        if classification:
            content += f"- **주제 분류**: {classification.primary_category.value}\n"
            content += f"- **대상 독자**: {classification.audience_type.value}\n"
            content += f"- **복잡도**: {classification.complexity_level.value}\n"
            content += f"- **시급성**: {self._format_urgency_detailed(classification.time_sensitivity)}\n"
        
        confidence_range = adaptive_report.metadata.get('confidence_range', {})
        if confidence_range:
            content += f"- **평균 신뢰도**: {confidence_range.get('avg', 0):.1%}\n"
        
        content += f"- **생성 시간**: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}\n\n"
        
        return content
    
    def _generate_detailed_issues_section(self, search_result: SearchResult) -> str:
        """Generate detailed issues section."""
        content = "## 📑 상세 이슈 목록\n\n"
        
        for i, issue in enumerate(search_result.issues, 1):
            content += f"### {i}. {issue.title}\n\n"
            content += f"**출처**: {issue.source}\n"
            content += f"**발행일**: {issue.published_date or 'N/A'}\n"
            content += f"**관련도**: {issue.relevance_score:.1%}\n"
            
            # Add confidence scores if available
            combined_conf = getattr(issue, 'combined_confidence', None)
            if combined_conf:
                content += f"**신뢰도**: {combined_conf:.1%}\n"
            
            content += f"\n**요약**: {issue.summary}\n\n"
            
            if issue.detailed_content:
                content += f"**상세 내용**:\n{issue.detailed_content}\n\n"
            
            if issue.background_context:
                content += f"**배경 정보**:\n{issue.background_context}\n\n"
            
            content += "---\n\n"
        
        return content
    
    def _generate_verification_appendix(self, search_result: SearchResult) -> str:
        """Generate verification and confidence appendix."""
        content = "## 🔍 신뢰도 검증\n\n"
        
        content += "### 검증 방법론\n\n"
        content += "본 보고서는 다음의 3단계 환각 탐지 시스템을 통해 검증되었습니다:\n\n"
        content += "1. **LLM-as-a-Judge**: GPT-4o를 활용한 사실 정확성 및 논리적 일관성 검사\n"
        content += "2. **RePPL 분석**: 텍스트 반복성, 퍼플렉시티, 의미적 엔트로피 분석\n"
        content += "3. **자기 일관성 검사**: 다중 프롬프트를 통한 응답 일관성 검증\n\n"
        
        content += "### 이슈별 신뢰도 상세\n\n"
        content += "| 순번 | 이슈 제목 | 관련도 | 신뢰도 | 상태 |\n"
        content += "|------|-----------|--------|--------|------|\n"
        
        for i, issue in enumerate(search_result.issues[:10], 1):
            title = issue.title[:50] + "..." if len(issue.title) > 50 else issue.title
            relevance = f"{issue.relevance_score:.1%}"
            confidence = getattr(issue, 'combined_confidence', 0.5)
            conf_text = f"{confidence:.1%}"
            status = "✅ 높음" if confidence > 0.8 else "⚠️ 보통" if confidence > 0.6 else "❌ 낮음"
            
            content += f"| {i} | {title} | {relevance} | {conf_text} | {status} |\n"
        
        content += "\n"
        return content
    
    def _generate_footer(self, classification) -> str:
        """Generate report footer."""
        content = "---\n\n"
        content += "## 📝 보고서 생성 정보\n\n"
        content += "이 보고서는 AI 기반 실시간 이슈 모니터링 시스템에 의해 자동 생성되었습니다.\n\n"
        
        if classification:
            content += f"- **분류 신뢰도**: {classification.confidence:.1%}\n"
            content += f"- **보조 카테고리**: {', '.join([cat.value for cat in classification.secondary_categories])}\n" if classification.secondary_categories else ""
        
        content += "- **데이터 소스**: Perplexity API (llama-3.1-sonar-large-128k-online)\n"
        content += "- **분석 엔진**: OpenAI GPT-4o\n"
        content += "- **환각 탐지**: 3단계 검증 시스템\n\n"
        content += "*이 보고서의 내용은 AI에 의해 생성되었으며, 중요한 의사결정 시 추가 검증이 필요할 수 있습니다.*\n"
        
        return content
    
    def _format_urgency_detailed(self, time_sensitivity: float) -> str:
        """Format time sensitivity with detailed description."""
        if time_sensitivity > 0.8:
            return "매우 높음 (즉시 대응 필요)"
        elif time_sensitivity > 0.6:
            return "높음 (24시간 내 검토 권장)"
        elif time_sensitivity > 0.4:
            return "보통 (1주일 내 검토)"
        else:
            return "낮음 (정기 모니터링)"


class AdaptiveDiscordFormatter:
    """Main Discord formatter orchestrating embed and file generation."""
    
    def __init__(self):
        self.embed_builder = DiscordEmbedBuilder()
        self.file_generator = DiscordFileGenerator()
    
    async def format_for_discord(
        self, 
        adaptive_report: AdaptiveReportStructure,
        search_result: SearchResult,
        include_file: bool = True
    ) -> Tuple[Embed, Optional[File]]:
        """Format adaptive report for Discord delivery."""
        try:
            # Generate embed
            embed = self.embed_builder.build_adaptive_embed(adaptive_report, search_result)
            
            # Generate file attachment
            file_attachment = None
            if include_file:
                file_attachment = self.file_generator.generate_detailed_report_file(
                    adaptive_report, 
                    search_result
                )
            
            logger.info(f"Discord formatting completed. Embed fields: {len(embed.fields)}, "
                       f"File: {'Yes' if file_attachment else 'No'}")
            
            return embed, file_attachment
            
        except Exception as e:
            logger.error(f"Discord formatting failed: {e}")
            # Return fallback embed
            fallback_embed = Embed(
                title=f"📊 {search_result.topic} 분석 결과",
                description=f"{search_result.total_found}개의 이슈가 발견되었습니다.",
                color=0x6C757D
            )
            return fallback_embed, None
    
    def create_summary_embed(
        self, 
        adaptive_report: AdaptiveReportStructure,
        search_result: SearchResult
    ) -> Embed:
        """Create a summary-only embed for quick preview."""
        classification = adaptive_report.metadata.get('classification')
        color = self.embed_builder._get_embed_color(
            classification.primary_category if classification else TopicCategory.GENERAL
        )
        
        embed = Embed(
            title=adaptive_report.title[:256],
            description=self.embed_builder._truncate_text(adaptive_report.executive_summary, 2048),
            color=color,
            timestamp=datetime.now()
        )
        
        # Add only key metrics
        self.embed_builder._add_metrics_field(embed, adaptive_report, search_result)
        
        # Add top recommendation
        if adaptive_report.recommendations:
            embed.add_field(
                name="💡 핵심 권장사항",
                value=adaptive_report.recommendations[0][:1024],
                inline=False
            )
        
        return embed