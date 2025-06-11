"""
Adaptive Reporting Integration Module

Integrates the adaptive report generation system with the existing Discord bot
and reporting infrastructure. Provides a unified interface for generating
and delivering adaptive reports.
"""

import asyncio
from typing import Optional, Tuple, Union
from discord import Embed, File
from loguru import logger

from src.models import SearchResult
from src.reporting.adaptive_report_generator import (
    AdaptiveReportGenerator, 
    AdaptiveReportStructure,
    AudienceType,
    TopicCategory
)
from src.reporting.discord_formatter import AdaptiveDiscordFormatter
from src.reporting.pdf_report_generator import PDFReportGenerator
from src.reporting.reporting import (
    format_search_summary,
    create_detailed_report_from_search_result
)


class AdaptiveReportingOrchestrator:
    """Orchestrates adaptive report generation and delivery."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.adaptive_generator = AdaptiveReportGenerator(openai_api_key)
        self.discord_formatter = AdaptiveDiscordFormatter()
        self.pdf_generator = PDFReportGenerator(openai_api_key)
        
    async def generate_complete_report_package(
        self,
        search_result: SearchResult,
        target_audience: Optional[AudienceType] = None,
        include_pdf: bool = True,
        include_legacy: bool = False
    ) -> Tuple[Embed, Optional[File], Optional[File]]:
        """
        Generate complete adaptive report package for Discord delivery.
        
        Returns:
            Tuple[Embed, markdown_file, pdf_file]
        """
        try:
            logger.info(f"Generating adaptive report package for: {search_result.topic}")
            
            # Generate adaptive report structure
            adaptive_report = await self.adaptive_generator.generate_adaptive_report(
                search_result, 
                target_audience
            )
            
            # Format for Discord
            embed, markdown_file = await self.discord_formatter.format_for_discord(
                adaptive_report, 
                search_result,
                include_file=True
            )
            
            # Generate PDF if requested
            pdf_file = None
            if include_pdf:
                pdf_file = await self._generate_adaptive_pdf(adaptive_report, search_result)
            
            logger.info("Adaptive report package generation completed successfully")
            return embed, markdown_file, pdf_file
            
        except Exception as e:
            logger.error(f"Adaptive report package generation failed: {e}")
            # Fallback to legacy reporting
            return await self._generate_fallback_package(search_result, include_pdf)
    
    async def generate_quick_summary(
        self,
        search_result: SearchResult,
        target_audience: Optional[AudienceType] = None
    ) -> Embed:
        """Generate quick summary embed for immediate response."""
        try:
            # Use existing fast summary for immediate response
            summary_text = format_search_summary(search_result)
            
            # Try to get basic classification for color
            classification = await self.adaptive_generator.classifier.classify_topic(
                search_result.topic, 
                search_result.issues[:3]  # Quick classification with fewer issues
            )
            
            color = self.discord_formatter.embed_builder._get_embed_color(classification.primary_category)
            
            embed = Embed(
                title=f"🔍 {search_result.topic} - 분석 진행 중",
                description=summary_text,
                color=color
            )
            
            # Add quick metrics
            embed.add_field(
                name="📊 발견된 이슈",
                value=f"{search_result.total_found}개",
                inline=True
            )
            
            if classification.time_sensitivity > 0.7:
                embed.add_field(
                    name="⚡ 상태",
                    value="긴급 분석 중...",
                    inline=True
                )
            else:
                embed.add_field(
                    name="⏳ 상태", 
                    value="상세 분석 중...",
                    inline=True
                )
            
            embed.set_footer(text="상세 보고서가 곧 생성됩니다...")
            
            return embed
            
        except Exception as e:
            logger.error(f"Quick summary generation failed: {e}")
            # Ultra-fallback
            return Embed(
                title=f"📊 {search_result.topic}",
                description=format_search_summary(search_result),
                color=0x6C757D
            )
    
    async def _generate_adaptive_pdf(
        self,
        adaptive_report: AdaptiveReportStructure,
        search_result: SearchResult
    ) -> Optional[File]:
        """Generate PDF using adaptive report data."""
        try:
            # Convert adaptive report to format expected by PDF generator
            # This bridges the new adaptive system with existing PDF generation
            
            enhanced_content = {
                'executive_summary': adaptive_report.executive_summary,
                'key_findings': self._extract_key_findings(adaptive_report),
                'trend_analysis': self._extract_trend_analysis(adaptive_report),
                'risks_opportunities': self._extract_risks_opportunities(adaptive_report),
                'recommended_actions': adaptive_report.recommendations,
                'metadata': adaptive_report.metadata
            }
            
            # Generate PDF with enhanced content
            pdf_path = await self.pdf_generator.generate_report(
                search_result,
                enhanced_content=enhanced_content
            )
            
            if pdf_path:
                return File(pdf_path)
            
            return None
            
        except Exception as e:
            logger.error(f"Adaptive PDF generation failed: {e}")
            return None
    
    async def _generate_fallback_package(
        self,
        search_result: SearchResult,
        include_pdf: bool
    ) -> Tuple[Embed, Optional[File], Optional[File]]:
        """Generate fallback package using legacy system."""
        try:
            logger.info("Using fallback legacy reporting system")
            
            # Use existing summary generation
            summary_text = format_search_summary(search_result)
            
            embed = Embed(
                title=f"📊 {search_result.topic} 분석 결과",
                description=summary_text,
                color=0x6C757D
            )
            
            # Generate legacy markdown file
            markdown_content = create_detailed_report_from_search_result(search_result)
            
            # Save markdown file
            import os
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_safe = "".join(c for c in search_result.topic if c.isalnum() or c in "._-")[:30]
            
            reports_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            markdown_filename = f"report_{topic_safe}_{timestamp}_legacy.md"
            markdown_path = os.path.join(reports_dir, markdown_filename)
            
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            markdown_file = File(markdown_path, filename=markdown_filename)
            
            # Generate PDF if requested
            pdf_file = None
            if include_pdf:
                try:
                    pdf_path = await self.pdf_generator.generate_report(search_result)
                    if pdf_path:
                        pdf_file = File(pdf_path)
                except Exception as pdf_e:
                    logger.error(f"Legacy PDF generation failed: {pdf_e}")
            
            return embed, markdown_file, pdf_file
            
        except Exception as e:
            logger.error(f"Fallback package generation failed: {e}")
            # Ultimate fallback
            embed = Embed(
                title=f"📊 {search_result.topic}",
                description=f"{search_result.total_found}개의 이슈가 발견되었습니다.",
                color=0x6C757D
            )
            return embed, None, None
    
    # Utility methods for extracting content from adaptive report
    def _extract_key_findings(self, adaptive_report: AdaptiveReportStructure) -> list:
        """Extract key findings from adaptive report sections."""
        key_findings_section = next(
            (section for section in adaptive_report.sections if section.name == "key_findings"),
            None
        )
        
        if key_findings_section:
            # Parse bullet points from content
            content = key_findings_section.content
            findings = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    findings.append(line.lstrip('- •').strip())
            return findings[:5]
        
        return ["주요 발견사항을 분석 중입니다."]
    
    def _extract_trend_analysis(self, adaptive_report: AdaptiveReportStructure) -> str:
        """Extract trend analysis from adaptive report."""
        # Look for trend-related sections
        trend_sections = [
            section for section in adaptive_report.sections 
            if 'trend' in section.name.lower() or 'timeline' in section.name.lower()
        ]
        
        if trend_sections:
            return trend_sections[0].content
        
        return "트렌드 분석을 진행 중입니다."
    
    def _extract_risks_opportunities(self, adaptive_report: AdaptiveReportStructure) -> str:
        """Extract risks and opportunities analysis."""
        # Look for relevant sections
        risk_sections = [
            section for section in adaptive_report.sections
            if any(keyword in section.name.lower() for keyword in ['risk', 'opportunity', 'impact', 'financial'])
        ]
        
        if risk_sections:
            return risk_sections[0].content
        
        return "리스크 및 기회 분석을 진행 중입니다."


# Convenience functions for easy integration
async def generate_adaptive_discord_report(
    search_result: SearchResult,
    target_audience: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Tuple[Embed, Optional[File], Optional[File]]:
    """
    Convenience function for generating adaptive Discord reports.
    
    Args:
        search_result: Search result to generate report from
        target_audience: Target audience ("executive", "technical", "general_public", etc.)
        openai_api_key: OpenAI API key
        
    Returns:
        Tuple of (embed, markdown_file, pdf_file)
    """
    orchestrator = AdaptiveReportingOrchestrator(openai_api_key)
    
    # Convert string audience to enum
    audience = None
    if target_audience:
        try:
            audience = AudienceType(target_audience.lower())
        except ValueError:
            logger.warning(f"Invalid audience type: {target_audience}, using default")
    
    return await orchestrator.generate_complete_report_package(
        search_result,
        target_audience=audience,
        include_pdf=True
    )


async def generate_quick_adaptive_summary(
    search_result: SearchResult,
    target_audience: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> Embed:
    """
    Convenience function for generating quick adaptive summaries.
    
    Args:
        search_result: Search result to summarize
        target_audience: Target audience
        openai_api_key: OpenAI API key
        
    Returns:
        Discord Embed with quick summary
    """
    orchestrator = AdaptiveReportingOrchestrator(openai_api_key)
    
    audience = None
    if target_audience:
        try:
            audience = AudienceType(target_audience.lower())
        except ValueError:
            pass
    
    return await orchestrator.generate_quick_summary(
        search_result,
        target_audience=audience
    )


# Integration with existing bot command structure
class AdaptiveReportingMixin:
    """Mixin class to add adaptive reporting capabilities to existing bot."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_orchestrator = AdaptiveReportingOrchestrator()
    
    async def send_adaptive_report(
        self,
        ctx,
        search_result: SearchResult,
        target_audience: Optional[str] = None
    ):
        """Send adaptive report to Discord channel."""
        try:
            # Send quick summary first
            quick_embed = await self.adaptive_orchestrator.generate_quick_summary(
                search_result,
                target_audience=AudienceType(target_audience) if target_audience else None
            )
            
            quick_message = await ctx.send(embed=quick_embed)
            
            # Generate complete package
            embed, markdown_file, pdf_file = await self.adaptive_orchestrator.generate_complete_report_package(
                search_result,
                target_audience=AudienceType(target_audience) if target_audience else None
            )
            
            # Send files
            files = []
            if markdown_file:
                files.append(markdown_file)
            if pdf_file:
                files.append(pdf_file)
            
            if files:
                await ctx.send(embed=embed, files=files)
                # Delete the quick summary
                try:
                    await quick_message.delete()
                except:
                    pass
            else:
                # Edit the quick message with final embed
                await quick_message.edit(embed=embed)
                
        except Exception as e:
            logger.error(f"Adaptive report sending failed: {e}")
            # Fallback to basic embed
            fallback_embed = Embed(
                title=f"📊 {search_result.topic}",
                description=f"보고서 생성 중 오류가 발생했습니다. 기본 정보만 제공합니다.\n\n{format_search_summary(search_result)}",
                color=0xDC3545
            )
            await ctx.send(embed=fallback_embed)