"""
Example integration of adaptive reporting system.

This file demonstrates how to integrate the new adaptive reporting system
with the existing Discord bot infrastructure.
"""

import asyncio
from datetime import datetime
from typing import Optional

# Example of how to modify the bot command to use adaptive reporting
from src.reporting.adaptive_integration import (
    generate_adaptive_discord_report,
    generate_quick_adaptive_summary,
    AdaptiveReportingOrchestrator
)
from src.models import SearchResult, IssueItem


async def example_monitor_command_with_adaptive_reporting(
    ctx,
    topic: str,
    period: str = "1주일",
    audience: Optional[str] = None
):
    """
    Example of how to modify the existing monitor command to use adaptive reporting.
    
    This replaces the static report generation with dynamic, adaptive reports.
    """
    try:
        # Step 1: Perform search (existing logic)
        # This would use your existing search functionality
        search_result = await perform_issue_search(topic, period)
        
        if not search_result or search_result.total_found == 0:
            await ctx.send("🔍 해당 주제에 대한 신뢰도 높은 이슈를 찾지 못했습니다.")
            return
        
        # Step 2: Send quick summary immediately
        quick_embed = await generate_quick_adaptive_summary(
            search_result,
            target_audience=audience
        )
        quick_message = await ctx.send(embed=quick_embed)
        
        # Step 3: Generate complete adaptive report package
        embed, markdown_file, pdf_file = await generate_adaptive_discord_report(
            search_result,
            target_audience=audience
        )
        
        # Step 4: Send complete report
        files = []
        if markdown_file:
            files.append(markdown_file)
        if pdf_file:
            files.append(pdf_file)
        
        if files:
            await ctx.send(embed=embed, files=files)
            # Delete quick summary
            try:
                await quick_message.delete()
            except:
                pass
        else:
            # Update quick message with final embed
            await quick_message.edit(embed=embed)
            
    except Exception as e:
        await ctx.send(f"❌ 보고서 생성 중 오류가 발생했습니다: {str(e)}")


async def example_advanced_monitor_command(
    ctx,
    topic: str,
    period: str = "1주일",
    audience: str = "general_public",
    format_type: str = "adaptive"
):
    """
    Advanced example with multiple options.
    """
    # Initialize orchestrator
    orchestrator = AdaptiveReportingOrchestrator()
    
    try:
        # Perform search
        search_result = await perform_issue_search(topic, period)
        
        if format_type == "adaptive":
            # Use new adaptive system
            embed, md_file, pdf_file = await orchestrator.generate_complete_report_package(
                search_result,
                target_audience=audience,
                include_pdf=True
            )
            
            files = [f for f in [md_file, pdf_file] if f is not None]
            await ctx.send(embed=embed, files=files if files else None)
            
        elif format_type == "quick":
            # Quick summary only
            embed = await orchestrator.generate_quick_summary(search_result)
            await ctx.send(embed=embed)
            
        else:
            # Legacy format
            embed, md_file, pdf_file = await orchestrator._generate_fallback_package(
                search_result,
                include_pdf=True
            )
            files = [f for f in [md_file, pdf_file] if f is not None]
            await ctx.send(embed=embed, files=files if files else None)
            
    except Exception as e:
        await ctx.send(f"❌ 오류: {str(e)}")


# Example of how to create different report types for different channels/users
async def example_channel_specific_reporting(ctx, topic: str, period: str = "1주일"):
    """
    Example of how to customize reports based on Discord channel or user roles.
    """
    orchestrator = AdaptiveReportingOrchestrator()
    
    # Determine audience based on channel name or user roles
    channel_name = ctx.channel.name.lower()
    user_roles = [role.name.lower() for role in ctx.author.roles] if hasattr(ctx.author, 'roles') else []
    
    # Determine target audience
    if 'executive' in channel_name or 'ceo' in user_roles or 'manager' in user_roles:
        audience = "executive"
    elif 'tech' in channel_name or 'developer' in user_roles or 'engineer' in user_roles:
        audience = "technical"
    elif 'research' in channel_name or 'scientist' in user_roles:
        audience = "researcher"
    elif 'business' in channel_name or 'analyst' in user_roles:
        audience = "business_analyst"
    else:
        audience = "general_public"
    
    # Generate search result
    search_result = await perform_issue_search(topic, period)
    
    # Generate appropriate report
    embed, md_file, pdf_file = await orchestrator.generate_complete_report_package(
        search_result,
        target_audience=audience
    )
    
    # Send with context message
    context_msg = {
        "executive": "경영진을 위한 전략적 인사이트 중심의 보고서입니다.",
        "technical": "기술적 세부사항과 구현 관점의 보고서입니다.",
        "researcher": "연구 방법론과 학술적 관점의 보고서입니다.",
        "business_analyst": "시장 분석과 비즈니스 영향 중심의 보고서입니다.",
        "general_public": "일반적인 이해를 위한 보고서입니다."
    }.get(audience, "")
    
    if context_msg:
        await ctx.send(f"💡 {context_msg}")
    
    files = [f for f in [md_file, pdf_file] if f is not None]
    await ctx.send(embed=embed, files=files if files else None)


# Example helper function for testing
async def perform_issue_search(topic: str, period: str) -> SearchResult:
    """
    Mock search function for testing.
    Replace this with your actual search implementation.
    """
    # This is a mock implementation
    # In reality, this would use your existing search system
    
    mock_issues = [
        IssueItem(
            title=f"{topic} 관련 최신 기술 동향",
            summary=f"{topic}에 대한 최신 기술 발전 현황을 분석합니다.",
            source="TechNews",
            published_date="2024-12-06",
            relevance_score=0.9,
            category="technology",
            content_snippet="기술 발전 스니펫...",
            detailed_content=f"{topic}의 상세한 기술적 분석 내용...",
            background_context=f"{topic}의 배경 및 맥락 정보..."
        ),
        IssueItem(
            title=f"{topic} 시장 영향 분석",
            summary=f"{topic}이 시장에 미치는 영향을 분석합니다.",
            source="BusinessDaily",
            published_date="2024-12-05",
            relevance_score=0.8,
            category="business",
            content_snippet="시장 영향 스니펫...",
            detailed_content=f"{topic}의 시장 영향 상세 분석...",
            background_context=f"{topic}의 시장 배경 정보..."
        )
    ]
    
    # Add mock confidence scores
    for issue in mock_issues:
        setattr(issue, 'combined_confidence', 0.85)
        setattr(issue, 'reppl_confidence', 0.8)
    
    return SearchResult(
        topic=topic,
        keywords=[topic, f"{topic} 기술", f"{topic} 시장"],
        period=period,
        issues=mock_issues,
        total_found=len(mock_issues),
        search_time=1.5
    )


# Example of how to add new slash commands
def setup_adaptive_commands(bot):
    """
    Example of how to add new slash commands for adaptive reporting.
    """
    
    @bot.slash_command(name="monitor_adaptive", description="적응형 이슈 모니터링")
    async def monitor_adaptive(
        ctx,
        topic: str,
        period: str = "1주일",
        audience: str = "general_public"
    ):
        await example_monitor_command_with_adaptive_reporting(ctx, topic, period, audience)
    
    @bot.slash_command(name="quick_summary", description="빠른 요약 보고서")
    async def quick_summary(ctx, topic: str, period: str = "1주일"):
        search_result = await perform_issue_search(topic, period)
        embed = await generate_quick_adaptive_summary(search_result)
        await ctx.send(embed=embed)
    
    @bot.slash_command(name="executive_brief", description="경영진 브리핑")
    async def executive_brief(ctx, topic: str, period: str = "1주일"):
        await example_monitor_command_with_adaptive_reporting(ctx, topic, period, "executive")
    
    @bot.slash_command(name="tech_report", description="기술 상세 보고서")
    async def tech_report(ctx, topic: str, period: str = "1주일"):
        await example_monitor_command_with_adaptive_reporting(ctx, topic, period, "technical")


if __name__ == "__main__":
    # Example test run
    async def test_adaptive_reporting():
        """Test the adaptive reporting system."""
        
        # Create mock search result
        search_result = await perform_issue_search("인공지능", "1주일")
        
        # Test different audiences
        audiences = ["executive", "technical", "general_public", "researcher"]
        
        for audience in audiences:
            print(f"\n=== Testing {audience} audience ===")
            
            try:
                embed, md_file, pdf_file = await generate_adaptive_discord_report(
                    search_result,
                    target_audience=audience
                )
                
                print(f"✅ Embed title: {embed.title}")
                print(f"✅ Embed fields: {len(embed.fields)}")
                print(f"✅ Markdown file: {'Yes' if md_file else 'No'}")
                print(f"✅ PDF file: {'Yes' if pdf_file else 'No'}")
                
            except Exception as e:
                print(f"❌ Error for {audience}: {e}")
    
    # Run test
    asyncio.run(test_adaptive_reporting())