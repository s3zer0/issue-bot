# 🤖 Adaptive Report Generation System

## Overview

The Adaptive Report Generation System is a dynamic and intelligent reporting framework that automatically adjusts report structure, content depth, and formatting based on:

- **Topic Classification** (technology, business, scientific, social/political, etc.)
- **Target Audience** (executives, technical teams, general public, researchers, analysts)
- **Content Complexity** (basic, intermediate, advanced, expert)
- **Time Sensitivity** (urgent, normal, low priority)

## Key Features

### 🎯 Intelligent Topic Classification
- Automatically categorizes topics using keyword analysis and LLM classification
- Supports 9+ topic categories with specialized handling for each
- Determines appropriate complexity level and target audience

### 📊 Dynamic Section Generation
- **Technology Topics**: Technical specifications, implementation timelines, compatibility analysis
- **Business Topics**: Market impact, competitor analysis, financial implications
- **Scientific Topics**: Methodology assessment, peer review status, research implications
- **Social/Political Topics**: Stakeholder analysis, public sentiment, policy implications

### 🎨 Adaptive Formatting
- Discord embeds with category-specific colors and layouts
- Automatic content length adjustment based on complexity
- Smart visualization selection (charts, tables, timelines, networks)
- Audience-appropriate language and technical depth

### ⚡ Smart Delivery
- Quick summary for immediate response
- Detailed markdown and PDF reports for comprehensive analysis
- File attachment management respecting Discord limits

## Usage Examples

### Basic Usage

```python
from src.reporting import generate_adaptive_discord_report

# Generate adaptive report
embed, markdown_file, pdf_file = await generate_adaptive_discord_report(
    search_result,
    target_audience="executive"  # or "technical", "general_public", etc.
)

# Send to Discord
await ctx.send(embed=embed, files=[markdown_file, pdf_file])
```

### Advanced Usage with Orchestrator

```python
from src.reporting import AdaptiveReportingOrchestrator, AudienceType

orchestrator = AdaptiveReportingOrchestrator()

# Generate complete package
embed, md_file, pdf_file = await orchestrator.generate_complete_report_package(
    search_result,
    target_audience=AudienceType.EXECUTIVE,
    include_pdf=True
)
```

### Quick Summary for Immediate Response

```python
from src.reporting import generate_quick_adaptive_summary

# Send immediate response while processing
quick_embed = await generate_quick_adaptive_summary(search_result)
await ctx.send(embed=quick_embed)
```

## Integration with Existing Bot

### Method 1: Replace Existing Commands

```python
# Before (static reporting)
@bot.slash_command(name="monitor")
async def monitor(ctx, topic: str, period: str = "1주일"):
    search_result = await perform_search(topic, period)
    embed = create_static_embed(search_result)
    await ctx.send(embed=embed)

# After (adaptive reporting)
@bot.slash_command(name="monitor")
async def monitor(ctx, topic: str, period: str = "1주일", audience: str = "general_public"):
    search_result = await perform_search(topic, period)
    embed, md_file, pdf_file = await generate_adaptive_discord_report(
        search_result, target_audience=audience
    )
    files = [f for f in [md_file, pdf_file] if f]
    await ctx.send(embed=embed, files=files)
```

### Method 2: Add Adaptive Commands

```python
@bot.slash_command(name="monitor_exec", description="Executive briefing")
async def monitor_exec(ctx, topic: str):
    search_result = await perform_search(topic)
    embed, md_file, pdf_file = await generate_adaptive_discord_report(
        search_result, target_audience="executive"
    )
    await ctx.send(embed=embed, files=[md_file, pdf_file])

@bot.slash_command(name="monitor_tech", description="Technical analysis")
async def monitor_tech(ctx, topic: str):
    search_result = await perform_search(topic)
    embed, md_file, pdf_file = await generate_adaptive_discord_report(
        search_result, target_audience="technical"
    )
    await ctx.send(embed=embed, files=[md_file, pdf_file])
```

### Method 3: Smart Channel-Based Adaptation

```python
async def smart_monitor(ctx, topic: str):
    # Auto-detect audience based on channel/user roles
    if 'executive' in ctx.channel.name:
        audience = "executive"
    elif 'tech' in ctx.channel.name:
        audience = "technical"
    else:
        audience = "general_public"
    
    search_result = await perform_search(topic)
    embed, md_file, pdf_file = await generate_adaptive_discord_report(
        search_result, target_audience=audience
    )
    await ctx.send(embed=embed, files=[md_file, pdf_file])
```

## Audience Types and Their Characteristics

### 👔 Executive (`executive`)
- **Focus**: Strategic insights, business impact, high-level summaries
- **Format**: Bullet points, executive summary emphasis, key metrics
- **Sections**: Market impact, financial implications, strategic recommendations
- **Length**: Concise, 100-250 words for summaries

### 🔧 Technical (`technical`)
- **Focus**: Implementation details, technical specifications, architecture
- **Format**: Code examples, technical diagrams, detailed analysis
- **Sections**: Technical specs, compatibility, implementation timelines
- **Length**: Comprehensive, 250-500 words for summaries

### 👥 General Public (`general_public`)
- **Focus**: Easy-to-understand explanations, practical implications
- **Format**: Simple language, analogies, visual aids
- **Sections**: Basic explanations, real-world impact, simple recommendations
- **Length**: Balanced, 150-300 words for summaries

### 🔬 Researcher (`researcher`)
- **Focus**: Methodology, data quality, research implications
- **Format**: Academic style, peer review status, research citations
- **Sections**: Methodology assessment, research implications, data analysis
- **Length**: Detailed, 300-500 words for summaries

### 📊 Business Analyst (`business_analyst`)
- **Focus**: Market analysis, competitive intelligence, business metrics
- **Format**: Charts, tables, trend analysis, competitive comparisons
- **Sections**: Market analysis, competitor overview, business opportunities
- **Length**: Comprehensive, 250-400 words for summaries

## Topic Categories and Specialized Sections

### 🚀 Technology
- Technical specifications and requirements
- Implementation timelines and roadmaps
- Compatibility and integration analysis
- Performance metrics and benchmarks

### 📈 Business
- Market impact and opportunity analysis
- Competitive landscape and positioning
- Financial implications and ROI analysis
- Strategic recommendations

### 🔬 Scientific
- Research methodology assessment
- Peer review and publication status
- Scientific implications and applications
- Data quality and statistical analysis

### 🏛️ Social/Political
- Stakeholder analysis and mapping
- Public sentiment and opinion analysis
- Policy implications and regulatory impact
- Community and social effects

### 🏥 Healthcare
- Clinical evidence and trials
- Safety and efficacy analysis
- Regulatory approval status
- Patient impact assessment

### 💰 Finance
- Market analysis and trends
- Risk assessment and management
- Investment implications
- Economic impact analysis

## Customization Options

### Complexity Levels
- **Basic**: Simple language, minimal technical terms
- **Intermediate**: Balanced technical content
- **Advanced**: Detailed technical analysis
- **Expert**: Comprehensive, domain-specific terminology

### Time Sensitivity
- **High (>0.8)**: Urgent attention indicators, immediate action items
- **Medium (0.4-0.8)**: Standard reporting with timeline recommendations
- **Low (<0.4)**: Strategic analysis, long-term implications

### Visualization Types
- **Charts**: For quantitative data and trends
- **Tables**: For comparisons and structured data
- **Timelines**: For sequential events and roadmaps
- **Networks**: For relationship mapping and stakeholders

## Error Handling and Fallbacks

The system includes multiple fallback levels:

1. **LLM Failure**: Falls back to keyword-based classification
2. **Classification Failure**: Uses general category with default settings
3. **Section Generation Failure**: Uses template-based content
4. **Complete Failure**: Falls back to legacy reporting system

## Performance Considerations

- **Quick Summaries**: Generated in 1-3 seconds for immediate response
- **Full Reports**: 10-30 seconds depending on complexity and LLM response time
- **Parallel Processing**: Multiple sections generated concurrently
- **Caching**: Topic classifications cached for similar queries

## Configuration

Environment variables for customization:
```env
# Default audience if not specified
DEFAULT_REPORT_AUDIENCE=general_public

# Default complexity level
DEFAULT_COMPLEXITY_LEVEL=intermediate

# Enable/disable PDF generation
ENABLE_ADAPTIVE_PDF=true

# Maximum sections per report
MAX_REPORT_SECTIONS=10
```

## Best Practices

1. **Use Quick Summaries First**: Send immediate response, then detailed report
2. **Match Audience to Channel**: Configure channel-specific default audiences
3. **Monitor LLM Usage**: Track API calls and costs
4. **Test Different Audiences**: Verify reports work well for all target groups
5. **Provide Feedback Loops**: Allow users to rate and improve report quality

## Migration from Legacy System

The new system is designed to coexist with the legacy reporting:

```python
# Gradual migration approach
async def hybrid_monitor(ctx, topic: str, use_adaptive: bool = True):
    search_result = await perform_search(topic)
    
    if use_adaptive:
        # New adaptive system
        embed, md_file, pdf_file = await generate_adaptive_discord_report(search_result)
    else:
        # Legacy system
        embed = create_legacy_embed(search_result)
        md_file = create_legacy_markdown(search_result)
        pdf_file = None
    
    files = [f for f in [md_file, pdf_file] if f]
    await ctx.send(embed=embed, files=files)
```

This allows for A/B testing and gradual rollout of the new system.