#!/usr/bin/env python3
"""
Demo script for the Dynamic Prompt Generation System.

This script demonstrates various capabilities of the prompt generation system
including refactoring analysis, keyword generation, and issue analysis.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.prompt_generator import (
    PromptGenerator,
    generate_refactoring_prompt,
    generate_keyword_prompt,
    generate_issue_analysis_prompt
)
from src.utils.project_analyzer import ProjectAnalyzer
from src.utils.prompt_templates import TemplateLanguage, SafetyLevel
from loguru import logger


async def demo_refactoring_prompt():
    """Demonstrate refactoring prompt generation."""
    print("🔨 DEMO: Refactoring Prompt Generation")
    print("=" * 50)
    
    # Analyze current project
    analyzer = ProjectAnalyzer(".")
    analysis = await analyzer.analyze_project()
    
    # Generate context for refactoring
    context = analyzer.generate_refactoring_context(analysis)
    
    # Generate prompt
    result = await generate_refactoring_prompt(
        project_structure=context["project_structure"],
        identified_issues=context["identified_issues"], 
        refactoring_goals=context["refactoring_goals"],
        bilingual=True,
        safety_level="standard"
    )
    
    print(f"Generated prompt ({result.template_used}):")
    print("-" * 30)
    print(result.prompt[:500] + "..." if len(result.prompt) > 500 else result.prompt)
    print("-" * 30)
    print(f"Safety Level: {result.safety_level}")
    print(f"Token Count: {result.token_count}")
    print()


async def demo_keyword_prompt():
    """Demonstrate keyword generation prompt."""
    print("🔍 DEMO: Keyword Generation Prompt")
    print("=" * 50)
    
    result = await generate_keyword_prompt(
        topic="AI-powered code refactoring",
        context="Focus on modern software engineering practices",
        target_audience="software developers",
        keyword_style="technical"
    )
    
    print(f"Generated prompt ({result.template_used}):")
    print("-" * 30)
    print(result.prompt[:500] + "..." if len(result.prompt) > 500 else result.prompt)
    print("-" * 30)
    print()


async def demo_issue_analysis_prompt():
    """Demonstrate issue analysis prompt generation."""
    print("📊 DEMO: Issue Analysis Prompt")
    print("=" * 50)
    
    # Sample issues data
    sample_issues = [
        {
            "id": "tech-001",
            "title": "Legacy Code Maintenance Challenges",
            "description": "Increasing difficulty maintaining large legacy codebases without proper documentation",
            "confidence": 0.87,
            "source": "Developer Survey 2024"
        },
        {
            "id": "tech-002", 
            "title": "AI Code Generation Reliability",
            "description": "Concerns about AI-generated code quality and security vulnerabilities",
            "confidence": 0.92,
            "source": "Security Research Report"
        }
    ]
    
    result = await generate_issue_analysis_prompt(
        issues_data=sample_issues,
        analysis_scope="comprehensive",
        focus_areas=["technical feasibility", "market impact", "risk assessment"]
    )
    
    print(f"Generated prompt ({result.template_used}):")
    print("-" * 30)
    print(result.prompt[:600] + "..." if len(result.prompt) > 600 else result.prompt)
    print("-" * 30)
    print()


async def demo_factory_usage():
    """Demonstrate using the factory pattern."""
    print("🏭 DEMO: Factory Pattern Usage")
    print("=" * 50)
    
    # Create different generators
    refactoring_gen = PromptGenerator.create("refactoring")
    keyword_gen = PromptGenerator.create("keyword_generation") 
    
    print(f"Available prompt types: {[t.value for t in PromptGenerator.list_available_types()]}")
    print()
    
    # Generate with factory
    context = {
        "topic": "quantum machine learning",
        "context": "Emerging field combining quantum computing and AI",
        "target_audience": "researchers",
        "keyword_style": "academic"
    }
    
    result = await keyword_gen.generate(context)
    print(f"Factory-generated keyword prompt snippet:")
    print(result.prompt[:300] + "...")
    print()


async def demo_template_fallback():
    """Demonstrate template fallback when no API key."""
    print("🛡️ DEMO: Template Fallback (No API)")
    print("=" * 50)
    
    # Create generator without API key
    generator = PromptGenerator.create("refactoring", api_key="")
    
    context = {
        "project_structure": "Simple project structure",
        "identified_issues": ["Example issue"],
        "refactoring_goals": ["Improve organization"]
    }
    
    result = await generator.generate(context)
    
    print(f"Template fallback prompt ({result.template_used}):")
    print("-" * 30)
    print(result.prompt[:400] + "..." if len(result.prompt) > 400 else result.prompt)
    print("-" * 30)
    print(f"Fallback used: {result.metadata.get('fallback_used', False)}")
    print()


async def demo_multilingual_support():
    """Demonstrate multilingual prompt generation."""
    print("🌐 DEMO: Multilingual Support")
    print("=" * 50)
    
    # Create generators with different languages
    for lang in [TemplateLanguage.ENGLISH, TemplateLanguage.KOREAN, TemplateLanguage.BILINGUAL]:
        generator = PromptGenerator.create("keyword_generation")
        
        context = {
            "topic": "인공지능 윤리",
            "context": "AI ethics research",
            "target_audience": "researchers",
            "keyword_style": "academic"
        }
        
        # Note: Language is passed as a parameter during generation
        result = await generator.generate(context, language=lang)
        
        print(f"{lang.value.upper()} prompt snippet:")
        print(result.prompt[:200] + "...")
        print()


async def main():
    """Run all demos."""
    print("🚀 Dynamic Prompt Generation System Demo")
    print("=" * 60)
    print()
    
    try:
        # Run all demo functions
        await demo_refactoring_prompt()
        await demo_keyword_prompt() 
        await demo_issue_analysis_prompt()
        await demo_factory_usage()
        await demo_template_fallback()
        await demo_multilingual_support()
        
        print("✅ All demos completed successfully!")
        print()
        print("💡 Next steps:")
        print("1. Try the CLI: python scripts/generate_prompt.py --list-types")
        print("2. Generate your own prompts with different parameters")
        print("3. Integrate the system into your workflow")
        print("4. Extend with custom prompt types")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())