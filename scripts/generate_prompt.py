#!/usr/bin/env python3
"""
CLI Tool for Dynamic Prompt Generation.

This script provides a command-line interface for generating various types of prompts
using the dynamic prompt generation system. It supports multiple prompt types and
can analyze projects to generate context-aware prompts.

Usage:
    python generate_prompt.py --type refactoring
    python generate_prompt.py --type keyword --topic "AI ethics"
    python generate_prompt.py --type issue_analysis --data issues.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.prompt_generator import (
    PromptGenerator, PromptType, PromptResult,
    generate_refactoring_prompt,
    generate_keyword_prompt,
    generate_issue_analysis_prompt
)
from src.utils.project_analyzer import ProjectAnalyzer
from src.utils.prompt_templates import TemplateLanguage, SafetyLevel
from src.config import config
from loguru import logger


class PromptGeneratorCLI:
    """Command-line interface for prompt generation."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.parser = self._create_parser()
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            description="Dynamic Prompt Generation CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --type refactoring
  %(prog)s --type keyword --topic "quantum computing" --audience technical
  %(prog)s --type issue_analysis --data issues.json --scope comprehensive
  %(prog)s --type refactoring --project /path/to/project --output prompt.txt
  %(prog)s --list-types
            """
        )
        
        # Main arguments
        parser.add_argument(
            "--type", "-t",
            choices=["refactoring", "keyword", "issue_analysis", "report"],
            help="Type of prompt to generate"
        )
        
        parser.add_argument(
            "--output", "-o",
            help="Output file path (default: stdout)"
        )
        
        parser.add_argument(
            "--format",
            choices=["text", "json"],
            default="text",
            help="Output format (default: text)"
        )
        
        parser.add_argument(
            "--language",
            choices=["english", "korean", "bilingual"],
            default="bilingual",
            help="Prompt language (default: bilingual)"
        )
        
        parser.add_argument(
            "--safety-level",
            choices=["minimal", "standard", "strict", "paranoid"],
            default="standard",
            help="Safety level for constraints (default: standard)"
        )
        
        parser.add_argument(
            "--model",
            default="gpt-4o",
            help="OpenAI model to use (default: gpt-4o)"
        )
        
        parser.add_argument(
            "--use-cache",
            action="store_true",
            help="Use cached results if available"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        
        # Utility arguments
        parser.add_argument(
            "--list-types",
            action="store_true",
            help="List available prompt types"
        )
        
        # Refactoring-specific arguments
        refactoring_group = parser.add_argument_group("refactoring", "Refactoring prompt options")
        refactoring_group.add_argument(
            "--project",
            help="Project directory to analyze (default: current directory)"
        )
        refactoring_group.add_argument(
            "--issues",
            nargs="*",
            help="Specific issues to address"
        )
        refactoring_group.add_argument(
            "--goals",
            nargs="*",
            help="Refactoring goals"
        )
        
        # Keyword generation arguments
        keyword_group = parser.add_argument_group("keyword", "Keyword generation options")
        keyword_group.add_argument(
            "--topic",
            help="Main topic for keyword generation"
        )
        keyword_group.add_argument(
            "--context",
            help="Additional context information"
        )
        keyword_group.add_argument(
            "--audience",
            default="technical",
            help="Target audience (default: technical)"
        )
        keyword_group.add_argument(
            "--style",
            default="comprehensive",
            help="Keyword style (default: comprehensive)"
        )
        
        # Issue analysis arguments
        issue_group = parser.add_argument_group("issue_analysis", "Issue analysis options")
        issue_group.add_argument(
            "--data",
            help="JSON file containing issues data"
        )
        issue_group.add_argument(
            "--scope",
            default="comprehensive",
            help="Analysis scope (default: comprehensive)"
        )
        issue_group.add_argument(
            "--focus",
            nargs="*",
            default=["technical", "market", "risk"],
            help="Focus areas for analysis"
        )
        
        return parser
    
    async def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI with provided arguments.
        
        Args:
            args: Command line arguments (uses sys.argv if None)
            
        Returns:
            Exit code (0 for success, 1 for error)
        """
        try:
            parsed_args = self.parser.parse_args(args)
            
            # Setup logging
            if parsed_args.verbose:
                logger.add(sys.stderr, level="DEBUG")
            else:
                logger.add(sys.stderr, level="INFO")
            
            # Handle utility commands
            if parsed_args.list_types:
                self._list_prompt_types()
                return 0
            
            # Validate required arguments
            if not parsed_args.type:
                self.parser.error("--type is required")
            
            # Generate prompt based on type
            result = await self._generate_prompt(parsed_args)
            
            # Output result
            await self._output_result(result, parsed_args)
            
            logger.success("Prompt generation completed successfully")
            return 0
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return 1
    
    def _list_prompt_types(self):
        """List available prompt types."""
        types = PromptGenerator.list_available_types()
        
        print("Available prompt types:")
        for prompt_type in types:
            print(f"  - {prompt_type.value}")
        
        print("\nUsage examples:")
        print("  python generate_prompt.py --type refactoring")
        print("  python generate_prompt.py --type keyword --topic 'AI safety'")
        print("  python generate_prompt.py --type issue_analysis --data issues.json")
    
    async def _generate_prompt(self, args: argparse.Namespace) -> PromptResult:
        """Generate prompt based on arguments."""
        
        # Convert string enums
        language = self._parse_language(args.language)
        safety_level = self._parse_safety_level(args.safety_level)
        
        # Generate based on type
        if args.type == "refactoring":
            return await self._generate_refactoring_prompt(args, language, safety_level)
        elif args.type == "keyword":
            return await self._generate_keyword_prompt(args, language, safety_level)
        elif args.type == "issue_analysis":
            return await self._generate_issue_analysis_prompt(args, language, safety_level)
        else:
            raise ValueError(f"Unsupported prompt type: {args.type}")
    
    async def _generate_refactoring_prompt(
        self, 
        args: argparse.Namespace,
        language: TemplateLanguage,
        safety_level: SafetyLevel
    ) -> PromptResult:
        """Generate refactoring prompt."""
        
        project_path = args.project or "."
        project_path = Path(project_path).resolve()
        
        if not project_path.exists():
            raise ValueError(f"Project directory does not exist: {project_path}")
        
        logger.info(f"Analyzing project at: {project_path}")
        
        # Analyze project
        analyzer = ProjectAnalyzer(str(project_path))
        analysis = await analyzer.analyze_project()
        
        # Generate context
        context = analyzer.generate_refactoring_context(analysis)
        
        # Override with CLI arguments if provided
        if args.issues:
            context["identified_issues"] = args.issues
        
        if args.goals:
            context["refactoring_goals"] = args.goals
        
        # Generate prompt
        return await generate_refactoring_prompt(
            project_structure=context["project_structure"],
            identified_issues=context["identified_issues"],
            refactoring_goals=context["refactoring_goals"],
            model=args.model,
            use_cache=args.use_cache,
            language=language,
            safety_level=safety_level
        )
    
    async def _generate_keyword_prompt(
        self,
        args: argparse.Namespace,
        language: TemplateLanguage,
        safety_level: SafetyLevel
    ) -> PromptResult:
        """Generate keyword prompt."""
        
        if not args.topic:
            raise ValueError("--topic is required for keyword generation")
        
        return await generate_keyword_prompt(
            topic=args.topic,
            context=args.context or "",
            target_audience=args.audience,
            keyword_style=args.style,
            model=args.model,
            use_cache=args.use_cache,
            language=language,
            safety_level=safety_level
        )
    
    async def _generate_issue_analysis_prompt(
        self,
        args: argparse.Namespace,
        language: TemplateLanguage,
        safety_level: SafetyLevel
    ) -> PromptResult:
        """Generate issue analysis prompt."""
        
        if not args.data:
            raise ValueError("--data is required for issue analysis")
        
        # Load issues data
        data_path = Path(args.data)
        if not data_path.exists():
            raise ValueError(f"Data file does not exist: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            issues_data = json.load(f)
        
        return await generate_issue_analysis_prompt(
            issues_data=issues_data,
            analysis_scope=args.scope,
            focus_areas=args.focus,
            model=args.model,
            use_cache=args.use_cache,
            language=language,
            safety_level=safety_level
        )
    
    def _parse_language(self, language_str: str) -> TemplateLanguage:
        """Parse language string to enum."""
        mapping = {
            "english": TemplateLanguage.ENGLISH,
            "korean": TemplateLanguage.KOREAN,
            "bilingual": TemplateLanguage.BILINGUAL
        }
        return mapping.get(language_str.lower(), TemplateLanguage.BILINGUAL)
    
    def _parse_safety_level(self, safety_str: str) -> SafetyLevel:
        """Parse safety level string to enum."""
        mapping = {
            "minimal": SafetyLevel.MINIMAL,
            "standard": SafetyLevel.STANDARD,
            "strict": SafetyLevel.STRICT,
            "paranoid": SafetyLevel.PARANOID
        }
        return mapping.get(safety_str.lower(), SafetyLevel.STANDARD)
    
    async def _output_result(self, result: PromptResult, args: argparse.Namespace):
        """Output the generated prompt result."""
        
        if args.format == "json":
            # Convert result to dict and handle enum serialization
            result_dict = result.to_dict()
            output_content = json.dumps(result_dict, indent=2, ensure_ascii=False, default=str)
        else:
            output_content = self._format_text_output(result)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            logger.info(f"Prompt saved to: {output_path}")
        else:
            print(output_content)
    
    def _format_text_output(self, result: PromptResult) -> str:
        """Format prompt result as text."""
        output = []
        
        # Header
        output.append("=" * 80)
        output.append("GENERATED PROMPT")
        output.append("=" * 80)
        output.append("")
        
        # Metadata
        output.append("METADATA:")
        output.append(f"  Template: {result.template_used}")
        output.append(f"  Safety Level: {result.safety_level}")
        output.append(f"  Generation Time: {result.generation_time}")
        if result.token_count:
            output.append(f"  Token Count: {result.token_count}")
        output.append("")
        
        # Main prompt
        output.append("PROMPT:")
        output.append("-" * 40)
        output.append(result.prompt)
        output.append("-" * 40)
        output.append("")
        
        # Additional metadata
        if result.metadata:
            output.append("ADDITIONAL METADATA:")
            for key, value in result.metadata.items():
                if key != "input_context":  # Skip large context data
                    output.append(f"  {key}: {value}")
        
        return "\n".join(output)


async def main():
    """Main entry point."""
    cli = PromptGeneratorCLI()
    exit_code = await cli.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())