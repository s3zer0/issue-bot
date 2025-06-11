"""
Dynamic Prompt Generation System using GPT-4o.

This module provides a flexible framework for generating context-aware prompts
for various AI tasks including refactoring, keyword generation, issue analysis,
and other AI-driven functionalities.
"""

import asyncio
import json
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import openai
from loguru import logger

from src.config import config


class PromptType(Enum):
    """Supported prompt types."""
    REFACTORING = "refactoring"
    KEYWORD_GENERATION = "keyword_generation"
    ISSUE_ANALYSIS = "issue_analysis"
    REPORT_GENERATION = "report_generation"
    CODE_REVIEW = "code_review"
    GENERAL = "general"


@dataclass
class PromptResult:
    """Result of prompt generation."""
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    template_used: str = ""
    context_hash: str = ""
    generation_time: datetime = field(default_factory=datetime.now)
    token_count: Optional[int] = None
    safety_level: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "metadata": self.metadata,
            "template_used": self.template_used,
            "context_hash": self.context_hash,
            "generation_time": self.generation_time.isoformat(),
            "token_count": self.token_count,
            "safety_level": self.safety_level
        }


@dataclass
class PromptHistory:
    """Track prompt generation history for improvement."""
    prompts: List[PromptResult] = field(default_factory=list)
    max_history: int = 100
    
    def add(self, result: PromptResult):
        """Add a new prompt result to history."""
        self.prompts.append(result)
        if len(self.prompts) > self.max_history:
            self.prompts.pop(0)
    
    def get_recent(self, count: int = 10) -> List[PromptResult]:
        """Get recent prompt results."""
        return self.prompts[-count:]
    
    def find_by_context_hash(self, context_hash: str) -> List[PromptResult]:
        """Find prompts by context hash."""
        return [p for p in self.prompts if p.context_hash == context_hash]


class BasePromptGenerator(ABC):
    """Base class for all prompt generators."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize the base prompt generator.
        
        Args:
            model: OpenAI model to use for prompt generation
            api_key: OpenAI API key (uses config if not provided)
        """
        self.model = model
        self.api_key = api_key or config.get_openai_api_key()
        self.client = openai.AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        self.history = PromptHistory()
        
        if not self.api_key:
            logger.warning("OpenAI API key not found. Prompt generation will use fallback templates.")
    
    @abstractmethod
    def get_prompt_type(self) -> PromptType:
        """Return the prompt type this generator handles."""
        pass
    
    @abstractmethod
    def get_safety_constraints(self) -> List[str]:
        """Return safety constraints specific to this prompt type."""
        pass
    
    @abstractmethod
    def get_default_template(self) -> str:
        """Return the default template for this prompt type."""
        pass
    
    def _generate_context_hash(self, context: Dict[str, Any]) -> str:
        """Generate a hash for the context to enable caching and history tracking."""
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(context_str.encode()).hexdigest()
    
    async def generate(self, context: Dict[str, Any], **kwargs) -> PromptResult:
        """
        Generate a prompt based on the provided context.
        
        Args:
            context: Context information for prompt generation
            **kwargs: Additional generation parameters
            
        Returns:
            PromptResult containing the generated prompt and metadata
        """
        context_hash = self._generate_context_hash(context)
        
        # Check for cached results
        cached_results = self.history.find_by_context_hash(context_hash)
        if cached_results and kwargs.get('use_cache', True):
            logger.debug(f"Using cached prompt for context hash: {context_hash}")
            return cached_results[-1]  # Return most recent
        
        try:
            if self.client:
                result = await self._generate_with_ai(context, **kwargs)
            else:
                result = await self._generate_with_template(context, **kwargs)
            
            result.context_hash = context_hash
            self.history.add(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            # Fallback to template-based generation
            result = await self._generate_with_template(context, **kwargs)
            result.context_hash = context_hash
            result.metadata['fallback_used'] = True
            result.metadata['error'] = str(e)
            
            return result
    
    async def _generate_with_ai(self, context: Dict[str, Any], **kwargs) -> PromptResult:
        """Generate prompt using AI (GPT-4o)."""
        prompt_type = self.get_prompt_type()
        safety_constraints = self.get_safety_constraints()
        
        # Build meta-prompt for GPT-4o
        meta_prompt = self._build_meta_prompt(context, prompt_type, safety_constraints, **kwargs)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert prompt engineer specializing in creating high-quality, context-aware prompts for AI systems. You understand the nuances of different AI tasks and can generate prompts that are both effective and safe."
                    },
                    {
                        "role": "user",
                        "content": meta_prompt
                    }
                ],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 2000)
            )
            
            generated_prompt = response.choices[0].message.content.strip()
            
            return PromptResult(
                prompt=generated_prompt,
                metadata={
                    "model_used": self.model,
                    "generation_method": "ai",
                    "temperature": kwargs.get('temperature', 0.7),
                    "input_context": context
                },
                template_used="ai_generated",
                token_count=response.usage.total_tokens if response.usage else None,
                safety_level=kwargs.get('safety_level', 'standard')
            )
            
        except Exception as e:
            logger.error(f"AI prompt generation failed: {e}")
            raise
    
    async def _generate_with_template(self, context: Dict[str, Any], **kwargs) -> PromptResult:
        """Generate prompt using template fallback."""
        template = self.get_default_template()
        
        try:
            # Simple template formatting
            formatted_prompt = template.format(**context)
        except KeyError as e:
            logger.warning(f"Template formatting failed, missing key: {e}")
            # Create a more robust fallback
            formatted_prompt = self._create_emergency_fallback(context)
        
        return PromptResult(
            prompt=formatted_prompt,
            metadata={
                "generation_method": "template",
                "input_context": context
            },
            template_used="default_template",
            safety_level=kwargs.get('safety_level', 'standard')
        )
    
    def _build_meta_prompt(
        self, 
        context: Dict[str, Any], 
        prompt_type: PromptType, 
        safety_constraints: List[str],
        **kwargs
    ) -> str:
        """Build meta-prompt for GPT-4o to generate the actual prompt."""
        
        bilingual_requirement = kwargs.get('bilingual', True)
        
        meta_prompt = f"""
I need you to generate a high-quality prompt for {prompt_type.value} tasks. 

**Context Information:**
{json.dumps(context, indent=2, default=str)}

**Prompt Type:** {prompt_type.value}

**Safety Constraints (CRITICAL - Must be included):**
{chr(10).join(f"- {constraint}" for constraint in safety_constraints)}

**Requirements:**
1. The prompt should be clear, specific, and actionable
2. Include appropriate context from the provided information
3. Ensure the prompt guides the AI to produce safe, reliable results
4. {"Include both English and Korean instructions if appropriate" if bilingual_requirement else "Use English only"}
5. The prompt should be production-ready and robust

**Output Format:**
Generate ONLY the prompt that will be used with another AI system. Do not include explanations, just the prompt itself.

**Additional Specifications:**
- Safety level: {kwargs.get('safety_level', 'standard')}
- Target audience: {kwargs.get('target_audience', 'technical')}
- Complexity level: {kwargs.get('complexity', 'intermediate')}
"""
        
        return meta_prompt.strip()
    
    def _create_emergency_fallback(self, context: Dict[str, Any]) -> str:
        """Create an emergency fallback prompt when all else fails."""
        prompt_type = self.get_prompt_type()
        
        return f"""
Task: {prompt_type.value}

Context: {json.dumps(context, default=str)}

Please proceed with this {prompt_type.value} task using the provided context.
Follow standard best practices and safety guidelines.

Instructions:
1. Analyze the provided context carefully
2. Apply appropriate {prompt_type.value} techniques
3. Ensure output quality and safety
4. Provide clear and actionable results

Safety reminder: {'; '.join(self.get_safety_constraints())}
"""


class RefactoringPromptGenerator(BasePromptGenerator):
    """Specialized prompt generator for code refactoring tasks."""
    
    def get_prompt_type(self) -> PromptType:
        return PromptType.REFACTORING
    
    def get_safety_constraints(self) -> List[str]:
        return [
            "NEVER change business logic or functional behavior",
            "ONLY reorganize file structure and update import paths",
            "Maintain all existing public APIs and entry points",
            "Do not modify configuration files or environment settings",
            "Preserve all existing functionality and backward compatibility",
            "Focus solely on structural improvements and code organization"
        ]
    
    def get_default_template(self) -> str:
        return """
You are a senior software engineer performing a code refactoring task.

**CRITICAL CONSTRAINTS:**
- DO NOT modify any business logic or functionality
- ONLY restructure files and update import statements
- Maintain complete backward compatibility
- Preserve all existing APIs and interfaces

**Project Context:**
{project_structure}

**Current Issues:**
{identified_issues}

**Refactoring Goals:**
{refactoring_goals}

**Task:**
Create a step-by-step refactoring plan that addresses the identified issues while strictly adhering to the constraints above. Focus on:

1. Directory structure reorganization
2. File movement and renaming
3. Import path updates
4. Module organization improvements

**Output Format:**
Provide a detailed, actionable refactoring plan with specific steps and file movements.

**Safety Reminder:** This is a structural refactoring only. No logic changes are permitted.
"""


class KeywordPromptGenerator(BasePromptGenerator):
    """Specialized prompt generator for keyword generation tasks."""
    
    def get_prompt_type(self) -> PromptType:
        return PromptType.KEYWORD_GENERATION
    
    def get_safety_constraints(self) -> List[str]:
        return [
            "Generate only relevant and accurate keywords",
            "Avoid biased or controversial terminology",
            "Focus on current and factual information",
            "Ensure keywords are appropriate for the intended audience",
            "Prioritize timeliness and relevance over quantity",
            "Include diverse perspectives when applicable"
        ]
    
    def get_default_template(self) -> str:
        return """
You are an expert keyword researcher specializing in current trends and technical topics.

**Topic:** {topic}
**Context:** {context}
**Target Audience:** {target_audience}
**Keyword Style:** {keyword_style}

**Task:**
Generate high-quality, relevant keywords for the specified topic. Consider:

1. Current trends and developments
2. Technical accuracy and precision
3. Search relevance and popularity
4. Emerging subtopics and related areas

**Output Requirements:**
- Provide 10-20 primary keywords
- Include 5-10 trending/emerging keywords
- Categorize by relevance level (High/Medium/Low)
- Brief explanation for each category

**Safety Guidelines:**
{safety_constraints}
"""


class IssueAnalysisPromptGenerator(BasePromptGenerator):
    """Specialized prompt generator for issue analysis tasks."""
    
    def get_prompt_type(self) -> PromptType:
        return PromptType.ISSUE_ANALYSIS
    
    def get_safety_constraints(self) -> List[str]:
        return [
            "Maintain objectivity and avoid bias in analysis",
            "Distinguish between facts and opinions clearly",
            "Consider multiple perspectives and viewpoints",
            "Flag potential misinformation or unreliable sources",
            "Focus on constructive analysis rather than criticism",
            "Respect privacy and confidentiality concerns"
        ]
    
    def get_default_template(self) -> str:
        return """
You are a senior analyst specializing in comprehensive issue evaluation.

**Issues to Analyze:**
{issues_data}

**Analysis Scope:**
{analysis_scope}

**Focus Areas:**
{focus_areas}

**Task:**
Perform a thorough analysis of the provided issues, considering:

1. Technical feasibility and implications
2. Market impact and relevance
3. Risk assessment and mitigation
4. Stakeholder perspectives
5. Long-term consequences

**Analysis Framework:**
- Categorize issues by priority and impact
- Identify patterns and relationships
- Assess reliability of sources
- Provide actionable insights

**Safety Considerations:**
{safety_constraints}
"""


class PromptGenerator:
    """Main factory class for creating specialized prompt generators."""
    
    _generators: Dict[PromptType, Type[BasePromptGenerator]] = {
        PromptType.REFACTORING: RefactoringPromptGenerator,
        PromptType.KEYWORD_GENERATION: KeywordPromptGenerator,
        PromptType.ISSUE_ANALYSIS: IssueAnalysisPromptGenerator,
    }
    
    @classmethod
    def create(
        self, 
        prompt_type: Union[PromptType, str], 
        model: str = "gpt-4o",
        api_key: Optional[str] = None
    ) -> BasePromptGenerator:
        """
        Create a specialized prompt generator.
        
        Args:
            prompt_type: Type of prompt generator to create
            model: OpenAI model to use
            api_key: OpenAI API key
            
        Returns:
            Specialized prompt generator instance
        """
        if isinstance(prompt_type, str):
            try:
                prompt_type = PromptType(prompt_type.lower())
            except ValueError:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        generator_class = self._generators.get(prompt_type)
        if not generator_class:
            raise ValueError(f"No generator available for prompt type: {prompt_type}")
        
        return generator_class(model=model, api_key=api_key)
    
    @classmethod
    def register_generator(
        cls, 
        prompt_type: PromptType, 
        generator_class: Type[BasePromptGenerator]
    ):
        """Register a new prompt generator type."""
        cls._generators[prompt_type] = generator_class
        logger.info(f"Registered new prompt generator: {prompt_type}")
    
    @classmethod
    def list_available_types(cls) -> List[PromptType]:
        """List all available prompt types."""
        return list(cls._generators.keys())


# Convenience functions for common use cases
async def generate_refactoring_prompt(
    project_structure: Dict[str, Any],
    identified_issues: List[str],
    refactoring_goals: List[str],
    **kwargs
) -> PromptResult:
    """Generate a refactoring prompt with project analysis."""
    generator = PromptGenerator.create(PromptType.REFACTORING)
    
    context = {
        "project_structure": project_structure,
        "identified_issues": identified_issues,
        "refactoring_goals": refactoring_goals
    }
    
    return await generator.generate(context, **kwargs)


async def generate_keyword_prompt(
    topic: str,
    context: str = "",
    target_audience: str = "technical",
    keyword_style: str = "comprehensive",
    **kwargs
) -> PromptResult:
    """Generate a keyword generation prompt."""
    generator = PromptGenerator.create(PromptType.KEYWORD_GENERATION)
    
    context_dict = {
        "topic": topic,
        "context": context,
        "target_audience": target_audience,
        "keyword_style": keyword_style
    }
    
    return await generator.generate(context_dict, **kwargs)


async def generate_issue_analysis_prompt(
    issues_data: List[Dict[str, Any]],
    analysis_scope: str = "comprehensive",
    focus_areas: List[str] = None,
    **kwargs
) -> PromptResult:
    """Generate an issue analysis prompt."""
    generator = PromptGenerator.create(PromptType.ISSUE_ANALYSIS)
    
    context = {
        "issues_data": issues_data,
        "analysis_scope": analysis_scope,
        "focus_areas": focus_areas or ["technical", "market", "risk"]
    }
    
    return await generator.generate(context, **kwargs)


# Export main classes and functions
__all__ = [
    'PromptType',
    'PromptResult',
    'PromptHistory',
    'BasePromptGenerator',
    'RefactoringPromptGenerator',
    'KeywordPromptGenerator', 
    'IssueAnalysisPromptGenerator',
    'PromptGenerator',
    'generate_refactoring_prompt',
    'generate_keyword_prompt',
    'generate_issue_analysis_prompt'
]