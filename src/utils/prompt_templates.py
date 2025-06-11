"""
Prompt Templates for Dynamic Prompt Generation System.

This module provides a comprehensive template system for various prompt types,
with inheritance support for common patterns and bilingual capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class TemplateLanguage(Enum):
    """Supported template languages."""
    ENGLISH = "en"
    KOREAN = "ko"
    BILINGUAL = "bilingual"


class SafetyLevel(Enum):
    """Safety levels for prompt templates."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class TemplateSection:
    """A section of a prompt template."""
    name: str
    content: str
    required: bool = True
    language: TemplateLanguage = TemplateLanguage.ENGLISH
    safety_constraints: List[str] = field(default_factory=list)
    
    def format(self, **kwargs) -> str:
        """Format the section with provided variables."""
        try:
            return self.content.format(**kwargs)
        except KeyError as e:
            if self.required:
                raise ValueError(f"Missing required template variable: {e}")
            return self.content


class BasePromptTemplate(ABC):
    """Base class for all prompt templates."""
    
    def __init__(
        self, 
        language: TemplateLanguage = TemplateLanguage.ENGLISH,
        safety_level: SafetyLevel = SafetyLevel.STANDARD
    ):
        """
        Initialize the base template.
        
        Args:
            language: Template language preference
            safety_level: Safety level for constraints
        """
        self.language = language
        self.safety_level = safety_level
        self.sections: Dict[str, TemplateSection] = {}
        self._initialize_sections()
    
    @abstractmethod
    def _initialize_sections(self):
        """Initialize template sections. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_required_variables(self) -> List[str]:
        """Return list of required template variables."""
        pass
    
    @abstractmethod
    def get_safety_constraints(self) -> List[str]:
        """Return safety constraints specific to this template."""
        pass
    
    def add_section(self, section: TemplateSection):
        """Add a section to the template."""
        self.sections[section.name] = section
    
    def get_section(self, name: str) -> Optional[TemplateSection]:
        """Get a section by name."""
        return self.sections.get(name)
    
    def format_template(self, **kwargs) -> str:
        """Format the complete template with provided variables."""
        # Validate required variables
        missing_vars = []
        for var in self.get_required_variables():
            if var not in kwargs:
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required template variables: {missing_vars}")
        
        # Add safety constraints to context
        kwargs['safety_constraints'] = self._format_safety_constraints()
        
        # Format sections in order
        formatted_sections = []
        for section_name in self._get_section_order():
            section = self.sections.get(section_name)
            if section:
                try:
                    formatted_content = section.format(**kwargs)
                    if formatted_content.strip():
                        formatted_sections.append(formatted_content)
                except Exception as e:
                    if section.required:
                        raise ValueError(f"Error formatting section '{section_name}': {e}")
        
        return "\n\n".join(formatted_sections)
    
    def _format_safety_constraints(self) -> str:
        """Format safety constraints for inclusion in template."""
        constraints = self.get_safety_constraints()
        if not constraints:
            return ""
        
        formatted = "**CRITICAL SAFETY CONSTRAINTS:**\n"
        for i, constraint in enumerate(constraints, 1):
            formatted += f"{i}. {constraint}\n"
        
        return formatted
    
    def _get_section_order(self) -> List[str]:
        """Get the order in which sections should be formatted."""
        # Default order - can be overridden by subclasses
        return [
            "header",
            "safety",
            "context",
            "instructions",
            "requirements", 
            "examples",
            "output_format",
            "footer"
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for serialization."""
        return {
            "language": self.language.value,
            "safety_level": self.safety_level.value,
            "sections": {
                name: {
                    "content": section.content,
                    "required": section.required,
                    "language": section.language.value
                }
                for name, section in self.sections.items()
            },
            "required_variables": self.get_required_variables(),
            "safety_constraints": self.get_safety_constraints()
        }


class RefactoringPromptTemplate(BasePromptTemplate):
    """Template for code refactoring prompts."""
    
    def _initialize_sections(self):
        """Initialize refactoring-specific sections."""
        
        # Header section
        self.add_section(TemplateSection(
            name="header",
            content="""# Code Refactoring Task

You are a senior software engineer performing a structural code refactoring."""
        ))
        
        # Safety section
        self.add_section(TemplateSection(
            name="safety",
            content="""{safety_constraints}

**REMINDER: This is STRUCTURAL REFACTORING ONLY**
- No business logic changes
- No functional modifications  
- Only file organization and import updates"""
        ))
        
        # Context section
        self.add_section(TemplateSection(
            name="context",
            content="""## Project Analysis

**Current Directory Structure:**
```
{project_structure}
```

**Identified Issues:**
{identified_issues}

**Refactoring Goals:**
{refactoring_goals}""" + ("""

**현재 디렉토리 구조:**
```
{project_structure}
```

**식별된 문제점:**
{identified_issues}

**리팩토링 목표:**
{refactoring_goals}""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Instructions section
        self.add_section(TemplateSection(
            name="instructions",
            content="""## Task Instructions

1. **Analyze** the current structure and identify improvement opportunities
2. **Plan** the new directory organization
3. **Map** file movements and import path changes
4. **Validate** that all functionality remains intact
5. **Document** the refactoring steps clearly""" + ("""

## 작업 지침

1. **분석**: 현재 구조를 분석하고 개선 기회를 식별
2. **계획**: 새로운 디렉토리 구조를 계획
3. **매핑**: 파일 이동 및 import 경로 변경사항을 매핑
4. **검증**: 모든 기능이 그대로 유지되는지 확인
5. **문서화**: 리팩토링 단계를 명확히 문서화""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Requirements section
        self.add_section(TemplateSection(
            name="requirements",
            content="""## Requirements

**Structure Requirements:**
- Group related functionality into logical modules
- Separate concerns appropriately
- Follow Python package conventions
- Maintain clean import paths

**Compatibility Requirements:**
- Preserve all public APIs
- Maintain backward compatibility
- Keep all entry points functional
- Ensure tests continue to pass""" + ("""

**구조 요구사항:**
- 관련 기능을 논리적 모듈로 그룹화
- 관심사를 적절히 분리
- Python 패키지 컨벤션 준수
- 깔끔한 import 경로 유지

**호환성 요구사항:**
- 모든 public API 보존
- 하위 호환성 유지
- 모든 진입점 기능 유지
- 테스트가 계속 통과하도록 보장""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Output format section
        self.add_section(TemplateSection(
            name="output_format",
            content="""## Output Format

Provide a detailed refactoring plan with:

1. **New Directory Structure** (with rationale)
2. **File Movement Plan** (source → destination)
3. **Import Update Requirements** (affected files and changes)
4. **Validation Steps** (how to verify nothing breaks)
5. **Implementation Order** (step-by-step execution plan)""" + ("""

## 출력 형식

다음을 포함한 상세 리팩토링 계획 제공:

1. **새로운 디렉토리 구조** (근거 포함)
2. **파일 이동 계획** (소스 → 대상)
3. **Import 업데이트 요구사항** (영향받는 파일과 변경사항)
4. **검증 단계** (문제없음을 확인하는 방법)
5. **구현 순서** (단계별 실행 계획)""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
    
    def get_required_variables(self) -> List[str]:
        return [
            "project_structure",
            "identified_issues", 
            "refactoring_goals"
        ]
    
    def get_safety_constraints(self) -> List[str]:
        base_constraints = [
            "NEVER modify business logic or functional behavior",
            "ONLY reorganize file structure and update import paths", 
            "Maintain ALL existing public APIs and entry points",
            "Preserve complete backward compatibility",
            "Do NOT modify configuration files or environment settings",
            "Focus SOLELY on structural improvements"
        ]
        
        if self.safety_level == SafetyLevel.STRICT:
            base_constraints.extend([
                "Create backup plans for each major change",
                "Validate each step before proceeding to the next",
                "Document all potential risks and mitigation strategies"
            ])
        elif self.safety_level == SafetyLevel.PARANOID:
            base_constraints.extend([
                "Require explicit approval for each file movement",
                "Test imports after each individual change",
                "Maintain rollback procedures for every modification",
                "Verify no hidden dependencies are broken"
            ])
        
        return base_constraints


class KeywordGenerationPromptTemplate(BasePromptTemplate):
    """Template for keyword generation prompts."""
    
    def _initialize_sections(self):
        """Initialize keyword generation sections."""
        
        # Header section
        self.add_section(TemplateSection(
            name="header",
            content="""# Keyword Generation Task

You are an expert keyword researcher and SEO specialist with deep knowledge of current trends and technical domains."""
        ))
        
        # Context section  
        self.add_section(TemplateSection(
            name="context",
            content="""## Research Context

**Primary Topic:** {topic}
**Context Information:** {context}
**Target Audience:** {target_audience}
**Keyword Style:** {keyword_style}
**Time Focus:** {time_focus}""" + ("""

## 연구 맥락

**주요 주제:** {topic}
**맥락 정보:** {context}
**대상 독자:** {target_audience}
**키워드 스타일:** {keyword_style}
**시간 초점:** {time_focus}""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Instructions section
        self.add_section(TemplateSection(
            name="instructions", 
            content="""## Task Instructions

Generate comprehensive, relevant keywords considering:

1. **Current Trends**: What's happening now in this domain
2. **Technical Accuracy**: Precise terminology and concepts
3. **Search Relevance**: Terms people actually search for
4. **Emerging Topics**: New developments and future directions
5. **Related Domains**: Connected fields and interdisciplinary areas""" + ("""

## 작업 지침

다음을 고려하여 포괄적이고 관련성 높은 키워드 생성:

1. **현재 트렌드**: 해당 도메인에서 현재 일어나고 있는 일
2. **기술적 정확성**: 정확한 용어와 개념
3. **검색 관련성**: 사람들이 실제로 검색하는 용어
4. **신흥 주제**: 새로운 발전과 미래 방향
5. **관련 도메인**: 연결된 분야와 학제간 영역""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Requirements section
        self.add_section(TemplateSection(
            name="requirements",
            content="""{safety_constraints}

## Quality Requirements

**Relevance**: All keywords must be directly relevant to the topic
**Timeliness**: Focus on current and emerging trends
**Diversity**: Include various perspectives and subtopics
**Accuracy**: Ensure technical terms are used correctly
**Completeness**: Cover the topic comprehensively""" + ("""

## 품질 요구사항

**관련성**: 모든 키워드는 주제와 직접적으로 관련되어야 함
**시의성**: 현재 및 신흥 트렌드에 초점
**다양성**: 다양한 관점과 하위 주제 포함
**정확성**: 기술 용어가 올바르게 사용되었는지 확인
**완전성**: 주제를 포괄적으로 다룸""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Output format section
        self.add_section(TemplateSection(
            name="output_format",
            content="""## Output Format

Organize keywords into these categories:

### 🔥 High Priority Keywords (10-15 keywords)
- Core terms central to the topic
- High search volume and relevance

### 📈 Trending Keywords (5-10 keywords)  
- Currently popular or emerging terms
- Recent developments and innovations

### 🔧 Technical Keywords (8-12 keywords)
- Specialized terminology
- Industry-specific concepts

### 🌐 Related Keywords (5-8 keywords)
- Adjacent topics and domains
- Cross-disciplinary connections

**For each keyword provide:**
- Relevance score (1-10)
- Brief explanation of current significance
- Suggested use cases""" + ("""

## 출력 형식

키워드를 다음 카테고리로 구성:

### 🔥 높은 우선순위 키워드 (10-15개 키워드)
- 주제의 핵심이 되는 용어
- 높은 검색량과 관련성

### 📈 트렌딩 키워드 (5-10개 키워드)
- 현재 인기 있거나 떠오르는 용어
- 최근 발전 및 혁신

### 🔧 기술적 키워드 (8-12개 키워드)
- 전문 용어
- 업계별 개념

### 🌐 관련 키워드 (5-8개 키워드)
- 인접 주제 및 도메인
- 학제간 연결

**각 키워드에 대해 제공:**
- 관련성 점수 (1-10)
- 현재 중요성에 대한 간략한 설명
- 제안된 사용 사례""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
    
    def get_required_variables(self) -> List[str]:
        return [
            "topic",
            "context",
            "target_audience",
            "keyword_style"
        ]
    
    def get_safety_constraints(self) -> List[str]:
        return [
            "Generate only factual and verified keywords",
            "Avoid biased, controversial, or inappropriate terms",
            "Focus on current and accurate information",
            "Ensure keywords are suitable for the target audience",
            "Prioritize quality and relevance over quantity",
            "Include diverse perspectives when applicable",
            "Verify technical accuracy of specialized terms"
        ]


class IssueAnalysisPromptTemplate(BasePromptTemplate):
    """Template for issue analysis prompts."""
    
    def _initialize_sections(self):
        """Initialize issue analysis sections."""
        
        # Header section
        self.add_section(TemplateSection(
            name="header",
            content="""# Issue Analysis Task

You are a senior analyst specializing in comprehensive technical and strategic issue evaluation."""
        ))
        
        # Context section
        self.add_section(TemplateSection(
            name="context",
            content="""## Analysis Context

**Issues Dataset:**
```json
{issues_data}
```

**Analysis Scope:** {analysis_scope}
**Focus Areas:** {focus_areas}
**Analysis Depth:** {analysis_depth}""" + ("""

## 분석 맥락

**이슈 데이터셋:**
```json
{issues_data}
```

**분석 범위:** {analysis_scope}
**초점 영역:** {focus_areas}
**분석 깊이:** {analysis_depth}""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Instructions section
        self.add_section(TemplateSection(
            name="instructions",
            content="""## Analysis Framework

Perform comprehensive analysis across these dimensions:

1. **Technical Assessment**
   - Feasibility evaluation
   - Implementation complexity
   - Technical risks and challenges

2. **Impact Analysis**
   - Market significance
   - Stakeholder effects
   - Long-term implications

3. **Risk Evaluation**
   - Potential downsides
   - Mitigation strategies
   - Contingency planning

4. **Pattern Recognition**
   - Common themes and trends
   - Relationships between issues
   - Emerging patterns""" + ("""

## 분석 프레임워크

다음 차원에서 포괄적 분석 수행:

1. **기술적 평가**
   - 실현 가능성 평가
   - 구현 복잡성
   - 기술적 위험과 도전

2. **영향 분석**
   - 시장 중요성
   - 이해관계자 효과
   - 장기적 영향

3. **위험 평가**
   - 잠재적 단점
   - 완화 전략
   - 비상 계획

4. **패턴 인식**
   - 공통 주제와 트렌드
   - 이슈 간 관계
   - 새로운 패턴""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Requirements section
        self.add_section(TemplateSection(
            name="requirements",
            content="""{safety_constraints}

## Analysis Requirements

**Objectivity**: Maintain neutral, fact-based analysis
**Completeness**: Address all specified focus areas
**Evidence**: Support conclusions with data and reasoning
**Actionability**: Provide practical insights and recommendations
**Clarity**: Present findings in clear, structured format""" + ("""

## 분석 요구사항

**객관성**: 중립적이고 사실 기반의 분석 유지
**완전성**: 지정된 모든 초점 영역 다루기
**증거**: 데이터와 추론으로 결론 뒷받침
**실행 가능성**: 실용적인 통찰과 권장사항 제공
**명확성**: 명확하고 구조화된 형식으로 결과 제시""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Output format section
        self.add_section(TemplateSection(
            name="output_format",
            content="""## Output Format

Structure your analysis as follows:

### 📋 Executive Summary
- Key findings overview
- Critical insights
- Primary recommendations

### 🔍 Detailed Analysis
For each major issue category:
- **Technical Feasibility**: Implementation assessment
- **Market Impact**: Significance and reach
- **Risk Profile**: Challenges and mitigation
- **Stakeholder Effects**: Who is affected and how

### 📊 Pattern Analysis
- Common themes across issues
- Emerging trends and directions
- Interconnections and dependencies

### 🎯 Recommendations
- Prioritized action items
- Resource allocation suggestions
- Timeline considerations

### 📈 Confidence Assessment
- Analysis reliability indicators
- Data quality notes
- Uncertainty factors""" + ("""

## 출력 형식

다음과 같이 분석을 구조화:

### 📋 경영진 요약
- 주요 발견사항 개요
- 중요한 통찰
- 주요 권장사항

### 🔍 상세 분석
각 주요 이슈 카테고리에 대해:
- **기술적 실현가능성**: 구현 평가
- **시장 영향**: 중요성과 범위
- **위험 프로필**: 도전과 완화
- **이해관계자 효과**: 누가 어떻게 영향받는지

### 📊 패턴 분석
- 이슈 전반의 공통 주제
- 새로운 트렌드와 방향
- 상호연결과 의존성

### 🎯 권장사항
- 우선순위가 정해진 실행 항목
- 자원 할당 제안
- 타임라인 고려사항

### 📈 신뢰도 평가
- 분석 신뢰도 지표
- 데이터 품질 노트
- 불확실성 요인""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
    
    def get_required_variables(self) -> List[str]:
        return [
            "issues_data",
            "analysis_scope",
            "focus_areas"
        ]
    
    def get_safety_constraints(self) -> List[str]:
        return [
            "Maintain strict objectivity and avoid personal bias",
            "Clearly distinguish between facts and opinions",
            "Consider multiple perspectives and viewpoints",
            "Flag potential misinformation or unreliable sources",
            "Focus on constructive analysis rather than criticism",
            "Respect privacy and confidentiality concerns",
            "Acknowledge limitations and uncertainties in data",
            "Provide balanced assessment of risks and opportunities"
        ]


class ReportGenerationPromptTemplate(BasePromptTemplate):
    """Template for report generation prompts."""
    
    def _initialize_sections(self):
        """Initialize report generation sections."""
        
        # Header section
        self.add_section(TemplateSection(
            name="header",
            content="""# Report Generation Task

You are a professional technical writer specializing in creating comprehensive, well-structured reports."""
        ))
        
        # Context section
        self.add_section(TemplateSection(
            name="context",
            content="""## Report Context

**Report Type:** {report_type}
**Target Audience:** {target_audience}
**Data Sources:** {data_sources}
**Report Scope:** {report_scope}
**Output Format:** {output_format}""" + ("""

## 보고서 맥락

**보고서 유형:** {report_type}
**대상 독자:** {target_audience}
**데이터 소스:** {data_sources}
**보고서 범위:** {report_scope}
**출력 형식:** {output_format}""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
        
        # Instructions section
        self.add_section(TemplateSection(
            name="instructions",
            content="""{safety_constraints}

## Report Generation Guidelines

1. **Structure**: Create logical, hierarchical organization
2. **Clarity**: Use clear, professional language
3. **Evidence**: Support all claims with data
4. **Completeness**: Address all required sections
5. **Accessibility**: Make content understandable to target audience""" + ("""

## 보고서 생성 가이드라인

1. **구조**: 논리적이고 계층적인 구성 만들기
2. **명확성**: 명확하고 전문적인 언어 사용
3. **증거**: 모든 주장을 데이터로 뒷받침
4. **완전성**: 필요한 모든 섹션 다루기
5. **접근성**: 대상 독자가 이해할 수 있도록 콘텐츠 작성""" if self.language == TemplateLanguage.BILINGUAL else "")
        ))
    
    def get_required_variables(self) -> List[str]:
        return [
            "report_type",
            "target_audience", 
            "data_sources",
            "report_scope"
        ]
    
    def get_safety_constraints(self) -> List[str]:
        return [
            "Ensure all information is accurate and verifiable",
            "Maintain professional and objective tone",
            "Respect confidentiality and privacy requirements",
            "Provide proper attribution for all sources",
            "Avoid speculation beyond available data",
            "Include appropriate disclaimers for limitations"
        ]


# Template factory for creating templates
class TemplateFactory:
    """Factory for creating prompt templates."""
    
    _templates = {
        "refactoring": RefactoringPromptTemplate,
        "keyword_generation": KeywordGenerationPromptTemplate,
        "issue_analysis": IssueAnalysisPromptTemplate,
        "report_generation": ReportGenerationPromptTemplate,
    }
    
    @classmethod
    def create(
        cls,
        template_type: str,
        language: TemplateLanguage = TemplateLanguage.ENGLISH,
        safety_level: SafetyLevel = SafetyLevel.STANDARD
    ) -> BasePromptTemplate:
        """Create a template instance."""
        template_class = cls._templates.get(template_type.lower())
        if not template_class:
            raise ValueError(f"Unknown template type: {template_type}")
        
        return template_class(language=language, safety_level=safety_level)
    
    @classmethod
    def register_template(cls, template_type: str, template_class):
        """Register a new template type."""
        cls._templates[template_type.lower()] = template_class
    
    @classmethod
    def list_available_templates(cls) -> List[str]:
        """List all available template types."""
        return list(cls._templates.keys())


# Export main classes and functions
__all__ = [
    'TemplateLanguage',
    'SafetyLevel', 
    'TemplateSection',
    'BasePromptTemplate',
    'RefactoringPromptTemplate',
    'KeywordGenerationPromptTemplate',
    'IssueAnalysisPromptTemplate',
    'ReportGenerationPromptTemplate',
    'TemplateFactory'
]