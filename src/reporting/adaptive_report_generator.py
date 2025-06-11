"""
Adaptive Report Generation Module

A dynamic and intelligent report generation system that adapts structure,
content, and format based on topic classification, issue characteristics,
and target audience context.
"""

import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import openai
from loguru import logger

from src.models import SearchResult, IssueItem
from src.config import config


class TopicCategory(Enum):
    """Topic category classification."""
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    SOCIAL_POLITICAL = "social_political"
    SCIENTIFIC = "scientific"
    HEALTHCARE = "healthcare"
    ENVIRONMENT = "environment"
    ENTERTAINMENT = "entertainment"
    FINANCE = "finance"
    GENERAL = "general"


class ContentComplexity(Enum):
    """Content complexity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AudienceType(Enum):
    """Target audience types."""
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    GENERAL_PUBLIC = "general_public"
    RESEARCHER = "researcher"
    BUSINESS_ANALYST = "business_analyst"


@dataclass
class TopicClassification:
    """Topic classification result."""
    primary_category: TopicCategory
    secondary_categories: List[TopicCategory] = field(default_factory=list)
    complexity_level: ContentComplexity = ContentComplexity.INTERMEDIATE
    audience_type: AudienceType = AudienceType.GENERAL_PUBLIC
    time_sensitivity: float = 0.5  # 0.0 = not time-sensitive, 1.0 = highly urgent
    confidence: float = 0.0
    keywords_used: List[str] = field(default_factory=list)


@dataclass
class ReportSection:
    """Dynamic report section configuration."""
    name: str
    title: str
    content: str
    priority: int  # Lower number = higher priority
    required: bool = False
    visualization_type: Optional[str] = None  # chart, table, network, timeline
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveReportStructure:
    """Complete adaptive report structure."""
    title: str
    executive_summary: str
    sections: List[ReportSection]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    formatting_rules: Dict[str, Any] = field(default_factory=dict)


class TopicClassifier:
    """Intelligent topic classification system."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or config.get_openai_api_key()
        
        # Classification keywords for each category
        self.category_keywords = {
            TopicCategory.TECHNOLOGY: [
                'ai', 'artificial intelligence', 'machine learning', 'blockchain', 'cryptocurrency',
                'software', 'hardware', 'semiconductor', 'quantum computing', 'cloud computing',
                'cybersecurity', 'app', 'platform', 'algorithm', 'data science', 'iot', 'internet of things',
                'vr', 'ar', 'virtual reality', 'augmented reality', 'tech', 'startup', 'programming'
            ],
            TopicCategory.BUSINESS: [
                'market', 'revenue', 'profit', 'investment', 'stock', 'ipo', 'merger', 'acquisition',
                'business', 'company', 'corporate', 'strategy', 'management', 'leadership', 'economy',
                'trade', 'commerce', 'sales', 'marketing', 'brand', 'competition', 'industry'
            ],
            TopicCategory.SOCIAL_POLITICAL: [
                'politics', 'government', 'policy', 'law', 'regulation', 'social', 'society',
                'democracy', 'election', 'voting', 'rights', 'justice', 'protest', 'movement',
                'community', 'culture', 'diversity', 'equality', 'immigration', 'education'
            ],
            TopicCategory.SCIENTIFIC: [
                'research', 'study', 'experiment', 'discovery', 'science', 'biology', 'chemistry',
                'physics', 'mathematics', 'astronomy', 'genetics', 'molecular', 'clinical trial',
                'peer review', 'journal', 'publication', 'methodology', 'hypothesis', 'data'
            ],
            TopicCategory.HEALTHCARE: [
                'health', 'medical', 'hospital', 'doctor', 'patient', 'disease', 'treatment',
                'therapy', 'medicine', 'drug', 'vaccine', 'clinical', 'surgery', 'diagnosis',
                'healthcare', 'pharmaceutical', 'biotech', 'wellness', 'fitness'
            ],
            TopicCategory.FINANCE: [
                'finance', 'banking', 'credit', 'loan', 'mortgage', 'insurance', 'investment',
                'portfolio', 'trading', 'forex', 'commodity', 'bond', 'equity', 'derivatives',
                'fintech', 'cryptocurrency', 'blockchain', 'defi', 'payment'
            ],
            TopicCategory.ENVIRONMENT: [
                'environment', 'climate', 'sustainability', 'renewable energy', 'solar', 'wind',
                'carbon', 'emission', 'pollution', 'recycling', 'green', 'eco', 'conservation',
                'biodiversity', 'ecosystem', 'global warming', 'clean energy'
            ]
        }
    
    async def classify_topic(self, topic: str, issues: List[IssueItem]) -> TopicClassification:
        """Classify topic and determine report characteristics."""
        try:
            # Basic keyword-based classification
            primary_category = self._classify_by_keywords(topic, issues)
            
            # Enhanced classification using LLM
            enhanced_classification = await self._classify_with_llm(topic, issues, primary_category)
            
            return enhanced_classification
            
        except Exception as e:
            logger.error(f"Topic classification error: {e}")
            return self._fallback_classification(topic)
    
    def _classify_by_keywords(self, topic: str, issues: List[IssueItem]) -> TopicCategory:
        """Basic keyword-based classification."""
        text_to_analyze = topic.lower()
        
        # Add issue titles and summaries to analysis
        for issue in issues[:5]:  # Analyze first 5 issues
            text_to_analyze += " " + issue.title.lower() + " " + issue.summary.lower()
        
        scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_to_analyze)
            scores[category] = score
        
        # Return category with highest score, or GENERAL if no clear match
        if not scores or max(scores.values()) == 0:
            return TopicCategory.GENERAL
        
        return max(scores, key=scores.get)
    
    async def _classify_with_llm(
        self, 
        topic: str, 
        issues: List[IssueItem], 
        initial_category: TopicCategory
    ) -> TopicClassification:
        """Enhanced classification using LLM analysis."""
        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Prepare context from issues
            issue_context = ""
            if issues:
                issue_context = "\n".join([
                    f"- {issue.title}: {issue.summary[:200]}..."
                    for issue in issues[:3]
                ])
            
            prompt = f"""
Analyze the following topic and related issues to provide detailed classification:

Topic: "{topic}"

Related Issues:
{issue_context}

Please provide a JSON response with the following structure:
{{
    "primary_category": "one of: technology, business, social_political, scientific, healthcare, environment, entertainment, finance, general",
    "secondary_categories": ["list of relevant secondary categories"],
    "complexity_level": "one of: basic, intermediate, advanced, expert",
    "audience_type": "one of: executive, technical, general_public, researcher, business_analyst",
    "time_sensitivity": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of classification"
}}

Consider:
- Technical depth and terminology used
- Business vs. academic vs. general public focus
- Urgency and time-sensitive nature
- Target audience based on content complexity
"""

            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert content classifier. Analyze topics and provide structured classification data in JSON format."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse LLM response
            import json
            try:
                result = json.loads(response.choices[0].message.content)
                return TopicClassification(
                    primary_category=TopicCategory(result.get("primary_category", initial_category.value)),
                    secondary_categories=[
                        TopicCategory(cat) for cat in result.get("secondary_categories", [])
                        if cat in [c.value for c in TopicCategory]
                    ],
                    complexity_level=ContentComplexity(result.get("complexity_level", "intermediate")),
                    audience_type=AudienceType(result.get("audience_type", "general_public")),
                    time_sensitivity=float(result.get("time_sensitivity", 0.5)),
                    confidence=float(result.get("confidence", 0.8))
                )
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse LLM classification response: {e}")
                return self._fallback_classification(topic)
                
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return self._fallback_classification(topic)
    
    def _fallback_classification(self, topic: str) -> TopicClassification:
        """Fallback classification when LLM fails."""
        return TopicClassification(
            primary_category=TopicCategory.GENERAL,
            complexity_level=ContentComplexity.INTERMEDIATE,
            audience_type=AudienceType.GENERAL_PUBLIC,
            time_sensitivity=0.5,
            confidence=0.3
        )


class DynamicSectionGenerator:
    """Generate dynamic report sections based on topic classification."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or config.get_openai_api_key()
    
    async def generate_sections(
        self, 
        classification: TopicClassification,
        search_result: SearchResult
    ) -> List[ReportSection]:
        """Generate dynamic sections based on classification."""
        sections = []
        
        # Core sections (always included)
        sections.extend(await self._generate_core_sections(classification, search_result))
        
        # Category-specific sections
        if classification.primary_category == TopicCategory.TECHNOLOGY:
            sections.extend(await self._generate_tech_sections(search_result))
        elif classification.primary_category == TopicCategory.BUSINESS:
            sections.extend(await self._generate_business_sections(search_result))
        elif classification.primary_category == TopicCategory.SCIENTIFIC:
            sections.extend(await self._generate_scientific_sections(search_result))
        elif classification.primary_category == TopicCategory.SOCIAL_POLITICAL:
            sections.extend(await self._generate_social_sections(search_result))
        
        # Sort by priority and filter based on relevance
        sections.sort(key=lambda x: x.priority)
        return self._filter_relevant_sections(sections, classification)
    
    async def _generate_core_sections(
        self, 
        classification: TopicClassification, 
        search_result: SearchResult
    ) -> List[ReportSection]:
        """Generate core sections for all reports."""
        sections = []
        
        # Key Findings (always required)
        key_findings = await self._generate_key_findings(search_result, classification)
        sections.append(ReportSection(
            name="key_findings",
            title="🔍 주요 발견사항",
            content=key_findings,
            priority=1,
            required=True
        ))
        
        # Confidence and Verification (for low-confidence topics)
        if any(getattr(issue, 'combined_confidence', 0.5) < 0.7 for issue in search_result.issues):
            verification_content = self._generate_verification_section(search_result)
            sections.append(ReportSection(
                name="verification",
                title="✅ 신뢰도 및 검증",
                content=verification_content,
                priority=2,
                required=True
            ))
        
        return sections
    
    async def _generate_tech_sections(self, search_result: SearchResult) -> List[ReportSection]:
        """Generate technology-specific sections."""
        sections = []
        
        # Technical Specifications
        tech_specs = await self._analyze_technical_specifications(search_result)
        if tech_specs:
            sections.append(ReportSection(
                name="tech_specs",
                title="⚙️ 기술적 사양 및 분석",
                content=tech_specs,
                priority=3
            ))
        
        # Implementation Timeline
        timeline = self._generate_implementation_timeline(search_result)
        if timeline:
            sections.append(ReportSection(
                name="timeline",
                title="📅 구현 타임라인",
                content=timeline,
                priority=4,
                visualization_type="timeline"
            ))
        
        # Compatibility Analysis
        compatibility = self._analyze_compatibility(search_result)
        if compatibility:
            sections.append(ReportSection(
                name="compatibility",
                title="🔗 호환성 분석",
                content=compatibility,
                priority=5
            ))
        
        return sections
    
    async def _generate_business_sections(self, search_result: SearchResult) -> List[ReportSection]:
        """Generate business-specific sections."""
        sections = []
        
        # Market Impact Analysis
        market_impact = await self._analyze_market_impact(search_result)
        sections.append(ReportSection(
            name="market_impact",
            title="📈 시장 영향 분석",
            content=market_impact,
            priority=3,
            visualization_type="chart"
        ))
        
        # Competitor Analysis
        competitor_analysis = self._generate_competitor_analysis(search_result)
        if competitor_analysis:
            sections.append(ReportSection(
                name="competitors",
                title="🏢 경쟁사 분석",
                content=competitor_analysis,
                priority=4,
                visualization_type="table"
            ))
        
        # Financial Implications
        financial_impact = self._analyze_financial_implications(search_result)
        if financial_impact:
            sections.append(ReportSection(
                name="financial",
                title="💰 재무적 영향",
                content=financial_impact,
                priority=5
            ))
        
        return sections
    
    async def _generate_scientific_sections(self, search_result: SearchResult) -> List[ReportSection]:
        """Generate scientific research-specific sections."""
        sections = []
        
        # Methodology Assessment
        methodology = self._assess_research_methodology(search_result)
        if methodology:
            sections.append(ReportSection(
                name="methodology",
                title="🔬 연구 방법론 평가",
                content=methodology,
                priority=3
            ))
        
        # Peer Review Status
        peer_review = self._analyze_peer_review_status(search_result)
        if peer_review:
            sections.append(ReportSection(
                name="peer_review",
                title="📝 동료 검토 현황",
                content=peer_review,
                priority=4
            ))
        
        # Research Implications
        implications = await self._analyze_research_implications(search_result)
        sections.append(ReportSection(
            name="implications",
            title="🧬 연구 시사점",
            content=implications,
            priority=5
        ))
        
        return sections
    
    async def _generate_social_sections(self, search_result: SearchResult) -> List[ReportSection]:
        """Generate social/political-specific sections."""
        sections = []
        
        # Stakeholder Analysis
        stakeholders = self._analyze_stakeholders(search_result)
        if stakeholders:
            sections.append(ReportSection(
                name="stakeholders",
                title="👥 이해관계자 분석",
                content=stakeholders,
                priority=3,
                visualization_type="network"
            ))
        
        # Public Sentiment
        sentiment = await self._analyze_public_sentiment(search_result)
        if sentiment:
            sections.append(ReportSection(
                name="sentiment",
                title="💭 여론 분석",
                content=sentiment,
                priority=4
            ))
        
        # Policy Implications
        policy = self._analyze_policy_implications(search_result)
        if policy:
            sections.append(ReportSection(
                name="policy",
                title="📋 정책적 시사점",
                content=policy,
                priority=5
            ))
        
        return sections
    
    # Content generation methods
    async def _generate_key_findings(
        self, 
        search_result: SearchResult, 
        classification: TopicClassification
    ) -> str:
        """Generate key findings based on classification."""
        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            issues_summary = "\n".join([
                f"- {issue.title}: {issue.summary}"
                for issue in search_result.issues[:5]
            ])
            
            audience_context = self._get_audience_context(classification.audience_type)
            
            prompt = f"""
Based on the following issues, generate 3-5 key findings for a {classification.audience_type.value} audience:

Issues:
{issues_summary}

Audience Context: {audience_context}
Topic Category: {classification.primary_category.value}
Complexity Level: {classification.complexity_level.value}

Generate key findings as bullet points, focusing on:
- Most significant insights
- Actionable information
- Relevant to the target audience
- Clear and concise language

Format as markdown bullet points.
"""

            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert analyst creating targeted insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating key findings: {e}")
            return self._fallback_key_findings(search_result)
    
    def _generate_verification_section(self, search_result: SearchResult) -> str:
        """Generate verification and confidence section."""
        content = "### 검증 방법론\n\n"
        
        # Hallucination detection summary
        content += "이 보고서의 신뢰도 검증을 위해 다음 방법들이 사용되었습니다:\n\n"
        content += "- **LLM-as-a-Judge**: GPT-4o를 활용한 사실 정확성 및 논리적 일관성 검사\n"
        content += "- **RePPL 분석**: 텍스트 반복성, 퍼플렉시티, 의미적 엔트로피 분석\n"
        content += "- **자기 일관성 검사**: 다중 프롬프트를 통한 응답 일관성 검증\n\n"
        
        content += "### 이슈별 신뢰도\n\n"
        for i, issue in enumerate(search_result.issues[:5], 1):
            confidence = getattr(issue, 'combined_confidence', 0.5)
            confidence_text = "높음" if confidence > 0.8 else "보통" if confidence > 0.6 else "낮음"
            content += f"{i}. **{issue.title}** - 신뢰도: {confidence:.1%} ({confidence_text})\n"
        
        return content
    
    # Utility methods
    def _get_audience_context(self, audience_type: AudienceType) -> str:
        """Get context description for audience type."""
        contexts = {
            AudienceType.EXECUTIVE: "고위 경영진, 전략적 의사결정에 중점",
            AudienceType.TECHNICAL: "기술 전문가, 구현 세부사항에 중점",
            AudienceType.GENERAL_PUBLIC: "일반 대중, 이해하기 쉬운 설명에 중점",
            AudienceType.RESEARCHER: "연구자, 방법론과 데이터에 중점",
            AudienceType.BUSINESS_ANALYST: "비즈니스 분석가, 시장 영향에 중점"
        }
        return contexts.get(audience_type, "일반적인 관점")
    
    def _filter_relevant_sections(
        self, 
        sections: List[ReportSection], 
        classification: TopicClassification
    ) -> List[ReportSection]:
        """Filter sections based on relevance and importance."""
        # Always include required sections
        filtered = [s for s in sections if s.required]
        
        # Add optional sections based on priority and content length
        optional_sections = [s for s in sections if not s.required]
        
        # Limit total sections based on complexity
        max_sections = {
            ContentComplexity.BASIC: 5,
            ContentComplexity.INTERMEDIATE: 7,
            ContentComplexity.ADVANCED: 10,
            ContentComplexity.EXPERT: 12
        }.get(classification.complexity_level, 7)
        
        available_slots = max_sections - len(filtered)
        filtered.extend(optional_sections[:available_slots])
        
        return filtered
    
    def _fallback_key_findings(self, search_result: SearchResult) -> str:
        """Fallback key findings when LLM fails."""
        content = ""
        for i, issue in enumerate(search_result.issues[:3], 1):
            content += f"- **발견 {i}**: {issue.title}\n"
            content += f"  {issue.summary[:100]}...\n\n"
        return content
    
    # Placeholder methods for specific analysis types
    # These would be implemented with actual analysis logic
    
    async def _analyze_technical_specifications(self, search_result: SearchResult) -> Optional[str]:
        """Analyze technical specifications from issues."""
        # Implementation would analyze technical content
        return None
    
    def _generate_implementation_timeline(self, search_result: SearchResult) -> Optional[str]:
        """Generate implementation timeline."""
        # Implementation would extract timeline information
        return None
    
    def _analyze_compatibility(self, search_result: SearchResult) -> Optional[str]:
        """Analyze compatibility issues."""
        # Implementation would analyze compatibility information
        return None
    
    async def _analyze_market_impact(self, search_result: SearchResult) -> str:
        """Analyze market impact for business topics."""
        # Simplified implementation
        return "시장 영향 분석이 진행 중입니다."
    
    def _generate_competitor_analysis(self, search_result: SearchResult) -> Optional[str]:
        """Generate competitor analysis."""
        return None
    
    def _analyze_financial_implications(self, search_result: SearchResult) -> Optional[str]:
        """Analyze financial implications."""
        return None
    
    def _assess_research_methodology(self, search_result: SearchResult) -> Optional[str]:
        """Assess research methodology."""
        return None
    
    def _analyze_peer_review_status(self, search_result: SearchResult) -> Optional[str]:
        """Analyze peer review status."""
        return None
    
    async def _analyze_research_implications(self, search_result: SearchResult) -> str:
        """Analyze research implications."""
        return "연구 시사점 분석이 진행 중입니다."
    
    def _analyze_stakeholders(self, search_result: SearchResult) -> Optional[str]:
        """Analyze stakeholders."""
        return None
    
    async def _analyze_public_sentiment(self, search_result: SearchResult) -> Optional[str]:
        """Analyze public sentiment."""
        return None
    
    def _analyze_policy_implications(self, search_result: SearchResult) -> Optional[str]:
        """Analyze policy implications."""
        return None


class AdaptiveReportGenerator:
    """Main adaptive report generator orchestrating all components."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or config.get_openai_api_key()
        self.classifier = TopicClassifier(openai_api_key)
        self.section_generator = DynamicSectionGenerator(openai_api_key)
    
    async def generate_adaptive_report(
        self, 
        search_result: SearchResult,
        target_audience: Optional[AudienceType] = None
    ) -> AdaptiveReportStructure:
        """Generate a complete adaptive report."""
        try:
            # Step 1: Classify the topic
            classification = await self.classifier.classify_topic(
                search_result.topic, 
                search_result.issues
            )
            
            # Override audience if specified
            if target_audience:
                classification.audience_type = target_audience
            
            logger.info(f"Topic classified as: {classification.primary_category.value}, "
                       f"Audience: {classification.audience_type.value}")
            
            # Step 2: Generate adaptive sections
            sections = await self.section_generator.generate_sections(
                classification, 
                search_result
            )
            
            # Step 3: Generate executive summary
            executive_summary = await self._generate_executive_summary(
                search_result, 
                classification
            )
            
            # Step 4: Generate recommendations
            recommendations = await self._generate_recommendations(
                search_result, 
                classification
            )
            
            # Step 5: Determine formatting rules
            formatting_rules = self._determine_formatting_rules(classification)
            
            return AdaptiveReportStructure(
                title=self._generate_adaptive_title(search_result.topic, classification),
                executive_summary=executive_summary,
                sections=sections,
                recommendations=recommendations,
                metadata={
                    'classification': classification,
                    'generation_time': datetime.now(),
                    'total_issues': search_result.total_found,
                    'confidence_range': self._calculate_confidence_range(search_result)
                },
                formatting_rules=formatting_rules
            )
            
        except Exception as e:
            logger.error(f"Adaptive report generation failed: {e}")
            return await self._generate_fallback_report(search_result)
    
    async def _generate_executive_summary(
        self, 
        search_result: SearchResult, 
        classification: TopicClassification
    ) -> str:
        """Generate adaptive executive summary."""
        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Adjust summary length based on complexity and audience
            word_counts = {
                ContentComplexity.BASIC: "100-150",
                ContentComplexity.INTERMEDIATE: "150-250",
                ContentComplexity.ADVANCED: "250-350",
                ContentComplexity.EXPERT: "300-500"
            }
            
            target_length = word_counts.get(classification.complexity_level, "150-250")
            
            issues_context = "\n".join([
                f"- {issue.title}: {issue.summary}"
                for issue in search_result.issues[:5]
            ])
            
            prompt = f"""
Create an executive summary for a {classification.audience_type.value} audience about "{search_result.topic}".

Topic Category: {classification.primary_category.value}
Time Sensitivity: {"High" if classification.time_sensitivity > 0.7 else "Medium" if classification.time_sensitivity > 0.4 else "Low"}
Target Length: {target_length} words

Key Issues:
{issues_context}

Focus on:
- Most critical insights for {classification.audience_type.value}
- {classification.primary_category.value}-specific implications
- Actionable takeaways
- {"Urgent attention needed" if classification.time_sensitivity > 0.7 else "Strategic considerations"}

Write in Korean, using professional but accessible language.
"""

            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert report writer creating executive summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=600
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return f"{search_result.topic}에 대한 {search_result.total_found}개의 주요 이슈가 발견되었습니다."
    
    async def _generate_recommendations(
        self, 
        search_result: SearchResult, 
        classification: TopicClassification
    ) -> List[str]:
        """Generate adaptive recommendations."""
        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)
            
            # Customize recommendations based on category and audience
            recommendation_focus = {
                TopicCategory.TECHNOLOGY: "implementation, adoption, risks",
                TopicCategory.BUSINESS: "market opportunities, competitive positioning, investment",
                TopicCategory.SCIENTIFIC: "research validation, application potential, funding",
                TopicCategory.SOCIAL_POLITICAL: "stakeholder engagement, policy response, public relations"
            }.get(classification.primary_category, "general action items")
            
            issues_context = "\n".join([
                f"- {issue.title}: {issue.summary}"
                for issue in search_result.issues[:3]
            ])
            
            prompt = f"""
Generate 3-5 specific, actionable recommendations for a {classification.audience_type.value} regarding "{search_result.topic}".

Focus Areas: {recommendation_focus}
Time Sensitivity: {classification.time_sensitivity}
Issues Context:
{issues_context}

Each recommendation should be:
- Specific and actionable
- Relevant to {classification.audience_type.value}
- Prioritized by importance
- Include timeframe if time-sensitive

Format as a numbered list in Korean.
"""

            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a strategic advisor providing actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            # Parse recommendations into list
            content = response.choices[0].message.content
            recommendations = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and clean up
                    rec = line.split('.', 1)[-1].strip() if '.' in line else line.lstrip('- ')
                    if rec:
                        recommendations.append(rec)
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return [
                f"{search_result.topic} 관련 추가 모니터링 필요",
                "주요 이해관계자와의 커뮤니케이션 강화",
                "관련 정책 및 규제 변화 추적"
            ]
    
    def _generate_adaptive_title(self, topic: str, classification: TopicClassification) -> str:
        """Generate adaptive report title."""
        category_prefixes = {
            TopicCategory.TECHNOLOGY: "🚀 기술 동향",
            TopicCategory.BUSINESS: "📈 비즈니스 인사이트",
            TopicCategory.SCIENTIFIC: "🔬 연구 동향",
            TopicCategory.SOCIAL_POLITICAL: "🏛️ 사회·정치 분석",
            TopicCategory.HEALTHCARE: "🏥 헬스케어 동향",
            TopicCategory.FINANCE: "💰 금융 시장",
            TopicCategory.ENVIRONMENT: "🌍 환경 이슈",
        }
        
        prefix = category_prefixes.get(classification.primary_category, "📊 이슈")
        
        urgency = ""
        if classification.time_sensitivity > 0.8:
            urgency = " [긴급]"
        elif classification.time_sensitivity > 0.6:
            urgency = " [주의]"
        
        return f"{prefix}: {topic}{urgency}"
    
    def _determine_formatting_rules(self, classification: TopicClassification) -> Dict[str, Any]:
        """Determine formatting rules based on classification."""
        return {
            'use_technical_terms': classification.complexity_level in [ContentComplexity.ADVANCED, ContentComplexity.EXPERT],
            'include_charts': classification.primary_category in [TopicCategory.BUSINESS, TopicCategory.FINANCE],
            'emphasize_urgency': classification.time_sensitivity > 0.7,
            'detailed_verification': any([
                classification.primary_category == TopicCategory.SCIENTIFIC,
                classification.time_sensitivity > 0.8
            ]),
            'executive_focus': classification.audience_type == AudienceType.EXECUTIVE,
            'color_scheme': self._get_color_scheme(classification.primary_category)
        }
    
    def _get_color_scheme(self, category: TopicCategory) -> str:
        """Get color scheme for category."""
        schemes = {
            TopicCategory.TECHNOLOGY: "blue",
            TopicCategory.BUSINESS: "green",
            TopicCategory.SCIENTIFIC: "purple",
            TopicCategory.SOCIAL_POLITICAL: "orange",
            TopicCategory.HEALTHCARE: "red",
            TopicCategory.FINANCE: "gold",
            TopicCategory.ENVIRONMENT: "forest_green"
        }
        return schemes.get(category, "default")
    
    def _calculate_confidence_range(self, search_result: SearchResult) -> Dict[str, float]:
        """Calculate confidence range for issues."""
        confidences = [
            getattr(issue, 'combined_confidence', 0.5) 
            for issue in search_result.issues
        ]
        
        if not confidences:
            return {'min': 0.0, 'max': 0.0, 'avg': 0.0}
        
        return {
            'min': min(confidences),
            'max': max(confidences),
            'avg': sum(confidences) / len(confidences)
        }
    
    async def _generate_fallback_report(self, search_result: SearchResult) -> AdaptiveReportStructure:
        """Generate fallback report when adaptive generation fails."""
        return AdaptiveReportStructure(
            title=f"📊 이슈 분석: {search_result.topic}",
            executive_summary=f"{search_result.topic}에 대한 {search_result.total_found}개의 이슈가 발견되었습니다.",
            sections=[
                ReportSection(
                    name="basic_findings",
                    title="주요 발견사항",
                    content="\n".join([f"- {issue.title}" for issue in search_result.issues[:5]]),
                    priority=1,
                    required=True
                )
            ],
            recommendations=["추가 분석 필요", "지속적인 모니터링"],
            metadata={'fallback': True}
        )