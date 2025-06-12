"""
Dynamic topic classification for adaptive report generation.

This module analyzes topics to determine their type and generate appropriate
report sections based on the content and context.
"""

import re
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class TopicType(Enum):
    """Topic classification types."""
    TECHNICAL_ANNOUNCEMENT = "technical_announcement"
    BUSINESS_STRATEGIC = "business_strategic"
    RESEARCH_SCIENTIFIC = "research_scientific"
    SOCIAL_POLITICAL = "social_political"
    FINANCIAL_MARKET = "financial_market"
    PRODUCT_LAUNCH = "product_launch"
    GENERAL = "general"


@dataclass
class TopicClassification:
    """Result of topic classification."""
    primary_type: TopicType
    confidence: float
    secondary_types: List[Tuple[TopicType, float]]
    keywords_matched: List[str]
    reasoning: str
    recommended_sections: List[str]


class TopicClassifier:
    """
    Intelligent topic classifier that determines the best report structure
    based on topic content and keywords.
    """
    
    def __init__(self):
        """Initialize the topic classifier with keyword patterns."""
        self.classification_patterns = {
            TopicType.TECHNICAL_ANNOUNCEMENT: {
                'keywords': {
                    'high': ['iOS', 'WWDC', 'API', 'SDK', 'framework', 'release', 'beta', 'update', 
                            'version', 'developer', 'programming', 'code', 'software', 'app'],
                    'medium': ['tech', 'technology', 'platform', 'system', 'digital', 'web', 
                             'mobile', 'cloud', 'database', 'security'],
                    'low': ['announce', 'launch', 'new', 'feature', 'tool']
                },
                'patterns': [
                    r'iOS\s+\d+',
                    r'WWDC\s+20\d{2}',
                    r'API\s+\w+',
                    r'version\s+\d+\.\d+',
                    r'beta\s+\d+',
                    r'developer\s+preview'
                ],
                'sections': ['technical_details', 'developer_impact', 'compatibility']
            },
            
            TopicType.PRODUCT_LAUNCH: {
                'keywords': {
                    'high': ['launch', 'release', 'product', 'unveil', 'introduce', 'debut', 
                            'announcement', 'reveal', 'new', 'iPhone', 'iPad', 'MacBook'],
                    'medium': ['price', 'availability', 'features', 'specs', 'design', 'model'],
                    'low': ['brand', 'company', 'market', 'consumer']
                },
                'patterns': [
                    r'iPhone\s+\d+',
                    r'iPad\s+\w+',
                    r'MacBook\s+\w+',
                    r'launch\s+\w+',
                    r'price\s+\$\d+'
                ],
                'sections': ['product_features', 'market_positioning', 'pricing_strategy']
            },
            
            TopicType.BUSINESS_STRATEGIC: {
                'keywords': {
                    'high': ['strategy', 'merger', 'acquisition', 'partnership', 'investment', 
                            'expansion', 'restructure', 'CEO', 'revenue', 'profit', 'growth'],
                    'medium': ['business', 'company', 'corporation', 'market', 'industry', 
                             'competition', 'stakeholder', 'shareholder'],
                    'low': ['change', 'plan', 'future', 'vision', 'goal']
                },
                'patterns': [
                    r'\$\d+\s+(million|billion)',
                    r'Q\d\s+20\d{2}',
                    r'revenue\s+growth',
                    r'market\s+share'
                ],
                'sections': ['business_impact', 'financial_implications', 'risk_analysis', 
                           'opportunity_assessment', 'stakeholder_impact']
            },
            
            TopicType.RESEARCH_SCIENTIFIC: {
                'keywords': {
                    'high': ['research', 'study', 'scientific', 'paper', 'journal', 'experiment', 
                            'analysis', 'data', 'methodology', 'hypothesis', 'AI', 'ML'],
                    'medium': ['academic', 'university', 'laboratory', 'peer-review', 'findings', 
                             'discovery', 'innovation', 'breakthrough'],
                    'low': ['science', 'technology', 'knowledge', 'theory']
                },
                'patterns': [
                    r'published\s+in',
                    r'research\s+shows',
                    r'study\s+finds',
                    r'according\s+to\s+researchers'
                ],
                'sections': ['research_methodology', 'key_findings', 'implications', 
                           'future_research', 'practical_applications']
            },
            
            TopicType.SOCIAL_POLITICAL: {
                'keywords': {
                    'high': ['policy', 'government', 'politics', 'legislation', 'regulation', 
                            'social', 'community', 'public', 'citizen', 'democracy'],
                    'medium': ['society', 'culture', 'movement', 'campaign', 'advocacy', 
                             'rights', 'justice', 'equality'],
                    'low': ['people', 'group', 'organization', 'institution']
                },
                'patterns': [
                    r'bill\s+\w+',
                    r'law\s+passed',
                    r'government\s+\w+',
                    r'policy\s+change'
                ],
                'sections': ['social_impact', 'stakeholder_analysis', 'policy_implications', 
                           'community_response', 'long_term_effects']
            },
            
            TopicType.FINANCIAL_MARKET: {
                'keywords': {
                    'high': ['stock', 'market', 'trading', 'investment', 'financial', 'economy', 
                            'earnings', 'IPO', 'dividend', 'portfolio', 'crypto', 'bitcoin'],
                    'medium': ['bank', 'finance', 'money', 'capital', 'fund', 'asset', 
                             'commodity', 'currency'],
                    'low': ['price', 'value', 'cost', 'worth']
                },
                'patterns': [
                    r'\$\d+',
                    r'\d+%\s+(up|down|gain|loss)',
                    r'stock\s+price',
                    r'market\s+cap'
                ],
                'sections': ['market_analysis', 'financial_impact', 'investment_implications', 
                           'risk_assessment', 'economic_indicators']
            }
        }
        
        # Base sections that appear in all reports
        self.base_sections = ['executive_summary', 'key_findings']
        
        # Section generation templates
        self.section_generators = {
            'technical_details': self._generate_technical_section,
            'developer_impact': self._generate_developer_impact_section,
            'business_impact': self._generate_business_impact_section,
            'risk_analysis': self._generate_risk_analysis_section,
            'opportunity_assessment': self._generate_opportunity_section,
            'stakeholder_impact': self._generate_stakeholder_section,
            'research_methodology': self._generate_research_methodology_section,
            'future_research': self._generate_future_research_section,
            'market_analysis': self._generate_market_analysis_section
        }
        
        logger.info("Topic classifier initialized with 6 classification types")
    
    def classify_topic(self, topic: str, keywords: List[str] = None, 
                      context: str = None) -> TopicClassification:
        """
        Classify a topic and determine appropriate report sections.
        
        Args:
            topic: The main topic to classify
            keywords: Additional keywords from the search
            context: Additional context about the topic
            
        Returns:
            TopicClassification with type, confidence, and recommended sections
        """
        # Combine all text for analysis
        text_to_analyze = topic.lower()
        if keywords:
            text_to_analyze += " " + " ".join(keywords).lower()
        if context:
            text_to_analyze += " " + context.lower()
        
        # Score each topic type
        scores = {}
        matched_keywords = {}
        
        for topic_type, patterns in self.classification_patterns.items():
            score, matches = self._calculate_type_score(text_to_analyze, patterns)
            scores[topic_type] = score
            matched_keywords[topic_type] = matches
        
        # Find best match
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]
        
        # Get secondary types (confidence > 0.3)
        secondary_types = [
            (t, s) for t, s in scores.items() 
            if t != best_type and s > 0.3
        ]
        secondary_types.sort(key=lambda x: x[1], reverse=True)
        
        # Generate recommended sections
        recommended_sections = self._get_recommended_sections(best_type, secondary_types)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_type, confidence, matched_keywords[best_type])
        
        return TopicClassification(
            primary_type=best_type,
            confidence=confidence,
            secondary_types=secondary_types,
            keywords_matched=matched_keywords[best_type],
            reasoning=reasoning,
            recommended_sections=recommended_sections
        )
    
    def _calculate_type_score(self, text: str, patterns: Dict) -> Tuple[float, List[str]]:
        """Calculate confidence score for a topic type."""
        score = 0.0
        matched_keywords = []
        
        # Keyword matching with weights
        for weight_level, keywords in patterns['keywords'].items():
            weight = {'high': 3.0, 'medium': 2.0, 'low': 1.0}[weight_level]
            
            for keyword in keywords:
                if keyword.lower() in text:
                    score += weight
                    matched_keywords.append(keyword)
        
        # Pattern matching (regex patterns get higher weight)
        for pattern in patterns.get('patterns', []):
            if re.search(pattern, text, re.IGNORECASE):
                score += 5.0  # High weight for pattern matches
        
        # Normalize score (simple normalization)
        max_possible_score = (
            len(patterns['keywords'].get('high', [])) * 3.0 +
            len(patterns['keywords'].get('medium', [])) * 2.0 +
            len(patterns['keywords'].get('low', [])) * 1.0 +
            len(patterns.get('patterns', [])) * 5.0
        )
        
        if max_possible_score > 0:
            normalized_score = min(1.0, score / max_possible_score)
        else:
            normalized_score = 0.0
        
        return normalized_score, matched_keywords[:5]  # Limit matched keywords
    
    def _get_recommended_sections(self, primary_type: TopicType, 
                                secondary_types: List[Tuple[TopicType, float]]) -> List[str]:
        """Get recommended sections based on topic classification."""
        sections = self.base_sections.copy()
        
        # Add sections for primary type
        primary_sections = self.classification_patterns[primary_type]['sections']
        sections.extend(primary_sections)
        
        # Add sections for strong secondary types (confidence > 0.5)
        for sec_type, confidence in secondary_types:
            if confidence > 0.5:
                sec_sections = self.classification_patterns[sec_type]['sections']
                # Add only non-duplicate sections
                for section in sec_sections:
                    if section not in sections:
                        sections.append(section)
        
        return sections
    
    def _generate_reasoning(self, topic_type: TopicType, confidence: float, 
                          matched_keywords: List[str]) -> str:
        """Generate explanation for the classification."""
        type_descriptions = {
            TopicType.TECHNICAL_ANNOUNCEMENT: "기술 발표/업데이트",
            TopicType.PRODUCT_LAUNCH: "제품 출시",
            TopicType.BUSINESS_STRATEGIC: "비즈니스/전략",
            TopicType.RESEARCH_SCIENTIFIC: "연구/과학",
            TopicType.SOCIAL_POLITICAL: "사회/정치",
            TopicType.FINANCIAL_MARKET: "금융/시장",
            TopicType.GENERAL: "일반"
        }
        
        description = type_descriptions.get(topic_type, "일반")
        keywords_text = ", ".join(matched_keywords[:3]) if matched_keywords else "없음"
        
        return (f"주제가 '{description}' 유형으로 분류됨 "
                f"(신뢰도: {confidence:.1%}). "
                f"매칭 키워드: {keywords_text}")
    
    def should_include_risk_analysis(self, classification: TopicClassification) -> bool:
        """Determine if risk analysis section should be included."""
        # Skip risk analysis for pure technical announcements
        if (classification.primary_type == TopicType.TECHNICAL_ANNOUNCEMENT and 
            classification.confidence > 0.7):
            # Check if there are no significant secondary types that need risk analysis
            business_risk_types = {TopicType.BUSINESS_STRATEGIC, TopicType.FINANCIAL_MARKET}
            has_business_component = any(
                t in business_risk_types and conf > 0.3 
                for t, conf in classification.secondary_types
            )
            return has_business_component
        
        # Include risk analysis for most other types
        return classification.primary_type in {
            TopicType.BUSINESS_STRATEGIC,
            TopicType.FINANCIAL_MARKET,
            TopicType.SOCIAL_POLITICAL,
            TopicType.RESEARCH_SCIENTIFIC  # Research implications
        }
    
    # Section generation methods (simplified implementations)
    def _generate_technical_section(self, issues, classification) -> str:
        return "기술적 세부사항 및 구현 관련 정보를 분석합니다."
    
    def _generate_developer_impact_section(self, issues, classification) -> str:
        return "개발자 커뮤니티 및 개발 생태계에 미치는 영향을 평가합니다."
    
    def _generate_business_impact_section(self, issues, classification) -> str:
        return "비즈니스 측면에서의 영향과 시장 변화를 분석합니다."
    
    def _generate_risk_analysis_section(self, issues, classification) -> str:
        return "잠재적 위험 요소와 도전 과제를 식별하고 분석합니다."
    
    def _generate_opportunity_section(self, issues, classification) -> str:
        return "새로운 기회와 성장 가능성을 탐색합니다."
    
    def _generate_stakeholder_section(self, issues, classification) -> str:
        return "주요 이해관계자들에게 미치는 영향을 분석합니다."
    
    def _generate_research_methodology_section(self, issues, classification) -> str:
        return "연구 방법론과 데이터 수집 과정을 설명합니다."
    
    def _generate_future_research_section(self, issues, classification) -> str:
        return "향후 연구 방향과 발전 가능성을 제시합니다."
    
    def _generate_market_analysis_section(self, issues, classification) -> str:
        return "시장 동향과 경제적 영향을 분석합니다."


# Global classifier instance
topic_classifier = TopicClassifier()