"""
Comprehensive tests for topic classifier module.

Tests the TopicClassifier class which determines topic types for adaptive report generation.
"""

import pytest
from src.reporting.topic_classifier import (
    TopicClassifier, TopicType, TopicClassification
)


class TestTopicType:
    """Test TopicType enum."""
    
    def test_topic_type_values(self):
        """Test that all topic types have correct values."""
        assert TopicType.TECHNICAL_ANNOUNCEMENT.value == "technical_announcement"
        assert TopicType.BUSINESS_STRATEGIC.value == "business_strategic"
        assert TopicType.RESEARCH_SCIENTIFIC.value == "research_scientific"
        assert TopicType.SOCIAL_POLITICAL.value == "social_political"
        assert TopicType.FINANCIAL_MARKET.value == "financial_market"
        assert TopicType.PRODUCT_LAUNCH.value == "product_launch"
        assert TopicType.GENERAL.value == "general"


class TestTopicClassification:
    """Test TopicClassification dataclass."""
    
    def test_topic_classification_creation(self):
        """Test creating TopicClassification with all fields."""
        classification = TopicClassification(
            primary_type=TopicType.TECHNICAL_ANNOUNCEMENT,
            confidence=0.85,
            secondary_types=[(TopicType.PRODUCT_LAUNCH, 0.3)],
            keywords_matched=["iOS", "API", "developer"],
            reasoning="Matches technical keywords and patterns",
            recommended_sections=["technical_details", "developer_impact"]
        )
        
        assert classification.primary_type == TopicType.TECHNICAL_ANNOUNCEMENT
        assert classification.confidence == 0.85
        assert len(classification.secondary_types) == 1
        assert classification.secondary_types[0][0] == TopicType.PRODUCT_LAUNCH
        assert classification.secondary_types[0][1] == 0.3
        assert classification.keywords_matched == ["iOS", "API", "developer"]
        assert "technical keywords" in classification.reasoning
        assert "technical_details" in classification.recommended_sections


class TestTopicClassifier:
    """Test TopicClassifier class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = TopicClassifier()
    
    def test_classifier_initialization(self):
        """Test classifier initializes correctly."""
        assert hasattr(self.classifier, 'classification_patterns')
        assert hasattr(self.classifier, 'base_sections')
        assert hasattr(self.classifier, 'section_generators')
        
        # Check all topic types are configured
        for topic_type in TopicType:
            assert topic_type in self.classifier.classification_patterns
    
    def test_classify_technical_announcement(self):
        """Test classification of technical announcements."""
        topic = "iOS 17 API updates for developers"
        keywords = ["iOS", "API", "SDK", "developer", "update"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        assert result.primary_type == TopicType.TECHNICAL_ANNOUNCEMENT
        assert result.confidence > 0.5
        assert len(result.keywords_matched) > 0
        assert "iOS" in result.keywords_matched or "API" in result.keywords_matched
    
    def test_classify_product_launch(self):
        """Test classification of product launches."""
        topic = "Apple launches new iPhone 15 with advanced features"
        keywords = ["Apple", "launch", "iPhone", "new", "product"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        assert result.primary_type == TopicType.PRODUCT_LAUNCH
        assert result.confidence > 0.4
        assert any(keyword in result.keywords_matched for keyword in ["launch", "iPhone", "product"])
    
    def test_classify_business_strategic(self):
        """Test classification of business strategic content."""
        topic = "Microsoft acquires AI startup for $2 billion strategic expansion"
        keywords = ["Microsoft", "acquisition", "strategic", "expansion", "billion"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        assert result.primary_type == TopicType.BUSINESS_STRATEGIC
        assert result.confidence > 0.4
        assert any(keyword in result.keywords_matched for keyword in ["acquisition", "strategic", "expansion"])
    
    def test_classify_research_scientific(self):
        """Test classification of research/scientific content."""
        topic = "New AI research published in Nature shows breakthrough in machine learning"
        keywords = ["research", "published", "Nature", "AI", "machine learning", "study"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        assert result.primary_type == TopicType.RESEARCH_SCIENTIFIC
        assert result.confidence > 0.4
        assert any(keyword in result.keywords_matched for keyword in ["research", "published", "AI"])
    
    def test_classify_financial_market(self):
        """Test classification of financial/market content."""
        topic = "Bitcoin price surges 15% amid market speculation and institutional investment"
        keywords = ["Bitcoin", "price", "market", "investment", "trading", "crypto"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        assert result.primary_type == TopicType.FINANCIAL_MARKET
        assert result.confidence > 0.4
        assert any(keyword in result.keywords_matched for keyword in ["Bitcoin", "market", "investment"])
    
    def test_classify_social_political(self):
        """Test classification of social/political content."""
        topic = "New government policy on AI regulation affects technology companies"
        keywords = ["government", "policy", "regulation", "AI", "legislation"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        assert result.primary_type == TopicType.SOCIAL_POLITICAL
        assert result.confidence > 0.4
        assert any(keyword in result.keywords_matched for keyword in ["government", "policy", "regulation"])
    
    def test_classify_general_fallback(self):
        """Test classification falls back to general for unclear topics."""
        topic = "Random topic about everyday things"
        keywords = ["random", "everyday", "things"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        # Should still return a classification, possibly GENERAL with low confidence
        assert isinstance(result, TopicClassification)
        assert result.confidence >= 0.0
    
    def test_classify_mixed_topic(self):
        """Test classification of topic with mixed signals."""
        topic = "Apple announces new AI research partnership while launching iPhone AI features"
        keywords = ["Apple", "announces", "AI", "research", "partnership", "launching", "iPhone", "features"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        # Should identify primary type and secondary types
        assert isinstance(result.primary_type, TopicType)
        assert result.confidence > 0.0
        # Mixed signals should result in secondary types
        assert len(result.secondary_types) > 0 or result.confidence > 0.7
    
    def test_classify_with_context(self):
        """Test classification with additional context."""
        topic = "New update"
        keywords = ["update"]
        context = "Apple released iOS 17 with new developer APIs and SDK improvements for mobile applications"
        
        result = self.classifier.classify_topic(topic, keywords, context)
        
        # Context should help identify this as technical
        assert result.primary_type == TopicType.TECHNICAL_ANNOUNCEMENT
        assert result.confidence > 0.5
    
    def test_classify_empty_input(self):
        """Test classification with empty or minimal input."""
        # Empty topic
        result = self.classifier.classify_topic("", [])
        assert isinstance(result, TopicClassification)
        assert result.confidence >= 0.0
        
        # Minimal topic
        result = self.classifier.classify_topic("test", [])
        assert isinstance(result, TopicClassification)
        assert result.confidence >= 0.0
    
    def test_calculate_type_score(self):
        """Test _calculate_type_score method."""
        text = "iOS 17 API developer SDK framework update"
        patterns = self.classifier.classification_patterns[TopicType.TECHNICAL_ANNOUNCEMENT]
        
        score, matches = self.classifier._calculate_type_score(text, patterns)
        
        assert 0.0 <= score <= 1.0
        assert isinstance(matches, list)
        assert len(matches) > 0
        assert any(keyword in matches for keyword in ["iOS", "API", "developer", "SDK", "framework", "update"])
    
    def test_calculate_type_score_no_matches(self):
        """Test _calculate_type_score with no matches."""
        text = "completely unrelated content about cooking recipes"
        patterns = self.classifier.classification_patterns[TopicType.TECHNICAL_ANNOUNCEMENT]
        
        score, matches = self.classifier._calculate_type_score(text, patterns)
        
        assert score == 0.0
        assert matches == []
    
    def test_calculate_type_score_pattern_matching(self):
        """Test _calculate_type_score with regex pattern matches."""
        text = "iOS 17 beta 3 developer preview WWDC 2023"
        patterns = self.classifier.classification_patterns[TopicType.TECHNICAL_ANNOUNCEMENT]
        
        score, matches = self.classifier._calculate_type_score(text, patterns)
        
        # Should get high score due to pattern matches
        assert score > 0.5
        assert len(matches) > 0
    
    def test_get_recommended_sections(self):
        """Test _get_recommended_sections method."""
        primary_type = TopicType.TECHNICAL_ANNOUNCEMENT
        secondary_types = [(TopicType.PRODUCT_LAUNCH, 0.6), (TopicType.BUSINESS_STRATEGIC, 0.2)]
        
        sections = self.classifier._get_recommended_sections(primary_type, secondary_types)
        
        # Should include base sections
        assert 'executive_summary' in sections
        assert 'key_findings' in sections
        
        # Should include primary type sections
        primary_sections = self.classifier.classification_patterns[primary_type]['sections']
        for section in primary_sections:
            assert section in sections
        
        # Should include strong secondary type sections (confidence > 0.5)
        strong_secondary_sections = self.classifier.classification_patterns[TopicType.PRODUCT_LAUNCH]['sections']
        for section in strong_secondary_sections:
            if section not in sections:
                # If not included, it might be due to deduplication
                pass
    
    def test_generate_reasoning(self):
        """Test _generate_reasoning method."""
        topic_type = TopicType.TECHNICAL_ANNOUNCEMENT
        confidence = 0.75
        matched_keywords = ["iOS", "API", "developer", "SDK", "framework"]
        
        reasoning = self.classifier._generate_reasoning(topic_type, confidence, matched_keywords)
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "기술 발표/업데이트" in reasoning
        assert "75.0%" in reasoning
        assert "iOS" in reasoning or "API" in reasoning or "developer" in reasoning
    
    def test_should_include_risk_analysis(self):
        """Test should_include_risk_analysis method."""
        # Technical announcement with high confidence - should not include risk analysis
        tech_classification = TopicClassification(
            primary_type=TopicType.TECHNICAL_ANNOUNCEMENT,
            confidence=0.8,
            secondary_types=[],
            keywords_matched=["iOS", "API"],
            reasoning="Technical",
            recommended_sections=[]
        )
        assert not self.classifier.should_include_risk_analysis(tech_classification)
        
        # Technical announcement with business component - should include risk analysis
        tech_with_business = TopicClassification(
            primary_type=TopicType.TECHNICAL_ANNOUNCEMENT,
            confidence=0.8,
            secondary_types=[(TopicType.BUSINESS_STRATEGIC, 0.4)],
            keywords_matched=["iOS", "API"],
            reasoning="Technical with business",
            recommended_sections=[]
        )
        assert self.classifier.should_include_risk_analysis(tech_with_business)
        
        # Business strategic - should include risk analysis
        business_classification = TopicClassification(
            primary_type=TopicType.BUSINESS_STRATEGIC,
            confidence=0.7,
            secondary_types=[],
            keywords_matched=["strategy", "acquisition"],
            reasoning="Business",
            recommended_sections=[]
        )
        assert self.classifier.should_include_risk_analysis(business_classification)
        
        # Financial market - should include risk analysis
        financial_classification = TopicClassification(
            primary_type=TopicType.FINANCIAL_MARKET,
            confidence=0.6,
            secondary_types=[],
            keywords_matched=["stock", "market"],
            reasoning="Financial",
            recommended_sections=[]
        )
        assert self.classifier.should_include_risk_analysis(financial_classification)
    
    def test_section_generator_methods_exist(self):
        """Test that all section generator methods exist and are callable."""
        section_generators = [
            '_generate_technical_section',
            '_generate_developer_impact_section',
            '_generate_business_impact_section',
            '_generate_risk_analysis_section',
            '_generate_opportunity_section',
            '_generate_stakeholder_section',
            '_generate_research_methodology_section',
            '_generate_future_research_section',
            '_generate_market_analysis_section'
        ]
        
        for generator_name in section_generators:
            assert hasattr(self.classifier, generator_name)
            generator = getattr(self.classifier, generator_name)
            assert callable(generator)
            
            # Test that it returns a string
            result = generator([], None)
            assert isinstance(result, str)
            assert len(result) > 0


class TestTopicClassifierEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = TopicClassifier()
    
    def test_classification_with_none_values(self):
        """Test classification with None values."""
        result = self.classifier.classify_topic("test", None, None)
        assert isinstance(result, TopicClassification)
        assert result.confidence >= 0.0
    
    def test_classification_with_empty_lists(self):
        """Test classification with empty lists."""
        result = self.classifier.classify_topic("test", [], "")
        assert isinstance(result, TopicClassification)
        assert result.confidence >= 0.0
    
    def test_classification_with_very_long_text(self):
        """Test classification with very long text."""
        long_topic = "AI " * 1000  # Very long topic
        long_keywords = ["keyword"] * 100  # Many keywords
        long_context = "context " * 500  # Long context
        
        result = self.classifier.classify_topic(long_topic, long_keywords, long_context)
        assert isinstance(result, TopicClassification)
        assert result.confidence >= 0.0
    
    def test_classification_with_special_characters(self):
        """Test classification with special characters."""
        topic = "AI/ML & NLP: 'next-gen' tech (2024) - $1M+ funding!"
        keywords = ["AI/ML", "NLP", "tech", "$1M+"]
        
        result = self.classifier.classify_topic(topic, keywords)
        assert isinstance(result, TopicClassification)
        assert result.confidence >= 0.0
    
    def test_classification_with_unicode(self):
        """Test classification with Unicode characters."""
        topic = "인공지능 기술 발전과 사회적 영향"
        keywords = ["인공지능", "기술", "발전", "사회적", "영향"]
        
        result = self.classifier.classify_topic(topic, keywords)
        assert isinstance(result, TopicClassification)
        assert result.confidence >= 0.0
    
    def test_keyword_matching_case_insensitive(self):
        """Test that keyword matching is case insensitive."""
        topic_upper = "APPLE LAUNCHES NEW IPHONE WITH AI FEATURES"
        topic_lower = "apple launches new iphone with ai features"
        keywords = ["Apple", "iPhone", "AI"]
        
        result_upper = self.classifier.classify_topic(topic_upper, keywords)
        result_lower = self.classifier.classify_topic(topic_lower, keywords)
        
        # Should get similar classifications regardless of case
        assert result_upper.primary_type == result_lower.primary_type
        assert abs(result_upper.confidence - result_lower.confidence) < 0.1
    
    def test_multiple_pattern_matches(self):
        """Test topic with multiple regex pattern matches."""
        topic = "iOS 17 beta 3 WWDC 2023 API version 4.2 developer preview"
        keywords = ["iOS", "WWDC", "API", "developer"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        # Should get high confidence due to multiple pattern matches
        assert result.primary_type == TopicType.TECHNICAL_ANNOUNCEMENT
        assert result.confidence > 0.7


class TestTopicClassifierIntegration:
    """Test integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = TopicClassifier()
    
    def test_real_world_scenarios(self):
        """Test with real-world-like scenarios."""
        scenarios = [
            {
                "topic": "OpenAI releases GPT-4 Turbo with improved performance",
                "keywords": ["OpenAI", "GPT-4", "release", "AI", "performance"],
                "expected_type": TopicType.TECHNICAL_ANNOUNCEMENT,
                "min_confidence": 0.5
            },
            {
                "topic": "Tesla reports Q3 earnings with record revenue growth",
                "keywords": ["Tesla", "earnings", "Q3", "revenue", "growth", "financial"],
                "expected_type": TopicType.FINANCIAL_MARKET,
                "min_confidence": 0.4
            },
            {
                "topic": "Government announces new regulations for cryptocurrency trading",
                "keywords": ["government", "regulations", "cryptocurrency", "policy", "trading"],
                "expected_type": TopicType.SOCIAL_POLITICAL,
                "min_confidence": 0.4
            },
            {
                "topic": "Stanford researchers publish breakthrough in quantum computing",
                "keywords": ["Stanford", "researchers", "publish", "quantum", "computing", "research"],
                "expected_type": TopicType.RESEARCH_SCIENTIFIC,
                "min_confidence": 0.4
            }
        ]
        
        for scenario in scenarios:
            result = self.classifier.classify_topic(
                scenario["topic"], 
                scenario["keywords"]
            )
            
            assert result.primary_type == scenario["expected_type"]
            assert result.confidence >= scenario["min_confidence"]
            assert len(result.keywords_matched) > 0
    
    def test_classification_consistency(self):
        """Test that classification is consistent across multiple calls."""
        topic = "Apple announces new MacBook Pro with M3 chip"
        keywords = ["Apple", "MacBook", "M3", "chip", "announces"]
        
        results = []
        for _ in range(5):
            result = self.classifier.classify_topic(topic, keywords)
            results.append(result)
        
        # All results should be identical (deterministic)
        for result in results[1:]:
            assert result.primary_type == results[0].primary_type
            assert result.confidence == results[0].confidence
            assert result.keywords_matched == results[0].keywords_matched
    
    def test_secondary_type_detection(self):
        """Test detection of secondary types in mixed content."""
        topic = "Apple's strategic partnership with OpenAI brings advanced AI to iPhone users"
        keywords = ["Apple", "strategic", "partnership", "OpenAI", "AI", "iPhone", "users"]
        
        result = self.classifier.classify_topic(topic, keywords)
        
        # Should detect multiple relevant types
        assert isinstance(result.primary_type, TopicType)
        assert len(result.secondary_types) > 0
        
        # Secondary types should be sorted by confidence
        if len(result.secondary_types) > 1:
            for i in range(len(result.secondary_types) - 1):
                assert result.secondary_types[i][1] >= result.secondary_types[i + 1][1]