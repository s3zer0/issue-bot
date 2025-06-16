"""
Comprehensive tests for data models in src/models.py.

This test suite ensures 100% coverage of all data model classes and their functionality,
including edge cases, validation, and proper data structure handling.
"""

import pytest
from dataclasses import FrozenInstanceError
from typing import List, Optional, Any, Dict

from src.models import (
    KeywordResult, RePPLScore, IssueItem, SearchResult
)


class TestKeywordResult:
    """Test KeywordResult data class."""
    
    def test_keyword_result_creation_with_all_fields(self):
        """Test creating KeywordResult with all required fields."""
        result = KeywordResult(
            topic="AI Ethics",
            primary_keywords=["ethics", "AI", "machine learning"],
            related_terms=["bias", "fairness", "transparency"],
            context_keywords=["technology", "society", "governance"],
            confidence_score=0.85,
            generation_time=2.5,
            raw_response="Mock LLM response",
            trusted_domains=["nature.com", "ieee.org"]
        )
        
        assert result.topic == "AI Ethics"
        assert result.primary_keywords == ["ethics", "AI", "machine learning"]
        assert result.related_terms == ["bias", "fairness", "transparency"]
        assert result.context_keywords == ["technology", "society", "governance"]
        assert result.confidence_score == 0.85
        assert result.generation_time == 2.5
        assert result.raw_response == "Mock LLM response"
        assert result.trusted_domains == ["nature.com", "ieee.org"]
    
    def test_keyword_result_creation_without_trusted_domains(self):
        """Test creating KeywordResult without trusted_domains uses default."""
        result = KeywordResult(
            topic="Quantum Computing",
            primary_keywords=["quantum", "computing"],
            related_terms=["qubits", "superposition"],
            context_keywords=["physics", "technology"],
            confidence_score=0.9,
            generation_time=1.8,
            raw_response="Mock response"
        )
        
        assert result.trusted_domains == []  # Default factory
    
    def test_keyword_result_empty_lists(self):
        """Test KeywordResult with empty keyword lists."""
        result = KeywordResult(
            topic="Empty Test",
            primary_keywords=[],
            related_terms=[],
            context_keywords=[],
            confidence_score=0.0,
            generation_time=0.1,
            raw_response="Empty response"
        )
        
        assert result.primary_keywords == []
        assert result.related_terms == []
        assert result.context_keywords == []
    
    def test_keyword_result_edge_confidence_values(self):
        """Test KeywordResult with edge confidence values."""
        # Minimum confidence
        result_min = KeywordResult(
            topic="Test",
            primary_keywords=["test"],
            related_terms=["testing"],
            context_keywords=["quality"],
            confidence_score=0.0,
            generation_time=0.1,
            raw_response="Min confidence"
        )
        assert result_min.confidence_score == 0.0
        
        # Maximum confidence
        result_max = KeywordResult(
            topic="Test",
            primary_keywords=["test"],
            related_terms=["testing"],
            context_keywords=["quality"],
            confidence_score=1.0,
            generation_time=0.1,
            raw_response="Max confidence"
        )
        assert result_max.confidence_score == 1.0


class TestRePPLScore:
    """Test RePPLScore data class."""
    
    def test_reppl_score_creation_complete(self):
        """Test creating RePPLScore with all fields."""
        score = RePPLScore(
            repetition_score=0.3,
            perplexity=45.2,
            semantic_entropy=0.7,
            confidence=0.85,
            repeated_phrases=["the same phrase", "repeated content"],
            analysis_details={
                "model": "gpt-4",
                "tokens_analyzed": 500,
                "processing_time": 1.2
            }
        )
        
        assert score.repetition_score == 0.3
        assert score.perplexity == 45.2
        assert score.semantic_entropy == 0.7
        assert score.confidence == 0.85
        assert score.repeated_phrases == ["the same phrase", "repeated content"]
        assert score.analysis_details["model"] == "gpt-4"
        assert score.analysis_details["tokens_analyzed"] == 500
    
    def test_reppl_score_empty_repeated_phrases(self):
        """Test RePPLScore with no repeated phrases."""
        score = RePPLScore(
            repetition_score=0.1,
            perplexity=25.0,
            semantic_entropy=0.2,
            confidence=0.95,
            repeated_phrases=[],
            analysis_details={}
        )
        
        assert score.repeated_phrases == []
        assert score.analysis_details == {}
    
    def test_reppl_score_edge_values(self):
        """Test RePPLScore with edge numerical values."""
        score = RePPLScore(
            repetition_score=0.0,
            perplexity=0.1,
            semantic_entropy=1.0,
            confidence=1.0,
            repeated_phrases=["single phrase"],
            analysis_details={"status": "edge_case"}
        )
        
        assert score.repetition_score == 0.0
        assert score.perplexity == 0.1
        assert score.semantic_entropy == 1.0
        assert score.confidence == 1.0


class TestIssueItem:
    """Test IssueItem data class."""
    
    def test_issue_item_required_fields_only(self):
        """Test creating IssueItem with only required fields."""
        issue = IssueItem(
            title="Test Issue",
            summary="This is a test issue summary",
            source="test_source",
            published_date="2024-01-15",
            relevance_score=0.8,
            category="technology",
            content_snippet="Brief content snippet"
        )
        
        assert issue.title == "Test Issue"
        assert issue.summary == "This is a test issue summary"
        assert issue.source == "test_source"
        assert issue.published_date == "2024-01-15"
        assert issue.relevance_score == 0.8
        assert issue.category == "technology"
        assert issue.content_snippet == "Brief content snippet"
        
        # Check optional fields are None by default
        assert issue.technical_core is None
        assert issue.importance is None
        assert issue.related_keywords is None
        assert issue.technical_analysis is None
        assert issue.detailed_content is None
        assert issue.background_context is None
        assert issue.impact_analysis is None
        assert issue.detail_collection_time is None
        assert issue.detail_confidence is None
    
    def test_issue_item_all_fields(self):
        """Test creating IssueItem with all fields populated."""
        issue = IssueItem(
            title="Complete Test Issue",
            summary="Comprehensive test summary",
            source="comprehensive_source",
            published_date="2024-01-20",
            relevance_score=0.95,
            category="research",
            content_snippet="Detailed snippet",
            technical_core="Core technical content",
            importance="Critical",
            related_keywords="AI, ML, deep learning",
            technical_analysis="Technical analysis details",
            detailed_content="Full detailed content here",
            background_context="Background and context information",
            impact_analysis={"business_impact": "High", "social_impact": "Medium"},
            detail_collection_time=3.5,
            detail_confidence=0.92
        )
        
        assert issue.technical_core == "Core technical content"
        assert issue.importance == "Critical"
        assert issue.related_keywords == "AI, ML, deep learning"
        assert issue.technical_analysis == "Technical analysis details"
        assert issue.detailed_content == "Full detailed content here"
        assert issue.background_context == "Background and context information"
        assert issue.impact_analysis == {"business_impact": "High", "social_impact": "Medium"}
        assert issue.detail_collection_time == 3.5
        assert issue.detail_confidence == 0.92
    
    def test_issue_item_none_published_date(self):
        """Test IssueItem with None published_date."""
        issue = IssueItem(
            title="No Date Issue",
            summary="Issue without publication date",
            source="unknown_source",
            published_date=None,
            relevance_score=0.6,
            category="general",
            content_snippet="No date snippet"
        )
        
        assert issue.published_date is None
    
    def test_issue_item_zero_relevance_score(self):
        """Test IssueItem with minimum relevance score."""
        issue = IssueItem(
            title="Low Relevance Issue",
            summary="Issue with low relevance",
            source="low_relevance_source",
            published_date="2024-01-01",
            relevance_score=0.0,
            category="misc",
            content_snippet="Low relevance snippet"
        )
        
        assert issue.relevance_score == 0.0
    
    def test_issue_item_maximum_relevance_score(self):
        """Test IssueItem with maximum relevance score."""
        issue = IssueItem(
            title="High Relevance Issue",
            summary="Issue with maximum relevance",
            source="high_relevance_source",
            published_date="2024-01-01",
            relevance_score=1.0,
            category="priority",
            content_snippet="High relevance snippet"
        )
        
        assert issue.relevance_score == 1.0
    
    def test_issue_item_empty_strings(self):
        """Test IssueItem with empty string values."""
        issue = IssueItem(
            title="",
            summary="",
            source="",
            published_date="",
            relevance_score=0.5,
            category="",
            content_snippet=""
        )
        
        assert issue.title == ""
        assert issue.summary == ""
        assert issue.source == ""
        assert issue.published_date == ""
        assert issue.category == ""
        assert issue.content_snippet == ""


class TestSearchResult:
    """Test SearchResult data class."""
    
    def test_search_result_minimal(self):
        """Test creating SearchResult with minimal required data."""
        issues = [
            IssueItem(
                title="Test Issue 1",
                summary="Summary 1",
                source="source1",
                published_date="2024-01-01",
                relevance_score=0.8,
                category="tech",
                content_snippet="Snippet 1"
            )
        ]
        
        result = SearchResult(
            query_keywords=["AI", "ethics"],
            total_found=1,
            issues=issues,
            search_time=2.5,
            api_calls_used=3,
            confidence_score=0.85,
            time_period="최근 1주일",
            raw_responses=["Response 1", "Response 2"]
        )
        
        assert result.query_keywords == ["AI", "ethics"]
        assert result.total_found == 1
        assert len(result.issues) == 1
        assert result.search_time == 2.5
        assert result.api_calls_used == 3
        assert result.confidence_score == 0.85
        assert result.time_period == "최근 1주일"
        assert result.raw_responses == ["Response 1", "Response 2"]
        
        # Check default values
        assert result.detailed_issues_count == 0
        assert result.total_detail_collection_time == 0.0
        assert result.average_detail_confidence == 0.0
    
    def test_search_result_with_detailed_stats(self):
        """Test SearchResult with detailed collection statistics."""
        issues = [
            IssueItem(
                title="Detailed Issue",
                summary="Detailed summary",
                source="detailed_source",
                published_date="2024-01-15",
                relevance_score=0.9,
                category="research",
                content_snippet="Detailed snippet"
            )
        ]
        
        result = SearchResult(
            query_keywords=["quantum", "computing"],
            total_found=1,
            issues=issues,
            search_time=5.2,
            api_calls_used=8,
            confidence_score=0.92,
            time_period="최근 2주일",
            raw_responses=["Detailed response"],
            detailed_issues_count=1,
            total_detail_collection_time=3.7,
            average_detail_confidence=0.88
        )
        
        assert result.detailed_issues_count == 1
        assert result.total_detail_collection_time == 3.7
        assert result.average_detail_confidence == 0.88
    
    def test_search_result_empty_issues(self):
        """Test SearchResult with no issues found."""
        result = SearchResult(
            query_keywords=["nonexistent", "topic"],
            total_found=0,
            issues=[],
            search_time=1.0,
            api_calls_used=2,
            confidence_score=0.0,
            time_period="최근 1일",
            raw_responses=["No results found"]
        )
        
        assert result.total_found == 0
        assert result.issues == []
        assert result.confidence_score == 0.0
    
    def test_search_result_multiple_issues(self):
        """Test SearchResult with multiple issues."""
        issues = [
            IssueItem(
                title=f"Issue {i}",
                summary=f"Summary {i}",
                source=f"source{i}",
                published_date=f"2024-01-{i:02d}",
                relevance_score=0.5 + (i * 0.1),
                category=f"cat{i}",
                content_snippet=f"Snippet {i}"
            )
            for i in range(1, 6)  # 5 issues
        ]
        
        result = SearchResult(
            query_keywords=["multiple", "issues", "test"],
            total_found=5,
            issues=issues,
            search_time=8.3,
            api_calls_used=12,
            confidence_score=0.78,
            time_period="최근 1개월",
            raw_responses=[f"Response {i}" for i in range(1, 6)]
        )
        
        assert len(result.issues) == 5
        assert result.total_found == 5
        assert len(result.raw_responses) == 5
        assert result.issues[0].title == "Issue 1"
        assert result.issues[4].title == "Issue 5"
    
    def test_search_result_edge_values(self):
        """Test SearchResult with edge case values."""
        result = SearchResult(
            query_keywords=[],
            total_found=0,
            issues=[],
            search_time=0.0,
            api_calls_used=0,
            confidence_score=0.0,
            time_period="",
            raw_responses=[]
        )
        
        assert result.query_keywords == []
        assert result.total_found == 0
        assert result.issues == []
        assert result.search_time == 0.0
        assert result.api_calls_used == 0
        assert result.confidence_score == 0.0
        assert result.time_period == ""
        assert result.raw_responses == []


class TestDataModelIntegration:
    """Test integration between different data models."""
    
    def test_complete_workflow_data_flow(self):
        """Test complete data flow from keywords to search results."""
        # 1. Create KeywordResult
        keyword_result = KeywordResult(
            topic="AI Safety",
            primary_keywords=["AI safety", "alignment", "control"],
            related_terms=["AGI", "superintelligence", "value alignment"],
            context_keywords=["ethics", "risk", "governance"],
            confidence_score=0.9,
            generation_time=2.1,
            raw_response="Generated AI safety keywords"
        )
        
        # 2. Create IssueItems with hallucination detection results
        issue1 = IssueItem(
            title="AI Alignment Research Progress",
            summary="Recent advances in AI alignment research",
            source="AI Research Journal",
            published_date="2024-01-20",
            relevance_score=0.95,
            category="research",
            content_snippet="Key developments in alignment..."
        )
        
        # Simulate adding hallucination detection results
        setattr(issue1, 'reppl_score', RePPLScore(
            repetition_score=0.1,
            perplexity=28.5,
            semantic_entropy=0.3,
            confidence=0.92,
            repeated_phrases=[],
            analysis_details={"source": "reppl_detector"}
        ))
        setattr(issue1, 'combined_confidence', 0.88)
        
        # 3. Create SearchResult
        search_result = SearchResult(
            query_keywords=keyword_result.primary_keywords,
            total_found=1,
            issues=[issue1],
            search_time=4.2,
            api_calls_used=5,
            confidence_score=0.88,
            time_period="최근 1주일",
            raw_responses=["Search response data"]
        )
        
        # Verify integration
        assert search_result.query_keywords == ["AI safety", "alignment", "control"]
        assert len(search_result.issues) == 1
        assert hasattr(search_result.issues[0], 'reppl_score')
        assert hasattr(search_result.issues[0], 'combined_confidence')
        assert search_result.issues[0].combined_confidence == 0.88
    
    def test_dynamic_attribute_assignment(self):
        """Test that IssueItem supports dynamic attribute assignment for hallucination detection."""
        issue = IssueItem(
            title="Dynamic Test",
            summary="Test dynamic attributes",
            source="test",
            published_date="2024-01-01",
            relevance_score=0.7,
            category="test",
            content_snippet="Test snippet"
        )
        
        # Add dynamic attributes (as mentioned in models.py docstring)
        setattr(issue, 'consistency_score', 0.85)
        setattr(issue, 'llm_judge_score', 0.78)
        setattr(issue, 'combined_confidence', 0.81)
        setattr(issue, 'hallucination_flags', ['minor_inconsistency'])
        
        assert hasattr(issue, 'consistency_score')
        assert hasattr(issue, 'llm_judge_score')
        assert hasattr(issue, 'combined_confidence')
        assert hasattr(issue, 'hallucination_flags')
        assert issue.consistency_score == 0.85
        assert issue.llm_judge_score == 0.78
        assert issue.combined_confidence == 0.81
        assert issue.hallucination_flags == ['minor_inconsistency']


class TestDataModelValidation:
    """Test data validation and constraints."""
    
    def test_keyword_result_type_validation(self):
        """Test that KeywordResult enforces expected types."""
        # This would normally be enforced by type hints in production
        result = KeywordResult(
            topic="Type Test",
            primary_keywords=["keyword1", "keyword2"],
            related_terms=["term1", "term2"],
            context_keywords=["context1", "context2"],
            confidence_score=0.8,
            generation_time=1.5,
            raw_response="Type validation test"
        )
        
        assert isinstance(result.topic, str)
        assert isinstance(result.primary_keywords, list)
        assert isinstance(result.related_terms, list)
        assert isinstance(result.context_keywords, list)
        assert isinstance(result.confidence_score, (int, float))
        assert isinstance(result.generation_time, (int, float))
        assert isinstance(result.raw_response, str)
    
    def test_search_result_consistency(self):
        """Test SearchResult internal consistency."""
        issues = [
            IssueItem(
                title=f"Issue {i}",
                summary=f"Summary {i}",
                source="test",
                published_date="2024-01-01",
                relevance_score=0.8,
                category="test",
                content_snippet="snippet"
            )
            for i in range(3)
        ]
        
        result = SearchResult(
            query_keywords=["test"],
            total_found=3,
            issues=issues,
            search_time=1.0,
            api_calls_used=1,
            confidence_score=0.8,
            time_period="test",
            raw_responses=["response"]
        )
        
        # Check consistency between total_found and actual issues count
        assert result.total_found == len(result.issues)