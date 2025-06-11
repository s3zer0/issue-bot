"""
Search module for issue searching functionality.

Includes both basic issue searching and the enhanced multi-source intelligence
gathering system with 50-100 issue discovery capability, advanced analytics,
and continuous learning.
"""

# Basic/Legacy search
from .issue_searcher import IssueSearcher

# Enhanced search engine components
from .enhanced_search_engine import (
    EnhancedIssue,
    SourceManager,
    NewsSourceManager,
    AcademicSourceManager,
    SocialMediaManager,
    TechnicalSourceManager,
    FinancialSourceManager,
    AdvancedQueryGenerator,
    SemanticDeduplicator,
    EnhancedRelevanceScorer,
    SentimentAnalyzer
)

# Main enhanced searcher
from .enhanced_issue_searcher import (
    EnhancedIssueSearcher,
    DeepWebMiner,
    PredictiveAnalyzer,
    CrossReferenceValidator,
    PerformanceOptimizer
)

# Learning system
from .learning_system import (
    ContinuousLearningOrchestrator,
    FeedbackCollector,
    PerformanceAnalyzer,
    AdaptiveParameterTuner,
    ABTestManager,
    UserFeedback,
    SearchMetrics,
    LearningInsights
)

# Integration and orchestration
from .enhanced_search_integration import (
    HybridSearchOrchestrator,
    SearchModeSelector,
    SearchStrategy,
    SearchResultComparator,
    EnhancedSearchMixin,
    search_issues_intelligent,
    search_issues_enhanced_only
)

__all__ = [
    # Legacy
    'IssueSearcher',
    
    # Enhanced search engine
    'EnhancedIssue',
    'SourceManager',
    'NewsSourceManager',
    'AcademicSourceManager', 
    'SocialMediaManager',
    'TechnicalSourceManager',
    'FinancialSourceManager',
    'AdvancedQueryGenerator',
    'SemanticDeduplicator',
    'EnhancedRelevanceScorer',
    'SentimentAnalyzer',
    
    # Main enhanced searcher
    'EnhancedIssueSearcher',
    'DeepWebMiner',
    'PredictiveAnalyzer',
    'CrossReferenceValidator',
    'PerformanceOptimizer',
    
    # Learning system
    'ContinuousLearningOrchestrator',
    'FeedbackCollector',
    'PerformanceAnalyzer',
    'AdaptiveParameterTuner',
    'ABTestManager',
    'UserFeedback',
    'SearchMetrics',
    'LearningInsights',
    
    # Integration
    'HybridSearchOrchestrator',
    'SearchModeSelector',
    'SearchStrategy',
    'SearchResultComparator',
    'EnhancedSearchMixin',
    
    # Convenience functions
    'search_issues_intelligent',
    'search_issues_enhanced_only'
]