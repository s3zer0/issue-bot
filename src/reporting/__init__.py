"""
Reporting module for generating reports in various formats.

Includes both legacy reporting functions and the new adaptive reporting system
that dynamically adjusts content and format based on topic classification
and target audience.
"""

# Legacy reporting functions
from .reporting import (
    format_search_summary, 
    format_detailed_issue_report,
    create_detailed_report_from_search_result,
    save_report_to_file
)
from .pdf_report_generator import PDFReportGenerator

# New adaptive reporting system
from .adaptive_report_generator import (
    AdaptiveReportGenerator,
    TopicClassifier,
    DynamicSectionGenerator,
    TopicCategory,
    AudienceType,
    ContentComplexity,
    AdaptiveReportStructure
)
from .discord_formatter import (
    AdaptiveDiscordFormatter,
    DiscordEmbedBuilder,
    DiscordFileGenerator
)
from .adaptive_integration import (
    AdaptiveReportingOrchestrator,
    generate_adaptive_discord_report,
    generate_quick_adaptive_summary,
    AdaptiveReportingMixin
)

__all__ = [
    # Legacy functions
    'format_search_summary', 
    'format_detailed_issue_report',
    'create_detailed_report_from_search_result',
    'save_report_to_file',
    'PDFReportGenerator',
    
    # Adaptive reporting system
    'AdaptiveReportGenerator',
    'TopicClassifier',
    'DynamicSectionGenerator',
    'AdaptiveDiscordFormatter',
    'AdaptiveReportingOrchestrator',
    
    # Enums and data structures
    'TopicCategory',
    'AudienceType', 
    'ContentComplexity',
    'AdaptiveReportStructure',
    
    # Convenience functions
    'generate_adaptive_discord_report',
    'generate_quick_adaptive_summary',
    
    # Integration helpers
    'AdaptiveReportingMixin',
    'DiscordEmbedBuilder',
    'DiscordFileGenerator'
]