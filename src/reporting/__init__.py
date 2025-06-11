"""
Reporting module for generating reports in various formats.
"""

from .reporting import (
    format_search_summary, 
    format_detailed_issue_report,
    create_detailed_report_from_search_result,
    save_report_to_file
)
from .pdf_report_generator import PDFReportGenerator

__all__ = [
    'format_search_summary', 
    'format_detailed_issue_report',
    'create_detailed_report_from_search_result',
    'save_report_to_file',
    'PDFReportGenerator'
]