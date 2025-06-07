# src/reporting.py
"""
분석 결과 보고서 생성 및 파일 관리를 담당하는 모듈
"""
import os
from datetime import datetime
from src.models import SearchResult, IssueItem


def format_search_summary(result: SearchResult) -> str:
    """검색 결과를 Discord Embed에 들어갈 요약 문자열로 포맷팅합니다."""
    if result.total_found == 0:
        return f"**이슈 검색 실패**\n❌ '{', '.join(result.query_keywords)}' 관련 이슈를 찾지 못했습니다."

    summary = f"**이슈 검색 완료** ({result.total_found}개 발견)\n\n"
    issues_to_display = result.issues[:3]

    for i, issue in enumerate(issues_to_display, 1):
        summary += f"**{i}. {issue.title}**\n"
        summary += f"- 출처: {issue.source} | 관련도: {int(issue.relevance_score * 100)}%\n\n"

    if result.total_found > len(issues_to_display):
        remaining_count = result.total_found - len(issues_to_display)
        summary += f"**... 외 {remaining_count}개의 이슈가 더 있습니다.**\n"

    return summary


def format_detailed_issue_report(issue: IssueItem) -> str:
    """단일 이슈에 대한 상세 보고서 마크다운을 생성합니다."""
    report = f"# 📋 {issue.title}\n\n"
    report += f"**출처**: {issue.source or 'N/A'} | **발행일**: {issue.published_date or 'N/A'}\n"
    if issue.relevance_score is not None and issue.detail_confidence is not None:
        report += f"**관련도**: {int(issue.relevance_score * 100)}% | **세부신뢰도**: {int(issue.detail_confidence * 100)}%\n\n"

    report += f"## 📝 요약\n{issue.summary}\n\n"

    if issue.detailed_content:
        # 상세 내용과 배경 정보를 분리해서 표시
        content_to_display = issue.detailed_content
        if issue.background_context:
            content_to_display = content_to_display.replace(f"**배경 정보**:{issue.background_context}", "")
        report += f"## 📖 상세 내용\n{content_to_display.strip()}\n\n"

    if issue.background_context:
        report += f"## 🔗 배경 정보\n{issue.background_context}\n"

    return report


def create_detailed_report_from_search_result(search_result: SearchResult) -> str:
    """전체 검색 결과로부터 종합 보고서 마크다운을 생성합니다."""
    if not search_result.detailed_issues_count:
        return "상세 분석된 이슈가 없습니다."

    full_report = f"# 🔍 종합 이슈 분석 보고서\n- 키워드: {', '.join(search_result.query_keywords)}\n- 기간: {search_result.time_period}\n\n---\n"

    for issue in search_result.issues:
        if issue.detailed_content:
            full_report += format_detailed_issue_report(issue) + "\n---\n"

    return full_report


def save_report_to_file(report_content: str, topic: str) -> str:
    """보고서 내용을 파일로 저장하고 파일 경로를 반환합니다."""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    filename = f"report_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    file_path = os.path.join(reports_dir, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return file_path