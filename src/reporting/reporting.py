"""
분석 결과 보고서 생성 및 파일 관리를 담당하는 모듈.

이슈 검색 결과를 마크다운 형식의 보고서로 변환하고, 파일로 저장하는 기능을 제공합니다.
보고서는 Discord Embed와 상세 보고서로 활용되며, 파일 저장을 통해 기록을 관리합니다.
"""

import os
from datetime import datetime
from src.models import SearchResult, IssueItem


def format_search_summary(result: SearchResult) -> str:
    """검색 결과를 Discord Embed에 적합한 요약 문자열로 포맷팅합니다.

    최대 3개의 이슈를 간략히 표시하며, 전체 이슈 개수와 관련 메타데이터를 포함합니다.

    Args:
        result (SearchResult): 포맷팅할 검색 결과 객체.

    Returns:
        str: Discord Embed에 삽입할 수 있는 마크다운 형식의 요약 문자열.

    Notes:
        - 이슈가 없을 경우 실패 메시지를 반환.
        - RePPL 신뢰도(`reppl_confidence`)는 동적으로 추가된 필드이므로 getattr로 안전하게 접근.
    """
    if result.total_found == 0:
        return f"**이슈 검색 실패**\n❌ 신뢰도 높은 관련 이슈를 찾지 못했습니다."

    summary = f"**검증된 이슈 발견** ({result.total_found}개)\n\n"
    issues_to_display = result.issues[:3]  # 최대 3개 이슈만 표시

    for i, issue in enumerate(issues_to_display, 1):
        summary += f"**{i}. {issue.title}**\n"
        reppl_conf = getattr(issue, 'reppl_confidence', None)
        conf_text = f" | RePPL 신뢰도: {int(reppl_conf * 100)}%" if reppl_conf else ""
        summary += f"- 출처: {issue.source} | 관련도: {int(issue.relevance_score * 100)}%{conf_text}\n\n"

    if result.total_found > len(issues_to_display):
        remaining_count = result.total_found - len(issues_to_display)
        summary += f"**... 외 {remaining_count}개의 이슈가 더 있습니다.**\n"

    return summary


def format_detailed_issue_report(issue: IssueItem) -> str:
    """단일 이슈에 대한 상세 보고서를 마크다운 형식으로 생성합니다.

    이슈의 제목, 요약, 출처, 상세 내용, 배경 정보 등을 체계적으로 포함합니다.

    Args:
        issue (IssueItem): 포맷팅할 이슈 객체.

    Returns:
        str: 마크다운 형식의 상세 이슈 보고서 문자열.

    Notes:
        - RePPL 신뢰도(`reppl_confidence`)는 동적으로 추가된 필드이므로 getattr로 접근.
        - 상세 내용과 배경 정보는 중복을 피하기 위해 조정.
        - 필수 필드가 누락된 경우 'N/A'로 대체.
    """
    report = f"## 📋 {issue.title}\n\n"
    report += f"**출처**: {issue.source or 'N/A'} | **발행일**: {issue.published_date or 'N/A'}\n"

    reppl_conf = getattr(issue, 'reppl_confidence', None)
    conf_text = f" | **RePPL 신뢰도**: {int(reppl_conf * 100)}%" if reppl_conf else ""
    report += f"**관련도**: {int(issue.relevance_score * 100)}%{conf_text}\n\n"

    report += f"### 📝 요약\n{issue.summary}\n\n"

    if issue.detailed_content:
        content_to_display = issue.detailed_content
        if issue.background_context:
            # 배경 정보 중복 제거
            content_to_display = content_to_display.replace(f"**배경 정보**:{issue.background_context}", "")
        report += f"### 📖 상세 내용\n{content_to_display.strip()}\n\n"

    if issue.background_context:
        report += f"### 🔗 배경 정보\n{issue.background_context}\n"

    return report


def create_detailed_report_from_search_result(search_result: SearchResult) -> str:
    """전체 검색 결과로부터 종합 보고서를 마크다운 형식으로 생성합니다.

    보고서 헤더와 상세 분석이 포함된 이슈를 포함하며, 상세 이슈가 없어도 기본 헤더는 생성됩니다.

    Args:
        search_result (SearchResult): 포맷팅할 검색 결과 객체.

    Returns:
        str: 마크다운 형식의 종합 보고서 문자열.

    Notes:
        - 상세 정보(`detailed_content`)가 있는 이슈만 포함.
        - 상세 이슈가 없을 경우 안내 문구를 추가.
        - 보고서 헤더는 항상 생성되어 키워드와 기간 정보를 제공.
    """
    # 보고서 헤더 생성
    full_report = f"# 🔍 종합 이슈 분석 보고서\n"
    full_report += f"- 키워드: {', '.join(search_result.query_keywords)}\n"
    full_report += f"- 기간: {search_result.time_period}\n\n---\n"

    # 상세 정보가 있는 이슈 필터링
    detailed_issues = [issue for issue in search_result.issues if issue.detailed_content]

    if detailed_issues:
        for issue in detailed_issues:
            full_report += format_detailed_issue_report(issue) + "\n---\n"
    else:
        full_report += "\n신뢰도 높은 상세 분석 이슈를 찾지 못했습니다."

    return full_report


def save_report_to_file(report_content: str, topic: str) -> str:
    """보고서 내용을 마크다운 파일로 저장하고 파일 경로를 반환합니다.

    파일은 'reports' 디렉토리에 저장되며, 주제와 타임스탬프를 기반으로 고유한 파일명을 생성합니다.

    Args:
        report_content (str): 저장할 보고서 내용.
        topic (str): 보고서의 주제 (파일명 생성에 사용).

    Returns:
        str: 생성된 파일의 절대 경로.

    Notes:
        - 'reports' 디렉토리가 없으면 자동 생성.
        - 파일명은 주제와 현재 시각(YYYYMMDD_HHMMSS)을 조합하여 고유성 보장.
        - UTF-8 인코딩을 사용하여 한글 및 특수문자 지원.
    """
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)  # 디렉토리 생성 (이미 존재 시 무시)
    filename = f"report_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    file_path = os.path.join(reports_dir, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    return file_path