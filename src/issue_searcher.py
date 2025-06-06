"""
이슈 검색 모듈 - Perplexity API 연동
생성된 키워드를 기반으로 실시간 이슈를 검색하고 세부 정보를 수집
"""

import asyncio
import json
import re
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import httpx
from loguru import logger

from src.config import config
from src.keyword_generator import KeywordResult


@dataclass
class EntityInfo:
    """관련 인물/기관 정보"""
    name: str
    role: str
    relevance: float
    entity_type: str
    description: str


@dataclass
class ImpactAnalysis:
    """영향도 분석 정보"""
    impact_level: str
    impact_score: float
    affected_sectors: List[str]
    geographic_scope: str
    time_sensitivity: str
    reasoning: str


@dataclass
class TimelineEvent:
    """시간순 이벤트 정보"""
    date: str
    event_type: str
    description: str
    importance: float
    source: str


@dataclass
class IssueItem:
    """개별 이슈 정보를 담는 데이터 클래스"""
    title: str
    summary: str
    source: str
    published_date: Optional[str]
    relevance_score: float
    category: str
    content_snippet: str
    detailed_content: Optional[str] = None
    related_entities: List[EntityInfo] = field(default_factory=list)
    impact_analysis: Optional[ImpactAnalysis] = None
    timeline_events: List[TimelineEvent] = field(default_factory=list)
    background_context: Optional[str] = None
    detail_collection_time: Optional[float] = None
    detail_confidence: Optional[float] = None


@dataclass
class SearchResult:
    """이슈 검색 결과를 담는 데이터 클래스"""
    query_keywords: List[str]
    total_found: int
    issues: List[IssueItem]
    search_time: float
    api_calls_used: int
    confidence_score: float
    time_period: str
    raw_responses: List[str]
    detailed_issues_count: int = 0
    total_detail_collection_time: float = 0.0
    average_detail_confidence: float = 0.0


class PerplexityClient:
    """Perplexity API 클라이언트"""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.get_perplexity_api_key()
        if not self.api_key: raise ValueError("Perplexity API 키가 설정되지 않았습니다")
        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-large-128k-online"
        self.timeout = 60
        self.max_retries = 3
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        logger.info(f"PerplexityClient 초기화 완료 (모델: {self.model})")

    async def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        payload = {"model": self.model, "messages": [{"role": "system", "content": "You are a precise and objective information analysis expert."}, {"role": "user", "content": prompt}], "max_tokens": 4096, "temperature": 0.3}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(self.base_url, headers=self.headers, json=payload)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    logger.error(f"API HTTP Error (Status: {e.response.status_code}): {e.response.text}")
                    if e.response.status_code == 429: await asyncio.sleep(2 ** attempt); continue
                    raise
                except httpx.RequestError as e:
                    logger.error(f"API Request Error (Attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1: raise
        raise ValueError("API call failed after multiple retries.")

    async def search_issues(self, keywords: List[str], time_period: str, max_results: int) -> Dict[str, Any]:
        prompt = f"""Keywords: {", ".join(keywords)}. Time Period: {time_period}. Find up to {max_results} relevant issues from recent news, blogs, and academic sources, and summarize them in the following format:
## **[Title]**
**요약**: [Summary]
**출처**: [Source]
**일자**: [Date]
**카테고리**: [Category]"""
        return await self._make_api_call(prompt)

    async def collect_detailed_information(self, issue_title: str) -> Dict[str, Any]:
        prompt = f"""Provide a detailed analysis for the following issue: **{issue_title}**. Include detailed content and background context."""
        logger.info(f"Requesting detailed info for: {issue_title[:50]}...")
        return await self._make_api_call(prompt)

class IssueSearcher:
    """이슈 검색 및 분석 클래스"""
    def __init__(self, api_key: Optional[str] = None):
        self.client = PerplexityClient(api_key)
        self.max_keywords_per_search = 5
        self.max_results_per_search = 10
        self.max_detailed_issues = 5
        logger.info("IssueSearcher 초기화 완료")

    def _prepare_search_keywords(self, keyword_result: KeywordResult) -> List[str]:
        keywords = keyword_result.primary_keywords[:3] + keyword_result.related_terms[:2]
        return list(dict.fromkeys(keywords))[:self.max_keywords_per_search]

    def _parse_issue_section(self, section: str) -> Optional[IssueItem]:
        try:
            title_match = re.search(r'##\s*\*\*(.*)\*\*', section)
            summary_match = re.search(r'\*\*요약\*\*:\s*(.*)', section, re.DOTALL)
            title = title_match.group(1).strip() if title_match else None
            summary = summary_match.group(1).strip().split('\n**')[0].strip() if summary_match else None
            if not title or not summary: return None

            source_match = re.search(r'\*\*출처\*\*:\s*(.*)', section)
            date_match = re.search(r'\*\*일자\*\*:\s*(.*)', section)
            category_match = re.search(r'\*\*카테고리\*\*:\s*(.*)', section)

            return IssueItem(
                title=title, summary=summary,
                source=source_match.group(1).strip() if source_match else 'Unknown',
                published_date=date_match.group(1).strip() if date_match else None,
                relevance_score=0.5, category=category_match.group(1).strip() if category_match else 'news',
                content_snippet=summary[:200]
            )
        except Exception: return None

    def _parse_api_response(self, api_response: Dict[str, Any]) -> List[IssueItem]:
        try:
            content = api_response['choices'][0]['message']['content']
            issue_blocks = re.finditer(r'(?s)(##\s*\*\*.*?(?=\n##\s*\*\*|\Z))', content)
            issues = [self._parse_issue_section(match.group(1).strip()) for match in issue_blocks]
            return [issue for issue in issues if issue]
        except (KeyError, IndexError): return []

    def _calculate_relevance_scores(self, issues: List[IssueItem], keyword_result: KeywordResult) -> List[IssueItem]:
        for issue in issues:
            text_to_check = f"{issue.title} {issue.summary}".lower()
            score = sum(0.2 for kw in keyword_result.primary_keywords if kw.lower() in text_to_check)
            score += sum(0.1 for kw in keyword_result.related_terms if kw.lower() in text_to_check)
            issue.relevance_score = min(1.0, round(score, 2))
        return issues

    def _calculate_detail_confidence(self, detailed_content: str) -> float:
        if not detailed_content: return 0.0
        score = min(1.0, len(detailed_content) / 1000.0)
        return round(score * 0.8 + 0.1, 2)

    async def _collect_issue_details(self, issue: IssueItem) -> IssueItem:
        start_time = time.time()
        logger.info(f"'{issue.title[:30]}...' 세부 정보 수집 중")
        try:
            response = await self.client.collect_detailed_information(issue.title)
            content = response['choices'][0]['message']['content']

            issue.detailed_content = content
            issue.detail_confidence = self._calculate_detail_confidence(content)

            if "**배경 정보**:" in content:
                issue.background_context = content.split('**배경 정보**:')[1].strip().split('\n**')[0]

            issue.detail_collection_time = time.time() - start_time
            logger.success(f"'{issue.title[:30]}...' 세부 정보 수집 완료 ({issue.detail_collection_time:.2f}초)")

        except Exception as e:
            logger.error(f"'{issue.title[:30]}...' 세부 정보 수집 실패: {e}")
            issue.detail_confidence = 0.0
            issue.detail_collection_time = time.time() - start_time
        return issue

    async def search_issues_from_keywords(self, keyword_result: KeywordResult, time_period: str, max_total_results: int = 20, collect_details: bool = True) -> SearchResult:
        start_time = time.time()
        search_keywords = self._prepare_search_keywords(keyword_result)

        try:
            api_response = await self.client.search_issues(search_keywords, time_period, max_total_results)
            logger.info(f"API 원본 응답:\n{json.dumps(api_response, ensure_ascii=False, indent=2)}")

            issues = self._parse_api_response(api_response)
            scored_issues = self._calculate_relevance_scores(issues, keyword_result)
            top_issues = sorted(scored_issues, key=lambda x: x.relevance_score, reverse=True)[:max_total_results]

            if collect_details and top_issues:
                issues_to_detail = top_issues[:self.max_detailed_issues]
                tasks = [self._collect_issue_details(issue) for issue in issues_to_detail]
                detailed_results = await asyncio.gather(*tasks)

                for i, updated_issue in enumerate(detailed_results):
                    original_index = top_issues.index(issues_to_detail[i])
                    top_issues[original_index] = updated_issue

            successful_details = [iss for iss in top_issues if iss.detail_confidence is not None and iss.detail_confidence > 0]
            detailed_issues_count = len(successful_details)
            total_detail_time = sum(iss.detail_collection_time for iss in successful_details if iss.detail_collection_time)
            avg_detail_confidence = sum(iss.detail_confidence for iss in successful_details) / detailed_issues_count if detailed_issues_count > 0 else 0.0

            return SearchResult(
                query_keywords=search_keywords, total_found=len(top_issues), issues=top_issues,
                search_time=time.time() - start_time, api_calls_used=1 + (len(top_issues[:self.max_detailed_issues]) if collect_details else 0),
                confidence_score=0.8, time_period=time_period, raw_responses=[json.dumps(api_response, ensure_ascii=False)],
                detailed_issues_count=detailed_issues_count, total_detail_collection_time=total_detail_time,
                average_detail_confidence=avg_detail_confidence
            )

        except Exception as e:
            logger.error(f"이슈 검색 과정에서 심각한 오류: {e}", exc_info=True)
            return SearchResult(query_keywords=search_keywords, total_found=0, issues=[], search_time=time.time() - start_time, api_calls_used=1, confidence_score=0.1, time_period=time_period, raw_responses=[])

    def format_search_summary(self, result: SearchResult) -> str:
        if result.total_found == 0: return f"**이슈 검색 실패**\n❌ '{', '.join(result.query_keywords)}' 관련 이슈를 찾지 못했습니다."
        summary = f"**이슈 검색 완료** ({result.total_found}개 발견)\n"
        for i, issue in enumerate(result.issues[:3], 1):
            summary += f"**{i}. {issue.title}**\n- 출처: {issue.source} | 관련도: {int(issue.relevance_score * 100)}%\n"
        return summary

    def format_detailed_issue_report(self, issue: IssueItem) -> str:
        """[수정됨] 상세 보고서 생성 시 배경 정보(background_context) 필드 추가"""
        report = f"# 📋 {issue.title}\n\n"
        report += f"**출처**: {issue.source or 'N/A'} | **발행일**: {issue.published_date or 'N/A'}\n"
        if issue.relevance_score is not None and issue.detail_confidence is not None:
            report += f"**관련도**: {int(issue.relevance_score * 100)}% | **세부신뢰도**: {int(issue.detail_confidence * 100)}%\n\n"
        report += f"## 📝 요약\n{issue.summary}\n\n"
        if issue.detailed_content:
            report += f"## 📖 상세 내용\n{issue.detailed_content}\n\n"
        if issue.background_context:
            report += f"## 🔗 배경 정보\n{issue.background_context}\n"
        return report

def create_issue_searcher(api_key: Optional[str] = None) -> IssueSearcher:
    return IssueSearcher(api_key=api_key)

async def search_issues_for_keywords(keyword_result: KeywordResult, time_period: str = "최근 1주일", collect_details: bool = True) -> SearchResult:
    searcher = create_issue_searcher()
    return await searcher.search_issues_from_keywords(keyword_result, time_period, collect_details=collect_details)

def create_detailed_report_from_search_result(search_result: SearchResult) -> str:
    if not search_result.issues: return "상세 분석할 이슈가 없습니다."
    searcher = create_issue_searcher()
    full_report = f"# 🔍 종합 이슈 분석 보고서\n- 키워드: {', '.join(search_result.query_keywords)}\n- 기간: {search_result.time_period}\n\n---\n"
    for issue in search_result.issues:
        if issue.detailed_content: full_report += searcher.format_detailed_issue_report(issue) + "\n---\n"
    return full_report