# src/issue_searcher.py
"""
이슈 검색 및 분석 모듈
- PerplexityClient를 활용하여 키워드 기반 이슈를 검색합니다.
- 검색 결과를 파싱하고 관련성 점수를 계산하여 구조화된 데이터를 반환합니다.
- 비동기 방식으로 개별 이슈의 상세 정보를 수집하고 분석합니다.
"""

import asyncio
import json
import re
import time
from typing import List, Dict, Optional, Any
from loguru import logger

from src.clients.perplexity_client import PerplexityClient
from src.models import IssueItem, SearchResult, KeywordResult


class IssueSearcher:
    """이슈 검색 및 분석을 수행하는 핵심 서비스 클래스.

    PerplexityClient를 통해 이슈를 검색하고, 결과를 파싱하며, 관련성 점수를 계산합니다.
    상세 정보 수집을 비동기적으로 처리하여 성능을 최적화합니다.
    """

    def __init__(self, api_key: Optional[str] = None):
        """IssueSearcher 인스턴스를 초기화합니다.

        Args:
            api_key (Optional[str]): Perplexity API 키. None일 경우 환경 변수에서 로드됩니다.
        """
        self.client = PerplexityClient(api_key)
        self.max_keywords_per_search = 5  # 한 번의 검색에 사용할 최대 키워드 수
        self.max_results_per_search = 10  # 검색당 반환할 최대 결과 수
        self.max_detailed_issues = 5  # 상세 정보를 수집할 최대 이슈 수
        logger.info("IssueSearcher 초기화 완료")

    async def search_issues_from_keywords(
            self,
            keyword_result: KeywordResult,
            time_period: str,
            max_total_results: int = 20,
            collect_details: bool = True
    ) -> SearchResult:
        """키워드를 기반으로 이슈를 검색하고, 필요 시 상세 정보를 비동기적으로 수집합니다.

        Args:
            keyword_result (KeywordResult): 검색에 사용할 키워드와 관련 정보.
            time_period (str): 검색 기간 (예: '최근 1주일').
            max_total_results (int): 반환할 최대 이슈 수. 기본값은 20.
            collect_details (bool): 상세 정보 수집 여부. 기본값은 True.

        Returns:
            SearchResult: 검색 결과와 메타데이터를 포함한 객체.

        Raises:
            Exception: API 호출 또는 데이터 처리 중 오류 발생 시.
        """
        start_time = time.time()
        search_keywords = self._prepare_search_keywords(keyword_result)

        try:
            # 1. Perplexity API를 통해 기본 이슈 검색 수행
            api_response = await self.client.search_issues(search_keywords, time_period, max_total_results)
            logger.info(f"API 원본 응답:\n{json.dumps(api_response, ensure_ascii=False, indent=2)}")

            # 2. API 응답을 파싱하여 이슈 목록 생성
            issues = self._parse_api_response(api_response)

            # 3. 키워드 기반으로 관련성 점수 계산
            scored_issues = self._calculate_relevance_scores(issues, keyword_result)
            top_issues = sorted(scored_issues, key=lambda x: x.relevance_score, reverse=True)[:max_total_results]

            # 4. 상세 정보 수집 (선택적)
            if collect_details and top_issues:
                await self._collect_and_update_details(top_issues)

            # 5. 최종 결과 객체 생성 및 반환
            return self._aggregate_search_result(
                start_time, search_keywords, top_issues, api_response,
                time_period, keyword_result.confidence_score, collect_details
            )

        except Exception as e:
            # 오류 발생 시 기본 결과 객체 반환 및 로깅
            logger.error(f"이슈 검색 과정에서 심각한 오류: {e}", exc_info=True)
            return SearchResult(
                query_keywords=search_keywords,
                total_found=0,
                issues=[],
                search_time=time.time() - start_time,
                api_calls_used=1,
                confidence_score=0.1,
                time_period=time_period,
                raw_responses=[]
            )

    def _prepare_search_keywords(self, keyword_result: KeywordResult) -> List[str]:
        """검색에 사용할 키워드를 준비합니다.

        기본 키워드와 관련 용어를 조합하여 중복 제거 후 최대 키워드 수 제한을 적용합니다.

        Args:
            keyword_result (KeywordResult): 키워드와 관련 용어를 포함한 객체.

        Returns:
            List[str]: 검색에 사용할 키워드 리스트.
        """
        keywords = keyword_result.primary_keywords[:3] + keyword_result.related_terms[:2]
        return list(dict.fromkeys(keywords))[:self.max_keywords_per_search]

    def _parse_api_response(self, api_response: Dict[str, Any]) -> List[IssueItem]:
        """API 응답을 파싱하여 IssueItem 객체 리스트로 변환합니다.

        Args:
            api_response (Dict[str, Any]): Perplexity API 응답 데이터.

        Returns:
            List[IssueItem]: 파싱된 이슈 객체 리스트. 오류 시 빈 리스트 반환.
        """
        try:
            content = api_response['choices'][0]['message']['content']
            # 이슈 섹션이 '## **' 패턴으로 구분된다고 가정하고 분리
            issue_blocks = re.finditer(r'(?s)(##\s*\*\*.*?(?=\n##\s*\*\*|\Z))', content)
            issues = [self._parse_issue_section(match.group(1).strip()) for match in issue_blocks]
            return [issue for issue in issues if issue]  # None이 아닌 유효한 이슈만 반환
        except (KeyError, IndexError) as e:
            logger.error(f"API 응답 파싱 실패: {e}")
            return []

    def _parse_issue_section(self, section: str) -> Optional[IssueItem]:
        """단일 이슈 텍스트 블록을 파싱하여 IssueItem 객체를 생성합니다.

        Args:
            section (str): 파싱할 이슈 텍스트 블록.

        Returns:
            Optional[IssueItem]: 파싱된 이슈 객체. 실패 시 None 반환.
        """
        try:
            # 제목과 요약 추출
            title_match = re.search(r'##\s*\*\*(.*)\*\*', section)
            summary_match = re.search(r'\*\*요약\*\*:\s*(.*?)(?=\*\*|$)', section, re.DOTALL)

            title = title_match.group(1).strip() if title_match else None
            summary = summary_match.group(1).strip() if summary_match else None

            if not title or not summary:
                logger.warning(f"제목 또는 요약 누락: {section[:50]}...")
                return None

            # 기본 필드 추출
            source_match = re.search(r'\*\*출처\*\*:\s*(.*?)(?=\*\*|$)', section)
            date_match = re.search(r'\*\*발행일\*\*:\s*(.*?)(?=\*\*|$)', section)
            category_match = re.search(r'\*\*카테고리\*\*:\s*(.*?)(?=\*\*|$)', section)

            # 추가 필드 추출 (기술적 핵심, 중요도, 관련 키워드)
            tech_core_match = re.search(r'\*\*기술적 핵심\*\*:\s*(.*?)(?=\*\*|$)', section, re.DOTALL)
            importance_match = re.search(r'\*\*중요도\*\*:\s*(.*?)(?=\*\*|$)', section, re.DOTALL)
            related_kw_match = re.search(r'\*\*관련 키워드\*\*:\s*(.*?)(?=\*\*|$)', section, re.DOTALL)

            # IssueItem 객체 생성
            issue = IssueItem(
                title=title,
                summary=summary,
                source=source_match.group(1).strip() if source_match else 'Unknown',
                published_date=date_match.group(1).strip() if date_match else None,
                relevance_score=0.5,  # 초기 점수, 이후 계산에서 갱신
                category=category_match.group(1).strip() if category_match else 'news',
                content_snippet=summary[:200]
            )

            # 동적으로 추가 필드 설정
            if tech_core_match:
                setattr(issue, 'technical_core', tech_core_match.group(1).strip())
            if importance_match:
                setattr(issue, 'importance', importance_match.group(1).strip())
            if related_kw_match:
                setattr(issue, 'related_keywords', related_kw_match.group(1).strip())

            return issue
        except Exception as e:
            logger.error(f"이슈 섹션 파싱 오류: {e}")
            return None

    def _calculate_relevance_scores(self, issues: List[IssueItem], keyword_result: KeywordResult) -> List[IssueItem]:
        """키워드 매칭을 통해 각 이슈의 관련성 점수를 계산합니다.

        Args:
            issues (List[IssueItem]): 점수를 계산할 이슈 리스트.
            keyword_result (KeywordResult): 비교에 사용할 키워드 데이터.

        Returns:
            List[IssueItem]: 관련성 점수가 계산된 이슈 리스트.
        """
        for issue in issues:
            # 점수 계산에 사용할 텍스트 준비
            text_to_check = f"{issue.title} {issue.summary} {issue.content_snippet}".lower()

            # 추가 필드 포함 (존재 시)
            if hasattr(issue, 'technical_core') and issue.technical_core:
                text_to_check += f" {issue.technical_core}".lower()
            if hasattr(issue, 'related_keywords') and issue.related_keywords:
                text_to_check += f" {issue.related_keywords}".lower()

            # 키워드 매칭 점수 계산
            score = sum(0.4 for kw in keyword_result.primary_keywords if kw.lower() in text_to_check)
            score += sum(0.1 for kw in keyword_result.related_terms if kw.lower() in text_to_check)

            # 키워드 매칭 시 보너스 점수
            if score > 0:
                score += 0.1

            # 중요도에 따른 가중치 적용
            if hasattr(issue, 'importance') and issue.importance:
                if 'Critical' in issue.importance:
                    score *= 1.3
                elif 'High' in issue.importance:
                    score *= 1.2

            issue.relevance_score = min(1.0, round(score, 2))

        return issues

    async def _collect_and_update_details(self, issues: List[IssueItem]):
        """상세 정보를 병렬로 수집하고 원본 이슈 리스트를 업데이트합니다.

        Args:
            issues (List[IssueItem]): 상세 정보를 수집할 이슈 리스트.
        """
        issues_to_detail = issues[:self.max_detailed_issues]
        tasks = [self._collect_issue_details(issue) for issue in issues_to_detail]
        detailed_results = await asyncio.gather(*tasks)

        # 원본 리스트에 상세 정보 반영
        for i, updated_issue in enumerate(detailed_results):
            original_index = issues.index(issues_to_detail[i])
            issues[original_index] = updated_issue

    async def _collect_issue_details(self, issue: IssueItem) -> IssueItem:
        """단일 이슈에 대한 상세 정보를 비동기적으로 수집합니다.

        Args:
            issue (IssueItem): 상세 정보를 수집할 이슈 객체.

        Returns:
            IssueItem: 상세 정보가 추가된 이슈 객체.
        """
        start_time = time.time()
        logger.info(f"'{issue.title[:30]}...' 세부 정보 수집 중")
        try:
            # Perplexity API를 통해 상세 정보 요청
            response = await self.client.collect_detailed_information(issue.title)
            content = response['choices'][0]['message']['content']

            # 상세 정보 저장
            issue.detailed_content = content
            issue.detail_confidence = self._calculate_detail_confidence(content)

            # 구조화된 상세 정보 파싱
            if "### 1. 핵심 기술 분석" in content:
                tech_section = content.split("### 1. 핵심 기술 분석")[1].split("### 2.")[0]
                setattr(issue, 'technical_analysis', tech_section.strip())

            if "### 2. 배경 및 맥락" in content:
                bg_section = content.split("### 2. 배경 및 맥락")[1].split("### 3.")[0]
                issue.background_context = bg_section.strip()

            if "### 3. 심층 영향 분석" in content:
                impact_section = content.split("### 3. 심층 영향 분석")[1].split("### 4.")[0]
                setattr(issue, 'impact_analysis', impact_section.strip())

            if "### 4. 실무 적용 가이드" in content:
                practical_section = content.split("### 4. 실무 적용 가이드")[1].split("### 5.")[0]
                setattr(issue, 'practical_guide', practical_section.strip())

            issue.detail_collection_time = time.time() - start_time
            logger.success(f"'{issue.title[:30]}...' 세부 정보 수집 완료 ({issue.detail_collection_time:.2f}초)")

        except Exception as e:
            logger.error(f"'{issue.title[:30]}...' 세부 정보 수집 실패: {e}")
            issue.detail_confidence = 0.0
            issue.detail_collection_time = time.time() - start_time

        return issue

    def _calculate_detail_confidence(self, detailed_content: str) -> float:
        """상세 정보의 신뢰도를 내용 길이에 기반하여 계산합니다.

        Args:
            detailed_content (str): 신뢰도를 계산할 상세 정보 텍스트.

        Returns:
            float: 계산된 신뢰도 점수 (0.0 ~ 0.9).
        """
        if not detailed_content:
            return 0.0
        score = min(1.0, len(detailed_content) / 1000.0)  # 1000자 기준 최대 1.0
        return round(score * 0.8 + 0.1, 2)  # 최소 0.1, 최대 0.9

    def _aggregate_search_result(
            self,
            start_time: float,
            keywords: List[str],
            issues: List[IssueItem],
            raw_response: Dict[str, Any],
            time_period: str,
            kw_confidence: float,
            details_collected: bool
    ) -> SearchResult:
        """검색 결과를 종합하여 최종 SearchResult 객체를 생성합니다.

        Args:
            start_time (float): 검색 시작 시간 (Unix timestamp).
            keywords (List[str]): 사용된 검색 키워드.
            issues (List[IssueItem]): 처리된 이슈 리스트.
            raw_response (Dict[str, Any]): 원본 API 응답.
            time_period (str): 검색 기간.
            kw_confidence (float): 키워드 신뢰도 점수.
            details_collected (bool): 상세 정보 수집 여부.

        Returns:
            SearchResult: 최종 결과 객체.
        """
        # 상세 정보 수집 성공 여부 확인
        successful_details = [
            iss for iss in issues
            if hasattr(iss, 'detail_confidence') and iss.detail_confidence is not None and iss.detail_confidence > 0
        ]
        detailed_issues_count = len(successful_details)
        total_detail_time = sum(getattr(iss, 'detail_collection_time', 0) for iss in successful_details)
        avg_detail_confidence = (
            sum(iss.detail_confidence for iss in successful_details) / detailed_issues_count
            if detailed_issues_count > 0 else 0.0
        )

        # 평균 관련성 점수 계산
        avg_relevance_score = sum(iss.relevance_score for iss in issues) / len(issues) if issues else 0.0

        # 최종 신뢰도 점수 계산
        final_confidence_score = (
                (avg_relevance_score * 0.5) +
                (avg_detail_confidence * 0.3) +
                (kw_confidence * 0.2)
        )

        return SearchResult(
            query_keywords=keywords,
            total_found=len(issues),
            issues=issues,
            search_time=time.time() - start_time,
            api_calls_used=1 + (len(issues[:self.max_detailed_issues]) if details_collected else 0),
            confidence_score=final_confidence_score,
            time_period=time_period,
            raw_responses=[json.dumps(raw_response, ensure_ascii=False)],
            detailed_issues_count=detailed_issues_count,
            total_detail_collection_time=total_detail_time,
            average_detail_confidence=avg_detail_confidence
        )


# --- 편의 함수 ---
def create_issue_searcher(api_key: Optional[str] = None) -> IssueSearcher:
    """IssueSearcher 인스턴스를 생성하는 팩토리 함수.

    Args:
        api_key (Optional[str]): Perplexity API 키. None일 경우 환경 변수에서 로드.

    Returns:
        IssueSearcher: 초기화된 IssueSearcher 인스턴스.
    """
    return IssueSearcher(api_key=api_key)


async def search_issues_for_keywords(
        keyword_result: KeywordResult,
        time_period: str = "최근 1주일",
        collect_details: bool = True
) -> SearchResult:
    """키워드 결과를 받아 이슈 검색을 수행하는 고수준 래퍼 함수.

    Args:
        keyword_result (KeywordResult): 검색에 사용할 키워드 데이터.
        time_period (str): 검색 기간. 기본값은 '최근 1주일'.
        collect_details (bool): 상세 정보 수집 여부. 기본값은 True.

    Returns:
        SearchResult: 검색 결과 객체.
    """
    searcher = create_issue_searcher()
    return await searcher.search_issues_from_keywords(keyword_result, time_period, collect_details=collect_details)