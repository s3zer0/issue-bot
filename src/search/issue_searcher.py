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
from datetime import datetime, timedelta
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
            max_total_results: int = 50,
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
            
            # 콘텐츠 정제: 디버깅 메시지 및 상태 메시지 제거
            content = self._sanitize_content(content)
            
            # 이슈 섹션이 '## **' 패턴으로 구분된다고 가정하고 분리
            issue_blocks = re.finditer(r'(?s)(##\s*\*\*.*?(?=\n##\s*\*\*|\Z))', content)
            issues = [self._parse_issue_section(match.group(1).strip()) for match in issue_blocks]
            return [issue for issue in issues if issue]  # None이 아닌 유효한 이슈만 반환
        except (KeyError, IndexError) as e:
            logger.error(f"API 응답 파싱 실패: {e}")
            return []

    def _sanitize_content(self, content: str) -> str:
        """
        콘텐츠에서 디버깅 메시지나 상태 메시지를 제거합니다.
        
        Args:
            content: 정제할 콘텐츠
            
        Returns:
            str: 정제된 콘텐츠
        """
        # 키워드 재생성 관련 메시지 패턴들
        debug_patterns = [
            r'Keywords? need to be regenerated for topic.*',
            r'키워드.*재생성.*',
            r'키워드 재생성.*',
            r'시뮬레이션 모드.*',
            r'API 실패.*폴백.*',
            r'최종 폴백.*',
            r'연속 실패.*회.*',
            r'Error occurred during keyword generation.*',
            r'키워드 생성 완료.*재생성 횟수.*',
            r'하이브리드 검색.*',
            r'폴백.*시뮬레이션.*',
        ]
        
        # 각 패턴에 대해 제거 수행
        for pattern in debug_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        # 빈 줄 여러 개를 하나로 정리
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # 디버깅 메시지 제거 로그
        if any(re.search(pattern, content, re.IGNORECASE) for pattern in debug_patterns):
            logger.info("콘텐츠에서 디버깅 메시지 제거됨")
        
        return content.strip()

    def _parse_issue_section(self, section: str) -> Optional[IssueItem]:
        """단일 이슈 텍스트 블록을 파싱하여 IssueItem 객체를 생성합니다.

        Args:
            section (str): 파싱할 이슈 텍스트 블록.

        Returns:
            Optional[IssueItem]: 파싱된 이슈 객체. 실패 시 None 반환.
        """
        try:
            # 제목과 요약 추출
            title_match = re.search(r'##\s*\*\*(.*?)\*\*', section)
            summary_match = re.search(r'\*\*요약\*\*:\s*(.*?)(?=\*\*|$)', section, re.DOTALL)

            title = title_match.group(1).strip() if title_match else None
            summary = summary_match.group(1).strip() if summary_match else None

            if not title or not summary:
                logger.warning(f"제목 또는 요약 누락: {section[:50]}...")
                return None

            # 기본 필드 추출 (더 유연한 패턴 사용)
            source = self._extract_field(section, ['출처', 'Source'])
            date_str = self._extract_field(section, ['발행일', 'Date', 'Published'])
            category = self._extract_field(section, ['카테고리', 'Category']) or 'news'

            # 추가 필드 추출
            tech_core = self._extract_field(section, ['기술적 핵심', 'Technical Core'])
            importance = self._extract_field(section, ['중요도', 'Importance'])
            related_kw = self._extract_field(section, ['관련 키워드', 'Related Keywords'])

            # 출처 정리 (URL이 없으면 웹사이트명만이라도 유지)
            source = self._clean_source(source)

            # 날짜 파싱 및 정리
            published_date = self._parse_date(date_str)

            # IssueItem 객체 생성
            issue = IssueItem(
                title=title,
                summary=summary,
                source=source,
                published_date=published_date,
                relevance_score=0.5,  # 초기 점수, 이후 계산에서 갱신
                category=category,
                content_snippet=summary[:200]
            )

            # 동적으로 추가 필드 설정
            if tech_core:
                setattr(issue, 'technical_core', tech_core)
            if importance:
                setattr(issue, 'importance', importance)
            if related_kw:
                setattr(issue, 'related_keywords', related_kw)

            return issue
        except Exception as e:
            logger.error(f"이슈 섹션 파싱 오류: {e}")
            return None

    def _extract_field(self, text: str, field_names: List[str]) -> Optional[str]:
        """다양한 필드명으로 값을 추출합니다."""
        for field_name in field_names:
            pattern = rf'\*\*{field_name}\*\*:\s*(.*?)(?=\*\*|$)'
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _clean_source(self, source: Optional[str]) -> str:
        """출처 정보를 정리합니다."""
        if not source or source.lower() in ['unknown', 'n/a', '알 수 없음']:
            return 'Unknown'

        # URL이 포함된 경우 도메인명 추출
        url_match = re.search(r'https?://([^/]+)', source)
        if url_match:
            domain = url_match.group(1)
            # www. 제거
            domain = re.sub(r'^www\.', '', domain)
            return domain

        return source

    def _parse_date(self, date_str: Optional[str]) -> Optional[str]:
        """날짜 문자열을 파싱하여 표준 형식으로 변환합니다."""
        if not date_str or date_str.lower() in ['n/a', '알 수 없음', 'unknown']:
            return None

        # 다양한 날짜 형식 시도
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%Y.%m.%d',
            '%Y년 %m월 %d일',
            '%B %d, %Y',  # January 15, 2024
            '%d %B %Y',  # 15 January 2024
        ]

        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue

        # 날짜 파싱 실패 시 원본 반환
        return date_str

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

            # 키워드 매칭 점수 계산 (개선된 가중치)
            primary_matches = sum(1 for kw in keyword_result.primary_keywords if kw.lower() in text_to_check)
            related_matches = sum(1 for kw in keyword_result.related_terms if kw.lower() in text_to_check)
            context_matches = sum(1 for kw in keyword_result.context_keywords if kw.lower() in text_to_check)

            # 기본 점수 계산
            base_score = 0.0
            if primary_matches > 0:
                base_score += min(0.5, primary_matches * 0.2)  # 최대 0.5
            if related_matches > 0:
                base_score += min(0.3, related_matches * 0.15)  # 최대 0.3
            if context_matches > 0:
                base_score += min(0.2, context_matches * 0.1)  # 최대 0.2

            # 중요도에 따른 가중치 적용
            if hasattr(issue, 'importance') and issue.importance:
                importance_lower = issue.importance.lower()
                if 'critical' in importance_lower:
                    base_score *= 1.3
                elif 'high' in importance_lower:
                    base_score *= 1.2
                elif 'low' in importance_lower:
                    base_score *= 0.8

            # 출처 신뢰도 가중치
            if issue.source and issue.source != 'Unknown':
                # 구체적인 출처가 있으면 가산점
                base_score += 0.05

                # 신뢰할 만한 출처 패턴
                trusted_sources = ['techcrunch', 'reuters', 'bloomberg', 'ieee', 'nature',
                                   'arxiv', 'github', 'microsoft', 'google', 'apple', 'aws']
                if any(src in issue.source.lower() for src in trusted_sources):
                    base_score += 0.1

            # 날짜 유효성 가산점
            if issue.published_date and issue.published_date != 'N/A':
                base_score += 0.05

            # 최종 점수 (0.0 ~ 1.0 범위로 제한)
            issue.relevance_score = min(1.0, max(0.1, round(base_score, 2)))

        return issues

    async def _collect_and_update_details(self, issues: List[IssueItem]):
        """상세 정보를 병렬로 수집하고 원본 이슈 리스트를 업데이트합니다.
        
        Performance Optimized: O(n) → O(1) index lookups

        Args:
            issues (List[IssueItem]): 상세 정보를 수집할 이슈 리스트.
        """
        issues_to_detail = issues[:self.max_detailed_issues]
        
        # Performance: Create O(1) index mapping instead of O(n) list.index() calls
        issue_index_map = {id(issue): i for i, issue in enumerate(issues)}
        
        # Parallel detail collection (already optimized)
        tasks = [self._collect_issue_details(issue) for issue in issues_to_detail]
        detailed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Performance: O(1) updates using pre-computed indices
        for original_issue, updated_result in zip(issues_to_detail, detailed_results):
            if not isinstance(updated_result, Exception):
                idx = issue_index_map[id(original_issue)]
                issues[idx] = updated_result
            else:
                logger.warning(f"Failed to collect details for issue: {original_issue.title[:30]}..., Error: {updated_result}")

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

            # 상세 정보에서 추가 메타데이터 추출
            self._extract_metadata_from_details(issue, content)

            issue.detail_collection_time = time.time() - start_time
            logger.success(f"'{issue.title[:30]}...' 세부 정보 수집 완료 ({issue.detail_collection_time:.2f}초)")

        except Exception as e:
            logger.error(f"'{issue.title[:30]}...' 세부 정보 수집 실패: {e}")
            issue.detail_confidence = 0.0
            issue.detail_collection_time = time.time() - start_time

        return issue

    def _extract_metadata_from_details(self, issue: IssueItem, content: str):
        """상세 정보에서 추가 메타데이터를 추출합니다."""
        # URL 패턴 찾기
        urls = re.findall(r'https?://[^\s\]]+', content)
        if urls and issue.source == 'Unknown':
            # 첫 번째 URL에서 도메인 추출
            domain = self._clean_source(urls[0])
            issue.source = domain

        # 날짜 정보 찾기 (상세 내용에서)
        if not issue.published_date or issue.published_date == 'N/A':
            date_patterns = [
                r'(\d{4}년\s*\d{1,2}월\s*\d{1,2}일)',
                r'(\d{4}-\d{1,2}-\d{1,2})',
                r'(\d{4}/\d{1,2}/\d{1,2})'
            ]
            for pattern in date_patterns:
                date_match = re.search(pattern, content)
                if date_match:
                    issue.published_date = self._parse_date(date_match.group(1))
                    break

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