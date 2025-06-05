"""
이슈 검색 모듈 - Perplexity API 연동
생성된 키워드를 기반으로 실시간 이슈를 검색하는 모듈
"""

import asyncio
import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import httpx
from loguru import logger

from src.config import config
from src.keyword_generator import KeywordResult


@dataclass
class IssueItem:
    """개별 이슈 정보를 담는 데이터 클래스"""
    title: str                      # 이슈 제목
    summary: str                    # 이슈 요약
    source: str                     # 출처 (URL 또는 매체명)
    published_date: Optional[str]   # 발행일
    relevance_score: float          # 관련성 점수 (0.0-1.0)
    category: str                   # 카테고리 (news, blog, social, academic)
    content_snippet: str            # 내용 일부


@dataclass
class SearchResult:
    """이슈 검색 결과를 담는 데이터 클래스"""
    query_keywords: List[str]       # 검색에 사용된 키워드
    total_found: int                # 총 발견된 이슈 수
    issues: List[IssueItem]         # 이슈 목록
    search_time: float              # 검색 소요 시간 (초)
    api_calls_used: int             # 사용된 API 호출 수
    confidence_score: float         # 검색 결과 신뢰도
    time_period: str                # 검색 기간
    raw_responses: List[str]        # 원본 API 응답들


class PerplexityClient:
    """Perplexity API 클라이언트"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.get_perplexity_api_key()
        if not self.api_key:
            raise ValueError("Perplexity API 키가 설정되지 않았습니다")

        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-small-128k-online"
        self.timeout = 60
        self.max_retries = 3

        # HTTP 클라이언트 설정
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"PerplexityClient 초기화 완료 (모델: {self.model})")

    async def search_issues(
        self,
        keywords: List[str],
        time_period: str = "최근 1주일",
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        키워드를 사용하여 이슈를 검색합니다

        Args:
            keywords: 검색할 키워드 리스트
            time_period: 검색 기간
            max_results: 최대 결과 수

        Returns:
            Dict: Perplexity API 응답
        """
        # 검색 쿼리 구성
        keyword_str = ", ".join(keywords[:5])  # 최대 5개 키워드만 사용

        prompt = f"""다음 키워드들과 관련된 {time_period} 동안의 최신 이슈와 뉴스를 찾아주세요: {keyword_str}

검색 요구사항:
1. 뉴스, 블로그, 소셜미디어, 학술논문에서 관련 이슈 검색
2. 각 이슈마다 제목, 간략한 요약, 출처를 포함
3. 최대 {max_results}개의 가장 관련성 높은 이슈 선별
4. 발행일자가 최근인 순서로 정렬

응답 형식:
각 이슈를 다음 형식으로 정리해주세요:
**제목**: [이슈 제목]
**요약**: [2-3문장 요약]
**출처**: [매체명 또는 URL]
**일자**: [발행일자]
**카테고리**: [news/blog/social/academic]

관련성이 높고 신뢰할 만한 최신 정보만 포함해주세요."""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 실시간 이슈 분석 전문가입니다. 주어진 키워드와 관련된 최신 뉴스와 이슈를 정확하고 신뢰할 수 있는 소스에서 찾아 체계적으로 정리해주세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "search_recency_filter": "week",  # 최근 1주일 결과만
            "return_related_questions": False,  # 관련 질문 불필요
            "return_images": False  # 이미지 불필요 (텍스트만)
        }

        logger.info(f"Perplexity API 호출 시작: 키워드={keyword_str}, 기간={time_period}")

        # API 호출 with 재시도
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload
                    )

                    if response.status_code == 200:
                        result = response.json()
                        logger.success(f"Perplexity API 호출 성공 (시도 {attempt + 1})")
                        return result
                    elif response.status_code == 401:
                        raise ValueError("Perplexity API 키가 유효하지 않습니다")
                    elif response.status_code == 429:
                        wait_time = 2 ** attempt
                        logger.warning(f"API 요청 한도 초과, {wait_time}초 대기 후 재시도...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = response.text
                        logger.error(f"Perplexity API 오류 (상태: {response.status_code}): {error_text}")
                        raise ValueError(f"API 호출 실패: {response.status_code}")

            except httpx.TimeoutException:
                logger.warning(f"API 호출 타임아웃 (시도 {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise ValueError("API 호출 타임아웃")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"API 호출 중 오류 (시도 {attempt + 1}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

        raise ValueError("모든 재시도 실패")


class IssueSearcher:
    """
    이슈 검색기 - Perplexity API를 사용한 이슈 탐색

    주요 기능:
    - 키워드 기반 이슈 검색
    - 검색 결과 파싱 및 구조화
    - 관련성 점수 계산
    - 검색 결과 신뢰도 평가
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = PerplexityClient(api_key)
        self.max_keywords_per_search = 5
        self.max_results_per_search = 10

        logger.info("IssueSearcher 초기화 완료")

    async def search_issues_from_keywords(
        self,
        keyword_result: KeywordResult,
        time_period: str = "최근 1주일",
        max_total_results: int = 20
    ) -> SearchResult:
        """
        키워드 생성 결과를 기반으로 이슈를 검색합니다

        Args:
            keyword_result: 키워드 생성 결과
            time_period: 검색 기간
            max_total_results: 최대 총 결과 수

        Returns:
            SearchResult: 검색 결과
        """
        start_time = time.time()
        logger.info(f"이슈 검색 시작: 주제='{keyword_result.topic}', 기간='{time_period}'")

        try:
            # 키워드 우선순위별 검색 전략
            search_keywords = self._prepare_search_keywords(keyword_result)

            # Perplexity API 호출
            api_response = await self.client.search_issues(
                keywords=search_keywords,
                time_period=time_period,
                max_results=max_total_results
            )

            # 응답 파싱
            issues = self._parse_api_response(api_response, search_keywords)

            # 관련성 점수 계산
            scored_issues = self._calculate_relevance_scores(issues, keyword_result)

            # 상위 결과만 선별
            top_issues = sorted(scored_issues, key=lambda x: x.relevance_score, reverse=True)[:max_total_results]

            # 신뢰도 점수 계산
            confidence_score = self._calculate_confidence_score(top_issues, keyword_result)

            search_time = time.time() - start_time

            # 원본 응답 저장 (안전한 방식)
            try:
                raw_response_str = json.dumps(api_response, ensure_ascii=False, indent=2)
            except (TypeError, ValueError) as e:
                logger.warning(f"API 응답 JSON 직렬화 실패: {e}")
                raw_response_str = str(api_response)

            result = SearchResult(
                query_keywords=search_keywords,
                total_found=len(top_issues),
                issues=top_issues,
                search_time=search_time,
                api_calls_used=1,
                confidence_score=confidence_score,
                time_period=time_period,
                raw_responses=[raw_response_str]
            )

            logger.success(
                f"이슈 검색 완료: {len(top_issues)}개 이슈 발견, "
                f"신뢰도 {confidence_score:.2f}, 소요시간 {search_time:.1f}초"
            )

            return result

        except Exception as e:
            logger.error(f"이슈 검색 실패: {str(e)}")
            # 폴백 결과 반환
            return self._create_fallback_result(keyword_result, time_period, time.time() - start_time)

    def _prepare_search_keywords(self, keyword_result: KeywordResult) -> List[str]:
        """검색을 위한 최적의 키워드 조합을 준비합니다"""
        # 우선순위: 핵심 키워드 → 관련 용어 → 동의어
        keywords = []
        keywords.extend(keyword_result.primary_keywords[:3])  # 최대 3개 핵심 키워드
        keywords.extend(keyword_result.related_terms[:2])     # 최대 2개 관련 용어

        # 중복 제거 및 길이 제한
        unique_keywords = list(dict.fromkeys(keywords))[:self.max_keywords_per_search]

        logger.debug(f"검색 키워드 준비 완료: {unique_keywords}")
        return unique_keywords

    def _parse_api_response(self, api_response: Dict[str, Any], search_keywords: List[str]) -> List[IssueItem]:
        """Perplexity API 응답을 파싱하여 IssueItem 리스트로 변환합니다"""
        issues = []

        try:
            # API 응답에서 메시지 내용 추출
            content = api_response['choices'][0]['message']['content']

            # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
            # 이는 Perplexity API의 실제 응답 형식에 맞춰 조정해야 함
            sections = content.split('**제목**:')

            for i, section in enumerate(sections[1:], 1):  # 첫 번째는 헤더이므로 제외
                try:
                    issue = self._parse_issue_section(section, i)
                    if issue:
                        issues.append(issue)
                except Exception as e:
                    logger.warning(f"이슈 섹션 파싱 실패 ({i}번째): {e}")
                    continue

            logger.info(f"API 응답 파싱 완료: {len(issues)}개 이슈 파싱됨")

        except Exception as e:
            logger.error(f"API 응답 파싱 실패: {e}")
            logger.debug(f"원본 응답: {api_response}")

        return issues

    def _parse_issue_section(self, section: str, index: int) -> Optional[IssueItem]:
        """개별 이슈 섹션을 파싱합니다"""
        try:
            lines = section.strip().split('\n')
            title = lines[0].strip()

            summary = ""
            source = "Unknown"
            published_date = None
            category = "news"

            # 각 라인을 파싱
            for line in lines[1:]:
                line = line.strip()
                if line.startswith('**요약**:'):
                    summary = line.replace('**요약**:', '').strip()
                elif line.startswith('**출처**:'):
                    source = line.replace('**출처**:', '').strip()
                elif line.startswith('**일자**:'):
                    published_date = line.replace('**일자**:', '').strip()
                elif line.startswith('**카테고리**:'):
                    category = line.replace('**카테고리**:', '').strip()

            if not title or not summary:
                return None

            return IssueItem(
                title=title,
                summary=summary,
                source=source,
                published_date=published_date,
                relevance_score=0.5,  # 기본값, 나중에 계산됨
                category=category,
                content_snippet=summary[:200]  # 요약의 일부
            )

        except Exception as e:
            logger.warning(f"이슈 섹션 파싱 오류: {e}")
            return None

    def _calculate_relevance_scores(self, issues: List[IssueItem], keyword_result: KeywordResult) -> List[IssueItem]:
        """각 이슈의 관련성 점수를 계산합니다"""
        all_keywords = []
        all_keywords.extend(keyword_result.primary_keywords)
        all_keywords.extend(keyword_result.related_terms)
        all_keywords.extend(keyword_result.synonyms)

        for issue in issues:
            score = 0.0
            total_text = f"{issue.title} {issue.summary}".lower()

            # 키워드 매칭 점수
            for keyword in all_keywords:
                if keyword.lower() in total_text:
                    if keyword in keyword_result.primary_keywords:
                        score += 0.3  # 핵심 키워드 가중치
                    elif keyword in keyword_result.related_terms:
                        score += 0.2  # 관련 용어 가중치
                    else:
                        score += 0.1  # 동의어 가중치

            # 신선도 점수 (최근일수록 높은 점수)
            if issue.published_date:
                # 실제로는 날짜 파싱이 필요하지만, 여기서는 간단히 처리
                score += 0.1

            # 출처 신뢰도 점수
            if any(trusted in issue.source.lower() for trusted in ['reuters', 'bbc', 'cnn', 'nyt']):
                score += 0.2

            issue.relevance_score = min(1.0, score)  # 최대 1.0으로 제한

        return issues

    def _calculate_confidence_score(self, issues: List[IssueItem], keyword_result: KeywordResult) -> float:
        """검색 결과의 전체 신뢰도를 계산합니다"""
        if not issues:
            return 0.0

        # 기본 신뢰도는 키워드 생성 신뢰도에서 시작
        base_confidence = keyword_result.confidence_score * 0.7

        # 이슈 개수 보너스
        count_bonus = min(0.2, len(issues) * 0.02)

        # 평균 관련성 점수 보너스
        avg_relevance = sum(issue.relevance_score for issue in issues) / len(issues)
        relevance_bonus = avg_relevance * 0.1

        total_confidence = base_confidence + count_bonus + relevance_bonus
        return min(1.0, total_confidence)

    def _create_fallback_result(self, keyword_result: KeywordResult, time_period: str, search_time: float) -> SearchResult:
        """검색 실패 시 폴백 결과를 생성합니다"""
        logger.warning("검색 실패로 인한 폴백 결과 생성")

        return SearchResult(
            query_keywords=keyword_result.primary_keywords[:3],
            total_found=0,
            issues=[],
            search_time=search_time,
            api_calls_used=0,
            confidence_score=0.1,  # 매우 낮은 신뢰도
            time_period=time_period,
            raw_responses=["검색 실패로 인한 응답 없음"]
        )

    def format_search_summary(self, result: SearchResult) -> str:
        """검색 결과를 요약 문자열로 포맷팅합니다"""
        if result.total_found == 0:
            return f"**이슈 검색 실패** (키워드: {', '.join(result.query_keywords[:3])})\n❌ 관련 이슈를 찾을 수 없습니다."

        confidence_percent = int(result.confidence_score * 100)

        summary = f"**이슈 검색 완료** (키워드: {', '.join(result.query_keywords[:3])})\n"
        summary += f"📊 총 {result.total_found}개 이슈 발견 | 신뢰도: {confidence_percent}% | 소요시간: {result.search_time:.1f}초\n\n"

        # 상위 3개 이슈 미리보기
        for i, issue in enumerate(result.issues[:3], 1):
            summary += f"**{i}. {issue.title}**\n"
            summary += f"   📰 {issue.source} | 관련도: {int(issue.relevance_score * 100)}%\n"
            summary += f"   📝 {issue.summary[:100]}{'...' if len(issue.summary) > 100 else ''}\n\n"

        if result.total_found > 3:
            summary += f"📋 추가 {result.total_found - 3}개 이슈가 더 있습니다.\n"

        return summary


# 편의 함수들
def create_issue_searcher(api_key: Optional[str] = None) -> IssueSearcher:
    """이슈 검색기 인스턴스를 생성합니다"""
    return IssueSearcher(api_key=api_key)


async def search_issues_for_keywords(keyword_result: KeywordResult, time_period: str = "최근 1주일") -> SearchResult:
    """키워드를 기반으로 이슈를 검색하는 편의 함수"""
    searcher = create_issue_searcher()
    return await searcher.search_issues_from_keywords(keyword_result, time_period)


if __name__ == "__main__":
    # pytest 실행 안내
    print("🔍 이슈 검색기 테스트")
    print("pytest로 테스트를 실행하세요:")
    print("pytest tests/test_issue_searcher.py -v")