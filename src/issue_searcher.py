"""
이슈 검색 모듈 - Perplexity API 연동
생성된 키워드를 기반으로 실시간 이슈를 검색하고 세부 정보를 수집하는 모듈
"""

import asyncio
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import httpx
from loguru import logger

from src.config import config
from src.keyword_generator import KeywordResult


@dataclass
class EntityInfo:
    """관련 인물/기관 정보"""
    name: str  # 인물/기관명
    role: str  # 역할/직책
    relevance: float  # 관련도 (0.0-1.0)
    entity_type: str  # 'person', 'organization', 'company', 'government'
    description: str  # 간단한 설명


@dataclass
class ImpactAnalysis:
    """영향도 분석 정보"""
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    impact_score: float  # 수치적 영향도 (0.0-1.0)
    affected_sectors: List[str]  # 영향받는 분야
    geographic_scope: str  # 'local', 'national', 'regional', 'global'
    time_sensitivity: str  # 'immediate', 'short-term', 'long-term'
    reasoning: str  # 영향도 판단 근거


@dataclass
class TimelineEvent:
    """시간순 이벤트 정보"""
    date: str  # 이벤트 발생일
    event_type: str  # 'announcement', 'development', 'reaction', 'consequence'
    description: str  # 이벤트 설명
    importance: float  # 중요도 (0.0-1.0)
    source: str  # 정보 출처


@dataclass
class IssueItem:
    """개별 이슈 정보를 담는 데이터 클래스 - 확장됨"""
    title: str  # 이슈 제목
    summary: str  # 이슈 요약
    source: str  # 출처 (URL 또는 매체명)
    published_date: Optional[str]  # 발행일
    relevance_score: float  # 관련성 점수 (0.0-1.0)
    category: str  # 카테고리 (news, blog, social, academic)
    content_snippet: str  # 내용 일부

    # 4단계 추가 정보
    detailed_content: Optional[str] = None  # 상세 내용
    related_entities: List[EntityInfo] = None  # 관련 인물/기관
    impact_analysis: Optional[ImpactAnalysis] = None  # 영향도 분석
    timeline_events: List[TimelineEvent] = None  # 시간순 이벤트
    background_context: Optional[str] = None  # 배경 정보
    detail_collection_time: Optional[float] = None  # 세부 정보 수집 시간
    detail_confidence: Optional[float] = None  # 세부 정보 신뢰도


@dataclass
class SearchResult:
    """이슈 검색 결과를 담는 데이터 클래스 - 확장됨"""
    query_keywords: List[str]  # 검색에 사용된 키워드
    total_found: int  # 총 발견된 이슈 수
    issues: List[IssueItem]  # 이슈 목록
    search_time: float  # 검색 소요 시간 (초)
    api_calls_used: int  # 사용된 API 호출 수
    confidence_score: float  # 검색 결과 신뢰도
    time_period: str  # 검색 기간
    raw_responses: List[str]  # 원본 API 응답들

    # 4단계 추가 정보
    detailed_issues_count: int = 0  # 세부 정보가 수집된 이슈 수
    total_detail_collection_time: float = 0.0  # 총 세부 정보 수집 시간
    average_detail_confidence: float = 0.0  # 평균 세부 정보 신뢰도


class PerplexityClient:
    """Perplexity API 클라이언트 - 4단계 기능 추가"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.get_perplexity_api_key()
        if not self.api_key:
            raise ValueError("Perplexity API 키가 설정되지 않았습니다")

        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-large-128k-online"
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
        키워드를 사용하여 이슈를 검색합니다 (기존 메서드)
        """
        # 기존 구현 유지
        keyword_str = ", ".join(keywords[:5])

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

        return await self._make_api_call(prompt)

    async def collect_detailed_information(
            self,
            issue_title: str,
            issue_summary: str,
            original_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        특정 이슈에 대한 세부 정보를 수집합니다 (4단계 신규)
        """
        keywords_str = ", ".join(original_keywords[:3])

        prompt = f"""다음 이슈에 대한 상세한 분석을 제공해주세요:

**이슈 제목**: {issue_title}
**기본 요약**: {issue_summary}
**관련 키워드**: {keywords_str}

다음 정보를 포함하여 분석해주세요:

1. **상세 내용**: 이슈의 구체적인 내용과 배경
2. **관련 인물/기관**: 
   - 이름, 역할, 관련도 (높음/중간/낮음)
   - 유형: 개인/기업/정부기관/국제기구
3. **영향도 분석**:
   - 영향 수준: 낮음/중간/높음/매우높음
   - 영향받는 분야들
   - 지리적 범위: 지역/국가/지역권/글로벌
   - 시간 민감도: 즉시/단기/장기
4. **시간순 전개**:
   - 주요 이벤트들의 시간순 나열
   - 각 이벤트의 중요도
5. **배경 정보**: 이슈를 이해하기 위한 맥락 정보

가능한 한 구체적이고 객관적인 정보를 제공해주세요."""

        logger.info(f"세부 정보 수집 API 호출: {issue_title[:50]}...")
        return await self._make_api_call(prompt)

    async def extract_entities_and_impact(
            self,
            issue_title: str,
            detailed_content: str
    ) -> Dict[str, Any]:
        """
        이슈에서 관련 인물/기관 및 영향도를 추출합니다 (4단계 신규)
        """
        prompt = f"""다음 이슈 내용을 분석하여 JSON 형식으로 정보를 추출해주세요:

**이슈**: {issue_title}
**내용**: {detailed_content[:1000]}

다음 JSON 형식으로 응답해주세요:
{{
    "entities": [
        {{
            "name": "인물/기관명",
            "role": "역할/직책",
            "relevance": 0.8,
            "entity_type": "person|organization|company|government",
            "description": "간단한 설명"
        }}
    ],
    "impact": {{
        "impact_level": "low|medium|high|critical",
        "impact_score": 0.7,
        "affected_sectors": ["기술", "경제", "정치"],
        "geographic_scope": "local|national|regional|global",
        "time_sensitivity": "immediate|short-term|long-term",
        "reasoning": "영향도 판단 근거"
    }},
    "confidence": 0.85
}}

객관적이고 구체적인 정보만 포함해주세요."""

        logger.info(f"엔티티 및 영향도 추출: {issue_title[:30]}...")
        return await self._make_api_call(prompt)

    async def extract_timeline(
            self,
            issue_title: str,
            detailed_content: str
    ) -> Dict[str, Any]:
        """
        이슈의 시간순 전개를 추출합니다 (4단계 신규)
        """
        prompt = f"""다음 이슈의 시간순 전개를 JSON 형식으로 정리해주세요:

**이슈**: {issue_title}
**내용**: {detailed_content[:1000]}

다음 JSON 형식으로 응답해주세요:
{{
    "timeline": [
        {{
            "date": "2024-01-15",
            "event_type": "announcement|development|reaction|consequence",
            "description": "이벤트 설명",
            "importance": 0.8,
            "source": "정보 출처"
        }}
    ],
    "background_context": "이슈를 이해하기 위한 배경 정보",
    "confidence": 0.9
}}

시간순으로 정렬하여 중요한 이벤트들만 포함해주세요."""

        logger.info(f"타임라인 추출: {issue_title[:30]}...")
        return await self._make_api_call(prompt)

    async def _make_api_call(self, prompt: str) -> Dict[str, Any]:
        """
        공통 API 호출 메서드
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "당신은 정확하고 객관적인 정보 분석 전문가입니다. 신뢰할 수 있는 소스의 정보만 사용하고, 요청된 형식에 맞춰 응답해주세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2500,
            "temperature": 0.3,
            "search_recency_filter": "week",
            "return_related_questions": False,
            "return_images": False
        }

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
                        logger.debug(f"API 호출 성공 (시도 {attempt + 1})")
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
                        logger.error(f"API 오류 (상태: {response.status_code}): {error_text}")
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
    이슈 검색기 - 4단계 세부 정보 수집 기능 추가

    주요 기능:
    - 키워드 기반 이슈 검색 (기존)
    - 검색 결과 파싱 및 구조화 (기존)
    - 관련성 점수 계산 (기존)
    - 세부 정보 수집 (신규)
    - 관련 인물/기관 추출 (신규)
    - 영향도 분석 (신규)
    - 시간순 전개 추적 (신규)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = PerplexityClient(api_key)
        self.max_keywords_per_search = 5
        self.max_results_per_search = 10

        # 4단계 설정
        self.enable_detailed_collection = True
        self.max_detailed_issues = 10  # 세부 정보를 수집할 최대 이슈 수
        self.detail_collection_timeout = 60  # 각 이슈별 세부 정보 수집 타임아웃

        logger.info("IssueSearcher 초기화 완료 (4단계 세부 정보 수집 지원)")

    async def search_issues_from_keywords(
            self,
            keyword_result: KeywordResult,
            time_period: str = "최근 1주일",
            max_total_results: int = 20,
            collect_details: bool = True
    ) -> SearchResult:
        """
        키워드 생성 결과를 기반으로 이슈를 검색하고 세부 정보를 수집합니다 (확장됨)

        Args:
            keyword_result: 키워드 생성 결과
            time_period: 검색 기간
            max_total_results: 최대 총 결과 수
            collect_details: 세부 정보 수집 여부 (4단계)

        Returns:
            SearchResult: 검색 결과 (세부 정보 포함)
        """
        start_time = time.time()
        logger.info(f"이슈 검색 시작: 주제='{keyword_result.topic}', 기간='{time_period}', 세부수집={collect_details}")

        try:
            # 1단계: 기본 이슈 검색 (기존 로직)
            search_keywords = self._prepare_search_keywords(keyword_result)
            api_response = await self.client.search_issues(
                keywords=search_keywords,
                time_period=time_period,
                max_results=max_total_results
            )

            # 2단계: 응답 파싱 (기존 로직)
            issues = self._parse_api_response(api_response, search_keywords)
            scored_issues = self._calculate_relevance_scores(issues, keyword_result)
            top_issues = sorted(scored_issues, key=lambda x: x.relevance_score, reverse=True)[:max_total_results]

            # 3단계: 세부 정보 수집 (신규 로직)
            detailed_issues_count = 0
            total_detail_time = 0.0
            detail_confidences = []

            if collect_details and self.enable_detailed_collection and top_issues:
                logger.info(f"4단계 세부 정보 수집 시작: {min(len(top_issues), self.max_detailed_issues)}개 이슈")

                # 상위 이슈들에 대해서만 세부 정보 수집
                issues_to_detail = top_issues[:self.max_detailed_issues]

                for i, issue in enumerate(issues_to_detail):
                    try:
                        detail_start = time.time()
                        logger.info(f"세부 정보 수집 중 ({i + 1}/{len(issues_to_detail)}): {issue.title[:50]}...")

                        # 세부 정보 수집 실행
                        enhanced_issue = await self._collect_issue_details(issue, search_keywords)

                        # 원본 이슈를 업데이트된 이슈로 교체
                        top_issues[top_issues.index(issue)] = enhanced_issue

                        detail_time = time.time() - detail_start
                        total_detail_time += detail_time
                        detailed_issues_count += 1

                        if enhanced_issue.detail_confidence:
                            detail_confidences.append(enhanced_issue.detail_confidence)

                        logger.success(f"세부 정보 수집 완료 ({i + 1}/{len(issues_to_detail)}): {detail_time:.1f}초")

                        # API 제한을 고려한 지연
                        if i < len(issues_to_detail) - 1:
                            await asyncio.sleep(1)

                    except asyncio.TimeoutError:
                        logger.warning(f"세부 정보 수집 타임아웃: {issue.title[:50]}")
                        continue
                    except Exception as e:
                        logger.error(f"세부 정보 수집 실패: {issue.title[:50]} - {str(e)}")
                        continue

            # 4단계: 최종 결과 생성
            confidence_score = self._calculate_confidence_score(top_issues, keyword_result)
            search_time = time.time() - start_time

            # 원본 응답 저장
            try:
                raw_response_str = json.dumps(api_response, ensure_ascii=False, indent=2)
            except (TypeError, ValueError) as e:
                logger.warning(f"API 응답 JSON 직렬화 실패: {e}")
                raw_response_str = str(api_response)

            # 평균 세부 정보 신뢰도 계산
            avg_detail_confidence = sum(detail_confidences) / len(detail_confidences) if detail_confidences else 0.0

            result = SearchResult(
                query_keywords=search_keywords,
                total_found=len(top_issues),
                issues=top_issues,
                search_time=search_time,
                api_calls_used=1 + detailed_issues_count * 2,  # 기본 검색 + 각 이슈별 2회 호출
                confidence_score=confidence_score,
                time_period=time_period,
                raw_responses=[raw_response_str],
                detailed_issues_count=detailed_issues_count,
                total_detail_collection_time=total_detail_time,
                average_detail_confidence=avg_detail_confidence
            )

            logger.success(
                f"이슈 검색 완료: {len(top_issues)}개 이슈 (세부정보 {detailed_issues_count}개), "
                f"신뢰도 {confidence_score:.2f}, 총 소요시간 {search_time:.1f}초"
            )

            return result

        except Exception as e:
            logger.error(f"이슈 검색 실패: {str(e)}")
            return self._create_fallback_result(keyword_result, time_period, time.time() - start_time)

    async def _collect_issue_details(
            self,
            issue: IssueItem,
            original_keywords: List[str]
    ) -> IssueItem:
        """
        개별 이슈의 세부 정보를 수집합니다 (4단계 신규)
        """
        detail_start_time = time.time()

        try:
            # 1. 기본 세부 정보 수집
            detailed_response = await asyncio.wait_for(
                self.client.collect_detailed_information(
                    issue.title,
                    issue.summary,
                    original_keywords
                ),
                timeout=self.detail_collection_timeout
            )

            detailed_content = self._extract_detailed_content(detailed_response)

            # 2. 병렬로 엔티티/영향도 및 타임라인 수집
            entity_task = asyncio.create_task(
                self.client.extract_entities_and_impact(issue.title, detailed_content)
            )
            timeline_task = asyncio.create_task(
                self.client.extract_timeline(issue.title, detailed_content)
            )

            # 결과 대기
            entity_response, timeline_response = await asyncio.gather(
                entity_task, timeline_task, return_exceptions=True
            )

            # 3. 응답 파싱
            entities, impact = self._parse_entity_and_impact_response(entity_response)
            timeline_events, background_context = self._parse_timeline_response(timeline_response)

            # 4. 신뢰도 계산
            detail_confidence = self._calculate_detail_confidence(
                detailed_content, entities, impact, timeline_events
            )

            # 5. 기존 IssueItem 업데이트
            issue.detailed_content = detailed_content
            issue.related_entities = entities
            issue.impact_analysis = impact
            issue.timeline_events = timeline_events
            issue.background_context = background_context
            issue.detail_collection_time = time.time() - detail_start_time
            issue.detail_confidence = detail_confidence

            return issue

        except asyncio.TimeoutError:
            logger.warning(f"세부 정보 수집 타임아웃: {issue.title[:50]}")
            raise
        except Exception as e:
            logger.error(f"세부 정보 수집 중 오류: {str(e)}")
            raise

    def _extract_detailed_content(self, api_response: Dict[str, Any]) -> str:
        """API 응답에서 상세 내용을 추출합니다"""
        try:
            content = api_response['choices'][0]['message']['content']
            # 상세 내용 섹션 추출 (간단한 파싱)
            if "**상세 내용**" in content:
                sections = content.split("**상세 내용**:")
                if len(sections) > 1:
                    detailed_section = sections[1].split("**")[0].strip()
                    return detailed_section
            return content[:1000]  # 폴백: 처음 1000자
        except Exception as e:
            logger.warning(f"상세 내용 추출 실패: {e}")
            return "상세 내용 추출 실패"

    def _parse_entity_and_impact_response(
            self,
            response: Any
    ) -> Tuple[List[EntityInfo], Optional[ImpactAnalysis]]:
        """엔티티 및 영향도 응답을 파싱합니다"""
        entities = []
        impact = None

        try:
            if isinstance(response, Exception):
                logger.warning(f"엔티티/영향도 수집 실패: {response}")
                return entities, impact

            content = response['choices'][0]['message']['content']

            # JSON 추출 시도
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # 엔티티 파싱
                for entity_data in data.get('entities', []):
                    entity = EntityInfo(
                        name=entity_data.get('name', ''),
                        role=entity_data.get('role', ''),
                        relevance=float(entity_data.get('relevance', 0.5)),
                        entity_type=entity_data.get('entity_type', 'unknown'),
                        description=entity_data.get('description', '')
                    )
                    entities.append(entity)

                # 영향도 파싱
                impact_data = data.get('impact', {})
                if impact_data:
                    impact = ImpactAnalysis(
                        impact_level=impact_data.get('impact_level', 'medium'),
                        impact_score=float(impact_data.get('impact_score', 0.5)),
                        affected_sectors=impact_data.get('affected_sectors', []),
                        geographic_scope=impact_data.get('geographic_scope', 'national'),
                        time_sensitivity=impact_data.get('time_sensitivity', 'short-term'),
                        reasoning=impact_data.get('reasoning', '')
                    )

        except Exception as e:
            logger.warning(f"엔티티/영향도 파싱 실패: {e}")

        return entities, impact

    def _parse_timeline_response(
            self,
            response: Any
    ) -> Tuple[List[TimelineEvent], str]:
        """타임라인 응답을 파싱합니다"""
        timeline_events = []
        background_context = ""

        try:
            if isinstance(response, Exception):
                logger.warning(f"타임라인 수집 실패: {response}")
                return timeline_events, background_context

            content = response['choices'][0]['message']['content']

            # JSON 추출 시도
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # 타임라인 파싱
                for event_data in data.get('timeline', []):
                    event = TimelineEvent(
                        date=event_data.get('date', ''),
                        event_type=event_data.get('event_type', 'development'),
                        description=event_data.get('description', ''),
                        importance=float(event_data.get('importance', 0.5)),
                        source=event_data.get('source', '')
                    )
                    timeline_events.append(event)

                # 배경 정보
                background_context = data.get('background_context', '')

        except Exception as e:
            logger.warning(f"타임라인 파싱 실패: {e}")

        return timeline_events, background_context

    def _calculate_detail_confidence(
            self,
            detailed_content: str,
            entities: List[EntityInfo],
            impact: Optional[ImpactAnalysis],
            timeline_events: List[TimelineEvent]
    ) -> float:
        confidence = 0.0

        # 내용 길이 점수 (최대 0.2) - 기존 0.3에서 축소
        if len(detailed_content) > 100:
            confidence += 0.2
        elif len(detailed_content) > 50:
            confidence += 0.15
        else:
            confidence += 0.1

        # 엔티티 정보 점수 (최대 0.4) - 기존 0.3에서 확대
        if entities:
            entity_score = min(0.3, len(entities) * 0.15)
            # 고품질 엔티티 보너스 강화
            high_relevance_entities = [e for e in entities if e.relevance > 0.8]
            if high_relevance_entities:
                entity_score += 0.1
            confidence += entity_score

        # 영향도 분석 점수 (최대 0.3) - 기존 0.2에서 확대
        if impact:
            confidence += 0.2
            # 구체적인 영향도 분석 보너스 강화
            if impact.affected_sectors and len(impact.affected_sectors) > 0:
                confidence += 0.1

        # 타임라인 점수 (최대 0.2) - 유지
        if timeline_events:
            timeline_score = min(0.2, len(timeline_events) * 0.1)
            confidence += timeline_score

        return min(1.0, confidence)

    # 기존 메서드들 유지 (변경 없음)
    def _prepare_search_keywords(self, keyword_result: KeywordResult) -> List[str]:
        """검색을 위한 최적의 키워드 조합을 준비합니다"""
        keywords = []
        keywords.extend(keyword_result.primary_keywords[:3])
        keywords.extend(keyword_result.related_terms[:2])
        unique_keywords = list(dict.fromkeys(keywords))[:self.max_keywords_per_search]
        logger.debug(f"검색 키워드 준비 완료: {unique_keywords}")
        return unique_keywords

    def _parse_api_response(self, api_response: Dict[str, Any], search_keywords: List[str]) -> List[IssueItem]:
        """Perplexity API 응답을 파싱하여 IssueItem 리스트로 변환합니다"""
        issues = []

        try:
            content = api_response['choices'][0]['message']['content']
            sections = content.split('**제목**:')

            for i, section in enumerate(sections[1:], 1):
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
                relevance_score=0.5,
                category=category,
                content_snippet=summary[:200],
                # 4단계 필드들은 초기값으로 설정
                related_entities=[],
                timeline_events=[]
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

            for keyword in all_keywords:
                if keyword.lower() in total_text:
                    if keyword in keyword_result.primary_keywords:
                        score += 0.3
                    elif keyword in keyword_result.related_terms:
                        score += 0.2
                    else:
                        score += 0.1

            if issue.published_date:
                score += 0.1

            if any(trusted in issue.source.lower() for trusted in ['reuters', 'bbc', 'cnn', 'nyt']):
                score += 0.2

            issue.relevance_score = min(1.0, score)

        return issues

    def _calculate_confidence_score(self, issues: List[IssueItem], keyword_result: KeywordResult) -> float:
        """검색 결과의 전체 신뢰도를 계산합니다"""
        if not issues:
            return 0.0

        base_confidence = keyword_result.confidence_score * 0.7
        count_bonus = min(0.2, len(issues) * 0.02)
        avg_relevance = sum(issue.relevance_score for issue in issues) / len(issues)
        relevance_bonus = avg_relevance * 0.1

        total_confidence = base_confidence + count_bonus + relevance_bonus
        return min(1.0, total_confidence)

    def _create_fallback_result(self, keyword_result: KeywordResult, time_period: str,
                                search_time: float) -> SearchResult:
        """검색 실패 시 폴백 결과를 생성합니다"""
        logger.warning("검색 실패로 인한 폴백 결과 생성")

        return SearchResult(
            query_keywords=keyword_result.primary_keywords[:3],
            total_found=0,
            issues=[],
            search_time=search_time,
            api_calls_used=0,
            confidence_score=0.1,
            time_period=time_period,
            raw_responses=["검색 실패로 인한 응답 없음"]
        )

    def format_search_summary(self, result: SearchResult) -> str:
        """검색 결과를 요약 문자열로 포맷팅합니다 (4단계 정보 포함)"""
        if result.total_found == 0:
            return f"**이슈 검색 실패** (키워드: {', '.join(result.query_keywords[:3])})\n❌ 관련 이슈를 찾을 수 없습니다."

        confidence_percent = int(result.confidence_score * 100)
        detail_confidence_percent = int(
            result.average_detail_confidence * 100) if result.average_detail_confidence > 0 else 0

        summary = f"**이슈 검색 완료** (키워드: {', '.join(result.query_keywords[:3])})\n"
        summary += f"📊 총 {result.total_found}개 이슈 발견 | 신뢰도: {confidence_percent}%"

        if result.detailed_issues_count > 0:
            summary += f" | 세부정보: {result.detailed_issues_count}개 ({detail_confidence_percent}%)"

        summary += f" | 소요시간: {result.search_time:.1f}초\n\n"

        # 상위 이슈들 미리보기 (세부 정보 포함)
        for i, issue in enumerate(result.issues[:3], 1):
            summary += f"**{i}. {issue.title}**\n"
            summary += f"   📰 {issue.source} | 관련도: {int(issue.relevance_score * 100)}%"

            # 세부 정보 추가 표시
            if issue.detail_confidence and issue.detail_confidence > 0:
                summary += f" | 세부신뢰도: {int(issue.detail_confidence * 100)}%"

            summary += "\n"
            summary += f"   📝 {issue.summary[:100]}{'...' if len(issue.summary) > 100 else ''}\n"

            # 관련 인물/기관 표시 (있는 경우)
            if issue.related_entities and len(issue.related_entities) > 0:
                top_entities = [e.name for e in
                                sorted(issue.related_entities, key=lambda x: x.relevance, reverse=True)[:2]]
                summary += f"   👥 관련: {', '.join(top_entities)}\n"

            # 영향도 표시 (있는 경우)
            if issue.impact_analysis:
                impact_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(
                    issue.impact_analysis.impact_level, "⚪")
                summary += f"   {impact_emoji} 영향도: {issue.impact_analysis.impact_level} ({issue.impact_analysis.geographic_scope})\n"

            summary += "\n"

        if result.total_found > 3:
            summary += f"📋 추가 {result.total_found - 3}개 이슈가 더 있습니다.\n"

        # 세부 정보 수집 요약
        if result.detailed_issues_count > 0:
            summary += f"\n🔍 **세부 분석**: {result.detailed_issues_count}개 이슈에 대한 상세 정보 수집 완료 "
            summary += f"(소요시간: {result.total_detail_collection_time:.1f}초)\n"

        return summary

    def format_detailed_issue_report(self, issue: IssueItem) -> str:
        """개별 이슈의 상세 보고서를 포맷팅합니다 (4단계 신규)"""
        if not issue.detailed_content:
            return f"**{issue.title}**\n📝 {issue.summary}\n📰 출처: {issue.source}"

        report = f"# 📋 상세 이슈 분석: {issue.title}\n\n"

        # 기본 정보
        report += f"**📰 출처**: {issue.source}\n"
        if issue.published_date:
            report += f"**📅 발행일**: {issue.published_date}\n"
        report += f"**🔍 관련도**: {int(issue.relevance_score * 100)}%\n"
        if issue.detail_confidence:
            report += f"**🎯 세부신뢰도**: {int(issue.detail_confidence * 100)}%\n"
        report += "\n"

        # 요약
        report += f"## 📝 요약\n{issue.summary}\n\n"

        # 상세 내용
        if issue.detailed_content:
            report += f"## 📖 상세 내용\n{issue.detailed_content}\n\n"

        # 관련 인물/기관
        if issue.related_entities and len(issue.related_entities) > 0:
            report += "## 👥 관련 인물/기관\n"
            for entity in sorted(issue.related_entities, key=lambda x: x.relevance, reverse=True):
                entity_emoji = {"person": "👤", "organization": "🏢", "company": "🏭", "government": "🏛️"}.get(
                    entity.entity_type, "📋")
                report += f"- {entity_emoji} **{entity.name}** ({entity.role})\n"
                report += f"  - 관련도: {int(entity.relevance * 100)}%\n"
                if entity.description:
                    report += f"  - {entity.description}\n"
                report += "\n"

        # 영향도 분석
        if issue.impact_analysis:
            impact = issue.impact_analysis
            impact_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(impact.impact_level, "⚪")

            report += f"## {impact_emoji} 영향도 분석\n"
            report += f"- **영향 수준**: {impact.impact_level} ({int(impact.impact_score * 100)}%)\n"
            report += f"- **지리적 범위**: {impact.geographic_scope}\n"
            report += f"- **시간 민감도**: {impact.time_sensitivity}\n"

            if impact.affected_sectors:
                report += f"- **영향받는 분야**: {', '.join(impact.affected_sectors)}\n"

            if impact.reasoning:
                report += f"- **판단 근거**: {impact.reasoning}\n"
            report += "\n"

        # 시간순 전개
        if issue.timeline_events and len(issue.timeline_events) > 0:
            report += "## ⏰ 시간순 전개\n"
            sorted_events = sorted(issue.timeline_events, key=lambda x: x.date)

            for event in sorted_events:
                event_emoji = {
                    "announcement": "📢",
                    "development": "🔄",
                    "reaction": "💬",
                    "consequence": "⚡"
                }.get(event.event_type, "📌")

                importance_stars = "⭐" * int(event.importance * 5)
                report += f"- {event_emoji} **{event.date}** ({event.event_type}) {importance_stars}\n"
                report += f"  {event.description}\n"
                if event.source:
                    report += f"  📰 출처: {event.source}\n"
                report += "\n"

        # 배경 정보
        if issue.background_context:
            report += f"## 🔗 배경 정보\n{issue.background_context}\n\n"

        # 메타 정보
        if issue.detail_collection_time:
            report += f"---\n*세부 정보 수집 시간: {issue.detail_collection_time:.1f}초*\n"

        return report


# 편의 함수들 (기존 + 확장)
def create_issue_searcher(api_key: Optional[str] = None) -> IssueSearcher:
    """이슈 검색기 인스턴스를 생성합니다"""
    return IssueSearcher(api_key=api_key)


async def search_issues_for_keywords(
        keyword_result: KeywordResult,
        time_period: str = "최근 1주일",
        collect_details: bool = True
) -> SearchResult:
    """키워드를 기반으로 이슈를 검색하는 편의 함수 (4단계 지원)"""
    searcher = create_issue_searcher()
    return await searcher.search_issues_from_keywords(keyword_result, time_period, collect_details=collect_details)


# 4단계 전용 편의 함수들
async def get_detailed_issue_analysis(issue_title: str, issue_summary: str, keywords: List[str]) -> Dict[str, Any]:
    """특정 이슈에 대한 상세 분석을 가져오는 편의 함수"""
    client = PerplexityClient()
    return await client.collect_detailed_information(issue_title, issue_summary, keywords)


def create_detailed_report_from_search_result(search_result: SearchResult) -> str:
    """SearchResult에서 모든 상세 이슈 보고서를 생성하는 편의 함수"""
    if not search_result.issues:
        return "상세 분석할 이슈가 없습니다."

    searcher = create_issue_searcher()
    full_report = f"# 🔍 종합 이슈 분석 보고서\n\n"
    full_report += f"**검색 키워드**: {', '.join(search_result.query_keywords)}\n"
    full_report += f"**검색 기간**: {search_result.time_period}\n"
    full_report += f"**총 이슈 수**: {search_result.total_found}개\n"
    full_report += f"**세부 분석 이슈**: {search_result.detailed_issues_count}개\n"
    full_report += f"**전체 신뢰도**: {int(search_result.confidence_score * 100)}%\n\n"

    full_report += "---\n\n"

    # 세부 정보가 있는 이슈들만 상세 보고서 생성
    detailed_issues = [issue for issue in search_result.issues if issue.detailed_content]

    if detailed_issues:
        for i, issue in enumerate(detailed_issues, 1):
            full_report += searcher.format_detailed_issue_report(issue)
            if i < len(detailed_issues):
                full_report += "\n---\n\n"
    else:
        full_report += "세부 분석된 이슈가 없습니다.\n"

    return full_report


if __name__ == "__main__":
    print("🔍 이슈 검색기 테스트 (4단계 세부 정보 수집 지원)")
    print("pytest로 테스트를 실행하세요:")
    print("pytest tests/test_issue_searcher.py -v")