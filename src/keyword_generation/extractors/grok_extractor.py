"""
Grok (X/Twitter) 기반 키워드 추출기.
"""

import asyncio
import time
import re
import json
from typing import List, Optional, Dict
from loguru import logger
import httpx

from ..base import BaseKeywordExtractor, KeywordExtractionResult, KeywordItem, KeywordImportance
from src.config import config


class GrokKeywordExtractor(BaseKeywordExtractor):
    """Grok API를 사용한 실시간 트렌드 키워드 추출기."""

    def __init__(self, api_key: Optional[str] = None):
        """Grok 추출기 초기화."""
        # API 키 설정: 인자로 받은 키가 없으면 config에서 가져옴
        self.api_key = api_key or config.get_grok_api_key()
        super().__init__("Grok", self.api_key)

        # API 설정
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self.model = config.get_grok_model()
        self.timeout = config.get_grok_timeout()
        self.max_retries = 3

        # API 키 확인
        if not self.api_key:
            logger.warning("Grok API 키가 없습니다. 시뮬레이션 모드로 동작합니다.")
            self.simulation_mode = True
        else:
            logger.info("Grok API 키가 설정되었습니다. 실제 API 모드로 동작합니다.")
            self.simulation_mode = False

        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json"
        }

        self.is_initialized = True
        logger.info(f"Grok 키워드 추출기 초기화 완료 (모드: {'시뮬레이션' if self.simulation_mode else '실제 API'})")

    async def extract_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        max_keywords: int = 20
    ) -> KeywordExtractionResult:
        """Grok을 사용하여 실시간 트렌드 키워드를 추출합니다."""
        start_time = time.time()
        logger.info(f"Grok 키워드 추출 시작: '{topic}'")

        try:
            if self.simulation_mode:
                keywords = await self._simulate_extraction(topic, context, max_keywords)
                raw_response = "시뮬레이션 모드"
            else:
                result = await self._real_extraction(topic, context, max_keywords)
                keywords = result['keywords']
                raw_response = result['raw_response']

            return KeywordExtractionResult(
                keywords=keywords,
                source_name=self.name,
                extraction_time=time.time() - start_time,
                raw_response=raw_response,
                metadata={
                    'mode': 'simulation' if self.simulation_mode else 'real_api',
                    'model': self.model if not self.simulation_mode else 'simulation'
                }
            )

        except Exception as e:
            logger.error(f"Grok 키워드 추출 실패: {e}")
            return KeywordExtractionResult(
                keywords=[],
                source_name=self.name,
                extraction_time=time.time() - start_time,
                error=str(e)
            )

    async def _real_extraction(
        self,
        topic: str,
        context: Optional[str],
        max_keywords: int
    ) -> Dict[str, any]:
        """실제 Grok API 호출."""
        prompt = self._build_extraction_prompt(topic, context, max_keywords)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are Grok, an AI with real-time access to X (Twitter) data. You specialize in identifying trending keywords, hashtags, and viral topics. Provide comprehensive keyword analysis with current social media trends."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.3,
            "stream": False
        }

        # API 호출 with 재시도 로직
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(
                    headers=self.headers,
                    timeout=httpx.Timeout(self.timeout)
                ) as client:
                    logger.debug(f"Grok API 호출 시도 {attempt + 1}/{self.max_retries}")
                    response = await client.post(self.base_url, json=payload)
                    response.raise_for_status()

                    data = response.json()
                    content = data['choices'][0]['message']['content']

                    # 키워드 파싱
                    keywords = self._parse_grok_response(content, topic)

                    logger.info(f"Grok API 호출 성공 (시도: {attempt + 1})")
                    return {
                        'keywords': keywords,
                        'raw_response': content
                    }

            except httpx.HTTPStatusError as e:
                logger.error(f"Grok API HTTP Error (Status: {e.response.status_code}): {e.response.text[:200]}")
                if e.response.status_code == 429:  # Rate Limit
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limit 초과. {wait_time}초 후 재시도... ({attempt + 1}/{self.max_retries})")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(wait_time)
                        continue
                raise

            except httpx.TimeoutException as e:
                logger.error(f"Grok API Timeout (Attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise ValueError(f"Grok API 호출이 {self.max_retries}번의 타임아웃으로 실패했습니다.")

                wait_time = min(2 ** attempt, 10)
                logger.warning(f"타임아웃으로 인해 {wait_time}초 후 재시도...")
                await asyncio.sleep(wait_time)

            except httpx.RequestError as e:
                error_detail = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Grok API Request Error (Attempt {attempt + 1}): {error_detail}")

                if attempt == self.max_retries - 1:
                    raise ValueError(f"Grok API 호출이 모든 재시도에 실패했습니다. 마지막 오류: {error_detail}")

                wait_time = min(3 ** attempt, 15)
                logger.warning(f"네트워크 오류로 인해 {wait_time}초 후 재시도...")
                await asyncio.sleep(wait_time)

        raise ValueError("Grok API 호출이 모든 재시도에 실패했습니다.")

    def _build_extraction_prompt(self, topic: str, context: Optional[str], max_keywords: int) -> str:
        """Grok API용 키워드 추출 프롬프트 생성."""
        prompt = f"""주제 "{topic}"에 대해 현재 X(Twitter)에서 트렌딩 중인 키워드를 분석해주세요.

다음 카테고리별로 키워드를 추출하여 JSON 형식으로 응답해주세요:

1. **해시태그 트렌드** (trending_hashtags): 현재 인기 해시태그
2. **실시간 토픽** (realtime_topics): 최신 뉴스나 이벤트 관련 키워드  
3. **커뮤니티 언급** (community_mentions): 사용자들이 자주 언급하는 키워드
4. **바이럴 키워드** (viral_keywords): 급상승 중인 용어들

각 카테고리별로 최대 {max_keywords//4}개씩 추출하고, 각 키워드에 대해 다음 정보를 포함해주세요:
- keyword: 키워드 자체
- confidence: 0.0-1.0 사이의 신뢰도
- trend_score: 0.0-1.0 사이의 트렌드 점수
- reason: 선택한 이유 (간단히)

응답 형식:
{{
  "trending_hashtags": [
    {{"keyword": "#키워드", "confidence": 0.9, "trend_score": 0.95, "reason": "이유"}},
    ...
  ],
  "realtime_topics": [...],
  "community_mentions": [...],
  "viral_keywords": [...]
}}"""

        if context:
            prompt += f"\n\n추가 맥락: {context}"

        return prompt

    def _parse_grok_response(self, content: str, topic: str) -> List[KeywordItem]:
        """Grok 응답에서 키워드 추출."""
        keywords = []

        try:
            # JSON 응답 파싱 시도
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # 카테고리별 키워드 처리
                category_mapping = {
                    'trending_hashtags': ('trending', KeywordImportance.HIGH),
                    'realtime_topics': ('realtime', KeywordImportance.HIGH),
                    'community_mentions': ('community', KeywordImportance.NORMAL),
                    'viral_keywords': ('viral', KeywordImportance.HIGH)
                }

                for category, keyword_list in data.items():
                    if category in category_mapping:
                        cat_name, importance = category_mapping[category]

                        for item in keyword_list:
                            if isinstance(item, dict) and 'keyword' in item:
                                keywords.append(KeywordItem(
                                    keyword=item['keyword'],
                                    sources=[self.name],
                                    importance=importance,
                                    confidence=item.get('confidence', 0.7),
                                    category=cat_name,
                                    metadata={
                                        'type': category,
                                        'trend_score': item.get('trend_score', 0.5),
                                        'reason': item.get('reason', ''),
                                        'source': 'grok_api'
                                    }
                                ))
            else:
                # JSON 파싱 실패 시 텍스트 파싱 시도
                keywords = self._parse_text_response(content, topic)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Grok 응답 파싱 실패: {e}")
            # 백업 파싱 시도
            keywords = self._parse_text_response(content, topic)

        return keywords

    def _parse_text_response(self, content: str, topic: str) -> List[KeywordItem]:
        """텍스트 응답에서 키워드 추출 (백업 파싱)."""
        keywords = []
        lines = content.split('\n')

        current_category = 'general'
        for line in lines:
            line = line.strip()

            # 카테고리 감지
            if any(cat in line.lower() for cat in ['hashtag', 'trending', 'viral']):
                current_category = 'trending'
            elif any(cat in line.lower() for cat in ['realtime', 'news', 'latest']):
                current_category = 'realtime'
            elif any(cat in line.lower() for cat in ['community', 'discussion']):
                current_category = 'community'

            # 키워드 추출 (단순 패턴)
            keyword_matches = re.findall(r'[#@]?\w+(?:\s+\w+)*', line)
            for match in keyword_matches:
                if len(match) > 2 and match.lower() not in ['the', 'and', 'for', 'with']:
                    keywords.append(KeywordItem(
                        keyword=match,
                        sources=[self.name],
                        importance=KeywordImportance.NORMAL,
                        confidence=0.6,
                        category=current_category,
                        metadata={'type': 'text_parsed', 'source': 'grok_api'}
                    ))

        return keywords[:20]  # 최대 20개로 제한

    async def _simulate_extraction(
        self,
        topic: str,
        context: Optional[str],
        max_keywords: int
    ) -> List[KeywordItem]:
        """Grok 키워드 추출 시뮬레이션."""
        # 실시간 트렌드를 시뮬레이션
        await asyncio.sleep(0.5)  # API 호출 시뮬레이션

        keyword_items = []
        base_topic = self.preprocess_topic(topic)

        # 트렌딩 해시태그 스타일 키워드 (HIGH importance)
        trending_keywords = [
            f"#{base_topic}",
            f"{base_topic}_trending",
            f"{base_topic}2024",
            f"breaking_{base_topic}"
        ]

        for kw in trending_keywords[:max_keywords // 3]:
            keyword_items.append(KeywordItem(
                keyword=kw,
                sources=[self.name],
                importance=KeywordImportance.HIGH,
                confidence=0.85,
                category='trending',
                metadata={'type': 'hashtag', 'trend_score': 0.9, 'source': 'simulation'}
            ))

        # 실시간 토픽 키워드 (NORMAL importance)
        realtime_keywords = [
            f"{base_topic} news",
            f"{base_topic} update",
            f"latest {base_topic}",
            f"{base_topic} announcement"
        ]

        for kw in realtime_keywords[:max_keywords // 3]:
            keyword_items.append(KeywordItem(
                keyword=kw,
                sources=[self.name],
                importance=KeywordImportance.NORMAL,
                confidence=0.75,
                category='realtime',
                metadata={'type': 'topic', 'source': 'simulation'}
            ))

        # 커뮤니티 언급 키워드 (NORMAL to LOW)
        community_keywords = [
            f"{base_topic} discussion",
            f"{base_topic} community",
            f"{base_topic} opinion",
            f"{base_topic} debate"
        ]

        for kw in community_keywords[:max_keywords // 3]:
            keyword_items.append(KeywordItem(
                keyword=kw,
                sources=[self.name],
                importance=KeywordImportance.NORMAL,
                confidence=0.65,
                category='community',
                metadata={'type': 'discussion', 'source': 'simulation'}
            ))

        return keyword_items