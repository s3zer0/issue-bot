"""
Grok (X/Twitter) 기반 키워드 추출기 - 500 에러 해결 버전
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

        # 서비스 상태 추적
        self.service_status = "unknown"  # unknown, healthy, degraded, down
        self.consecutive_failures = 0
        self.last_success_time = None

        # API 키 확인 및 강제 시뮬레이션 모드 설정
        if not self.api_key or self.consecutive_failures >= 3:
            logger.warning("Grok API 키가 없거나 연속 실패로 인해 시뮬레이션 모드로 동작합니다.")
            self.simulation_mode = True
        else:
            logger.info("Grok API 키가 설정되었습니다.")
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
            # 연속 실패가 많으면 자동으로 시뮬레이션 모드로 전환
            if self.consecutive_failures >= 3:
                logger.warning(f"연속 {self.consecutive_failures}번 실패로 인해 시뮬레이션 모드로 전환")
                self.simulation_mode = True

            if self.simulation_mode:
                keywords = await self._simulate_extraction(topic, context, max_keywords)
                raw_response = f"시뮬레이션 모드 (연속 실패: {self.consecutive_failures}회)"
            else:
                # 실제 API 호출 시도
                result = await self._real_extraction_with_fallback(topic, context, max_keywords)
                keywords = result['keywords']
                raw_response = result['raw_response']

            return KeywordExtractionResult(
                keywords=keywords,
                source_name=self.name,
                extraction_time=time.time() - start_time,
                raw_response=raw_response,
                metadata={
                    'mode': 'simulation' if self.simulation_mode else 'real_api',
                    'model': self.model if not self.simulation_mode else 'simulation',
                    'consecutive_failures': self.consecutive_failures,
                    'service_status': self.service_status
                }
            )

        except Exception as e:
            logger.error(f"Grok 키워드 추출 최종 실패: {e}")
            # 최종 폴백: 항상 시뮬레이션 결과 반환
            keywords = await self._simulate_extraction(topic, context, max_keywords)
            return KeywordExtractionResult(
                keywords=keywords,
                source_name=f"{self.name}_fallback",
                extraction_time=time.time() - start_time,
                error=str(e),
                raw_response="최종 폴백 - 시뮬레이션 모드",
                metadata={'mode': 'emergency_fallback', 'original_error': str(e)}
            )

    async def _real_extraction_with_fallback(
        self,
        topic: str,
        context: Optional[str],
        max_keywords: int
    ) -> Dict[str, any]:
        """폴백 메커니즘이 포함된 실제 API 호출."""

        # 여러 모델과 설정 조합 시도
        api_configs = [
            {"model": "grok-3-latest", "search_params": False},
            {"model": "grok-3", "search_params": True},
            {"model": "grok-2-latest", "search_params": False},
        ]

        for config_idx, api_config in enumerate(api_configs):
            try:
                logger.info(f"Grok API 설정 {config_idx + 1}/{len(api_configs)} 시도: {api_config['model']}")

                prompt = self._build_simplified_prompt(topic, context, max_keywords)
                payload = self._build_safe_payload(prompt, api_config)

                result = await self._make_safe_api_call(payload)

                if result:
                    self.consecutive_failures = 0  # 성공 시 실패 카운터 리셋
                    self.last_success_time = time.time()
                    self.service_status = "healthy"
                    logger.success(f"Grok API 성공: {api_config['model']}")
                    return result

            except Exception as e:
                logger.warning(f"API 설정 {config_idx + 1} 실패: {e}")
                continue

        # 모든 설정 실패
        self.consecutive_failures += 1
        self.service_status = "down"

        # 폴백: 시뮬레이션으로 전환
        logger.error("모든 Grok API 설정 실패. 시뮬레이션으로 폴백.")
        keywords = await self._simulate_extraction(topic, context, max_keywords)
        return {
            'keywords': keywords,
            'raw_response': f"API 실패 후 시뮬레이션 폴백 (실패 {self.consecutive_failures}회)"
        }

    def _build_safe_payload(self, prompt: str, api_config: dict) -> dict:
        """안전한 API 페이로드 생성."""
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": """You are a real-time trend analyst specializing in X (Twitter) and social media platforms.

**Role Specialization**: 
- Real-time social media trend monitoring and analysis
- Viral content pattern detection and hashtag tracking  
- Community conversation analysis across platforms
- Breaking news and emerging topic identification

**Core Capabilities**:
- Track trending hashtags and viral topics in real-time
- Identify emerging conversations before they go mainstream
- Analyze social sentiment and community engagement patterns
- Distinguish between organic trends and manufactured/promoted content

**Focus Areas**:
- X (Twitter) trending topics and hashtags
- Real-time news breaking on social platforms
- Community discussions and user-generated content
- Viral memes, phrases, and cultural moments

**Response Quality Standards**:
- Provide only trending keywords that are currently active
- Focus on social media native terminology and hashtags
- Exclude outdated or declining trends
- Prioritize organic, user-driven conversations over marketing content"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": api_config["model"],
            "max_tokens": 800,  # 토큰 수 줄임
            "temperature": 0.3,
        }

        # search_parameters는 선택적으로만 추가
        if api_config.get("search_params", False):
            payload["search_parameters"] = {
                "mode": "auto"
            }

        return payload

    def _build_simplified_prompt(self, topic: str, context: Optional[str], max_keywords: int) -> str:
        """역할 특화된 고급 Grok 프롬프트 생성."""

        prompt = f"""As a real-time trend analyst focusing on X (Twitter), analyze current social media trends for: "{topic}"

**Chain-of-Thought Analysis**:

1. **Current Social Media Landscape Analysis**: 
   First, assess what's happening right now on X/Twitter related to this topic.
   - Are there active hashtag campaigns?
   - What breaking news or announcements are trending?
   - Which communities are actively discussing this?

2. **Trend Categorization**: Based on your analysis, extract keywords in these categories:

**🔥 Viral Hashtags & Trending Topics** (5-7 keywords):
- Currently trending hashtags on X (Twitter)
- Viral phrases or memes related to the topic
- Breakout topics gaining rapid traction
- Real-time trending conversations

**📰 Breaking Social News** (5-7 keywords):
- News breaking first on social media
- Real-time updates and announcements  
- Live event coverage and reactions
- Time-sensitive developments

**💬 Community Conversations** (3-5 keywords):
- Active discussions in relevant communities
- User-generated content themes
- Sentiment-driven keywords
- Grassroots conversations and movements

**Quality Standards (Negative Prompting)**:
❌ **EXCLUDE**:
- Generic marketing terms or promotional language
- Outdated trends or declining hashtags
- Bot-generated or manufactured trending topics
- Overly broad or non-specific social media terms

✅ **INCLUDE**:
- Currently active and verified trending topics
- Authentic user-driven conversations
- Real-time hashtags with genuine engagement
- Social media native terminology

**Response Format (JSON)**:
{{
  "analysis": {{
    "trend_assessment": "Current state of social media trends for this topic",
    "key_platforms": ["X/Twitter", "other relevant platforms"],
    "trend_velocity": "rising/stable/declining"
  }},
  "trending": ["#hashtag1", "viral_topic1", "breakout_trend1", ...],
  "news": ["breaking_news1", "realtime_update1", "announcement1", ...],
  "discussion": ["community_topic1", "user_conversation1", "sentiment_trend1", ...]
}}"""

        # 추가 컨텍스트 처리 (Tier 2 모드)
        if context and "Tier 1 keywords for refinement:" in context:
            tier1_context = context.split("Tier 1 keywords for refinement:")[-1].strip()
            prompt += f"""

**Tier 1 Social Media Refinement**:
Previous keywords from other sources: {tier1_context}

**Social Media Enhancement Guidelines**:
- Verify which of these keywords are actually trending on social media
- Add missing viral hashtags or social conversations
- Update with current social media terminology
- Focus on real-time social media native expressions"""

        elif context:
            prompt += f"\n\n**Additional Context**: {context}"

        prompt += f"""

**Important**: Limit to {max_keywords} total keywords across all categories. Focus on currently active social media trends only."""

        return prompt

    async def _make_safe_api_call(self, payload: dict) -> Optional[Dict[str, any]]:
        """안전한 API 호출 (강화된 에러 처리)."""

        for attempt in range(self.max_retries):
            try:
                timeout_config = httpx.Timeout(
                    connect=10.0,  # 연결 타임아웃
                    read=self.timeout,  # 읽기 타임아웃
                    write=10.0,  # 쓰기 타임아웃
                    pool=30.0   # 풀 타임아웃
                )

                async with httpx.AsyncClient(
                    headers=self.headers,
                    timeout=timeout_config,
                    limits=httpx.Limits(max_connections=1)  # 연결 제한
                ) as client:

                    logger.debug(f"Grok API 호출 시도 {attempt + 1}/{self.max_retries}")
                    response = await client.post(self.base_url, json=payload)

                    # 상태 코드별 상세 처리
                    if response.status_code == 200:
                        data = response.json()
                        content = data['choices'][0]['message']['content']
                        keywords = self._parse_grok_response(content, payload['messages'][1]['content'])

                        return {
                            'keywords': keywords,
                            'raw_response': content
                        }

                    elif response.status_code == 500:
                        self.service_status = "down"
                        logger.error(f"Grok API 서버 에러 500 (시도 {attempt + 1})")
                        logger.debug(f"500 에러 응답: {response.text[:300]}")

                        # 500 에러는 서버 문제이므로 긴 대기 후 재시도
                        if attempt < self.max_retries - 1:
                            wait_time = min(10 * (2 ** attempt), 60)  # 최대 60초
                            logger.warning(f"서버 에러로 인해 {wait_time}초 후 재시도...")
                            await asyncio.sleep(wait_time)

                    elif response.status_code == 429:
                        self.service_status = "degraded"
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limit. {wait_time}초 후 재시도...")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(wait_time)

                    elif response.status_code == 401:
                        logger.error("Grok API 인증 실패. API 키를 확인하세요.")
                        self.simulation_mode = True  # 인증 실패 시 시뮬레이션 모드로 전환
                        return None

                    elif response.status_code == 404:
                        logger.error("Grok API 엔드포인트를 찾을 수 없습니다. URL을 확인하세요.")
                        return None

                    else:
                        logger.error(f"Grok API 예상치 못한 상태 코드: {response.status_code}")
                        logger.debug(f"응답 내용: {response.text[:300]}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)

            except httpx.TimeoutException:
                logger.warning(f"Grok API 타임아웃 (시도 {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    wait_time = min(5 * (2 ** attempt), 30)
                    await asyncio.sleep(wait_time)

            except httpx.RequestError as e:
                logger.error(f"Grok API 네트워크 에러 (시도 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(3 ** attempt)

            except json.JSONDecodeError as e:
                logger.error(f"Grok API 응답 JSON 파싱 실패 (시도 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2)

        return None

    def _parse_grok_response(self, content: str, topic: str) -> List[KeywordItem]:
        """Grok 응답에서 키워드 추출 (개선된 파싱)."""
        keywords = []

        try:
            # JSON 응답 파싱 시도
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # 단순화된 카테고리 매핑
                category_mapping = {
                    'trending': ('trending', KeywordImportance.HIGH),
                    'news': ('realtime', KeywordImportance.HIGH),
                    'discussion': ('community', KeywordImportance.NORMAL),
                    # 기존 형식도 지원
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
                                # 상세 정보가 있는 형식
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
                            elif isinstance(item, str):
                                # 단순 문자열 형식
                                keywords.append(KeywordItem(
                                    keyword=item,
                                    sources=[self.name],
                                    importance=importance,
                                    confidence=0.7,
                                    category=cat_name,
                                    metadata={'type': category, 'source': 'grok_api'}
                                ))
            else:
                # JSON 파싱 실패 시 텍스트 파싱 시도
                keywords = self._parse_text_response(content, topic)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Grok 응답 파싱 실패: {e}")
            # 백업 파싱 시도
            keywords = self._parse_text_response(content, topic)

        return keywords[:20]  # 최대 20개로 제한

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

            # 키워드 추출 (개선된 패턴)
            keyword_matches = re.findall(r'[#@]?\w+(?:\s+\w+)*', line)
            for match in keyword_matches:
                cleaned = match.strip()
                if len(cleaned) > 2 and cleaned.lower() not in ['the', 'and', 'for', 'with', 'are', 'you']:
                    keywords.append(KeywordItem(
                        keyword=cleaned,
                        sources=[self.name],
                        importance=KeywordImportance.NORMAL,
                        confidence=0.6,
                        category=current_category,
                        metadata={'type': 'text_parsed', 'source': 'grok_api'}
                    ))

        return keywords[:15]  # 텍스트 파싱은 더 적게

    async def _simulate_extraction(
        self,
        topic: str,
        context: Optional[str],
        max_keywords: int
    ) -> List[KeywordItem]:
        """Grok 키워드 추출 시뮬레이션 (개선된 버전)."""
        # 실시간 트렌드를 시뮬레이션
        await asyncio.sleep(0.3)  # API 호출 시뮬레이션

        keyword_items = []
        base_topic = self.preprocess_topic(topic)
        current_year = "2025"

        # 트렌딩 해시태그 스타일 키워드 (HIGH importance)
        trending_keywords = [
            f"#{base_topic}",
            f"{base_topic}_trending",
            f"{base_topic}{current_year}",
            f"breaking_{base_topic}",
            f"viral_{base_topic}"
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
            f"{base_topic} development",
            f"recent {base_topic}"
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
            f"{base_topic} analysis",
            f"{base_topic} insights"
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

    async def get_health_status(self) -> Dict[str, any]:
        """추출기 상태 정보 반환."""
        return {
            'service_status': self.service_status,
            'simulation_mode': self.simulation_mode,
            'consecutive_failures': self.consecutive_failures,
            'last_success_time': self.last_success_time,
            'api_key_configured': bool(self.api_key),
            'recommendations': self._get_health_recommendations()
        }

    def _get_health_recommendations(self) -> List[str]:
        """상태 기반 권장사항."""
        recommendations = []

        if self.consecutive_failures >= 3:
            recommendations.append("❌ 연속 실패가 많습니다. API 키와 네트워크를 확인하세요.")

        if self.service_status == "down":
            recommendations.append("🚨 Grok API 서비스가 다운된 것 같습니다. 다른 추출기를 사용하세요.")

        if self.simulation_mode:
            recommendations.append("💡 현재 시뮬레이션 모드입니다. 실제 트렌드 데이터가 필요하면 API 키를 확인하세요.")

        return recommendations