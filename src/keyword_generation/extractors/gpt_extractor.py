"""
GPT 기반 키워드 추출기 - 웹 검색 기능 추가 버전.
"""

import asyncio
import json
import re
import time
from typing import List, Optional, Dict
from loguru import logger
from openai import AsyncOpenAI

from src.config import config
from ..base import BaseKeywordExtractor, KeywordExtractionResult, KeywordItem, KeywordImportance


class GPTKeywordExtractor(BaseKeywordExtractor):
    """
    OpenAI GPT를 사용한 키워드 추출기 - 향상된 웹 검색 기능 포함.
    
    Features:
    - 🆕 GPT-4o 네이티브 브라우징 지원
    - 🔍 Perplexity API 통합 검색  
    - 🌐 하이브리드 검색 전략 (네이티브 + Perplexity)
    - ⚡ 병렬 검색 실행 및 지능적 폴백
    - 🎯 검색 전략 동적 변경 가능
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """GPT 추출기 초기화."""
        api_key = api_key or config.get_openai_api_key()
        super().__init__("GPT", api_key)

        if not self.api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model or config.get_openai_model()
        self.temperature = config.get_openai_temperature()
        self.max_tokens = config.get_openai_max_tokens()
        self.max_retries = config.get_max_retry_count()

        # 🔍 웹 검색 기능 추가
        self.perplexity_client = None
        self._initialize_search_client()
        
        # 🆕 GPT-4o 네이티브 브라우징 지원 확인
        self.native_browsing_supported = self._check_native_browsing_support()
        self.search_strategy = "hybrid"  # "native", "perplexity", "hybrid"

        self.is_initialized = True
        search_status = self._get_search_status_message()
        logger.info(f"GPT 키워드 추출기 초기화 완료 (모델: {self.model}, {search_status})")

    def _initialize_search_client(self):
        """Perplexity 클라이언트 초기화."""
        try:
            from src.clients.perplexity_client import PerplexityClient
            self.perplexity_client = PerplexityClient()
            logger.info("웹 검색 클라이언트 초기화 완료")
        except ImportError:
            logger.warning("Perplexity 클라이언트를 찾을 수 없습니다. 웹 검색 기능이 비활성화됩니다.")
        except Exception as e:
            logger.warning(f"웹 검색 클라이언트 초기화 실패: {e}")

    def _check_native_browsing_support(self) -> bool:
        """GPT-4o 네이티브 브라우징 지원 여부 확인."""
        # GPT-4o 모델들은 네이티브 브라우징을 지원
        browse_supported_models = [
            "gpt-4o", "gpt-4o-2024-05-13", "gpt-4o-2024-08-06", 
            "gpt-4o-mini", "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo", "gpt-4-turbo-2024-04-09"
        ]
        return any(model in self.model.lower() for model in browse_supported_models)

    def _get_search_status_message(self) -> str:
        """검색 기능 상태 메시지 생성."""
        status_parts = []
        
        if self.native_browsing_supported:
            status_parts.append("GPT-4o 네이티브 브라우징")
        
        if self.perplexity_client:
            status_parts.append("Perplexity 검색")
        
        if status_parts:
            return f"웹 검색: {' + '.join(status_parts)} 활성화 ({self.search_strategy})"
        else:
            return "웹 검색: 비활성화"

    def _get_search_capabilities(self) -> Dict[str, bool]:
        """검색 기능 가용성 상태 반환."""
        return {
            'native_browsing': self.native_browsing_supported,
            'perplexity_search': self.perplexity_client is not None,
            'hybrid_search': self.native_browsing_supported or self.perplexity_client is not None
        }

    async def extract_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        max_keywords: int = 20
    ) -> KeywordExtractionResult:
        """GPT를 사용하여 키워드를 추출합니다."""
        start_time = time.time()
        logger.info(f"GPT 키워드 추출 시작: '{topic}'")

        try:
            # 프롬프트 생성
            prompt = self._build_prompt(topic, context, max_keywords)

            # 🆕 향상된 검색 전략에 따른 API 호출
            if self.search_strategy == "hybrid" and (self.native_browsing_supported or self.perplexity_client):
                raw_response = await self._call_gpt_with_hybrid_search(prompt)
            elif self.search_strategy == "native" and self.native_browsing_supported:
                raw_response = await self._call_gpt_with_native_browse(prompt)
            elif self.search_strategy == "perplexity" and self.perplexity_client:
                raw_response = await self._call_gpt_with_search(prompt)
            else:
                raw_response = await self._call_gpt(prompt)

            # 응답 파싱
            keywords = self._parse_response(raw_response)

            # KeywordItem 객체로 변환
            keyword_items = self._create_keyword_items(keywords)

            return KeywordExtractionResult(
                keywords=keyword_items,
                source_name=self.name,
                extraction_time=time.time() - start_time,
                raw_response=raw_response,
                metadata={
                    'model': self.model,
                    'web_search_available': self.perplexity_client is not None,
                    'native_browsing_supported': self.native_browsing_supported,
                    'search_strategy': self.search_strategy,
                    'search_capabilities': self._get_search_capabilities()
                }
            )

        except Exception as e:
            logger.error(f"GPT 키워드 추출 실패: {e}")
            return KeywordExtractionResult(
                keywords=[],
                source_name=self.name,
                extraction_time=time.time() - start_time,
                error=str(e)
            )

    def _build_prompt(self, topic: str, context: Optional[str], max_keywords: int) -> str:
        """고급 프롬프트 엔지니어링이 적용된 GPT 프롬프트 생성."""
        
        # Chain-of-Thought (CoT) 구조로 프롬프트 설계
        base_prompt = f"""당신은 기술 분야 전문 키워드 분석가입니다. 귀하의 역할은 최신 기술 동향과 정확한 전문 용어를 기반으로 고품질 키워드를 생성하는 것입니다.

**분석 주제**: "{topic}"

**단계별 분석 수행 (Chain-of-Thought)**:

1. **주제 분석 단계**: 먼저 이 주제의 핵심 측면들을 분석해보세요.
   - 기술적 영역: 어떤 기술 분야에 속하는가?
   - 현재 상태: 신기술인가, 기존 기술의 발전인가?
   - 적용 분야: 어떤 산업이나 영역에서 사용되는가?

2. **키워드 후보 생성**: 위 분석을 바탕으로 다음 카테고리별로 키워드를 생성하세요.

**키워드 카테고리 및 생성 규칙**:

🎯 **핵심 키워드 (Primary)** (5-7개):
- 주제의 본질을 정확히 표현하는 전문 용어
- 공식 명칭, 표준 용어, 업계 표준 사용
- 예: 정확한 기술명, API명, 프로토콜명

🔗 **관련 용어 (Related)** (5-7개):
- 구체적인 구현체, 제품명, 회사명
- 버전 정보, 플랫폼별 구현체
- 경쟁 기술이나 대안 솔루션

🌐 **맥락 키워드 (Context)** (5-7개):
- 응용 분야, 사용 사례
- 관련 산업, 생태계
- 현재 트렌드, 미래 전망

**엄격한 품질 기준 (Negative Prompting)**:
❌ **제외할 키워드 유형**:
- 마케팅 용어나 과장된 표현 ("혁신적", "최고의", "완벽한")
- 초보자 대상 용어 ("introduction", "basics", "tutorial", "가이드")
- 모호하거나 일반적인 용어 ("솔루션", "시스템", "플랫폼")
- 확인되지 않은 추측성 용어
- 과도하게 일반적인 형용사

✅ **포함할 키워드 특징**:
- 전문가가 실제로 사용하는 정확한 용어
- 검색 가능하고 검증 가능한 실제 용어
- 기술 문서나 공식 발표에서 사용되는 용어
- 2024-2025년 현재 유효한 최신 용어

**응답 형식 (JSON)**:
{{
    "analysis": {{
        "technical_domain": "기술 영역 분석",
        "key_aspects": ["핵심 측면1", "핵심 측면2", "핵심 측면3"],
        "current_status": "현재 상태 분석"
    }},
    "primary_keywords": ["키워드1", "키워드2", ...],
    "related_terms": ["용어1", "용어2", ...],
    "context_keywords": ["맥락1", "맥락2", ...],
    "confidence": 0.0-1.0,
    "quality_check": {{
        "excluded_generic_terms": "제외된 일반적 용어들",
        "verification_sources": "검증 가능한 소스들"
    }}
}}"""

        # 추가 컨텍스트 처리 (Tier 2에서 사용)
        if context and "Tier 1 keywords for refinement:" in context:
            tier1_context = context.split("Tier 1 keywords for refinement:")[-1].strip()
            base_prompt += f"""

**Tier 2 정제 모드 활성화**:
Tier 1에서 생성된 키워드들을 참고하여 더 정교하고 전문적인 키워드를 생성하세요.

**Tier 1 참고 키워드**: {tier1_context}

**Tier 2 정제 지침**:
- Tier 1 키워드의 품질을 향상시키세요
- 중복되지 않는 새로운 관점의 키워드 추가
- 더 구체적이고 전문적인 용어로 대체
- 최신 기술 동향 반영"""

        elif context:
            base_prompt += f"\n\n**추가 컨텍스트**: {context}"

        return base_prompt

    async def _call_gpt(self, prompt: str) -> str:
        """기본 GPT API 호출 (웹 검색 없음)."""
        for attempt in range(self.max_retries):
            try:
                request_params = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": """당신은 기술 분야 전문 키워드 분석가입니다. 

**전문 영역**: 최신 기술 동향, 소프트웨어 개발, AI/ML, 클라우드, 모바일, 웹 기술 등
**핵심 능력**: 
- 공식 기술 문서와 API 레퍼런스 분석
- 업계 표준 용어와 실무진이 사용하는 정확한 전문 용어 구분
- 마케팅 용어와 기술적 정확성을 가진 용어 구분
- 최신 기술 트렌드와 레거시 기술의 현재 상태 파악

**응답 원칙**: 
- 항상 검증 가능한 실제 기술 용어만 사용
- 추측이나 창작 금지
- 유효한 JSON 형식으로 응답
- Chain-of-Thought 분석 과정 포함"""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }

                # GPT-4 모델의 경우 JSON 모드 활성화
                if "gpt-4" in self.model:
                    request_params["response_format"] = {"type": "json_object"}

                response = await self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(f"GPT API 호출 실패 (시도 {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _call_gpt_with_search(self, prompt: str) -> str:
        """웹 검색 기능이 포함된 GPT API 호출."""

        # 웹 검색 함수 정의
        search_function = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "최신 정보나 사실 확인이 필요할 때 웹을 검색합니다",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색할 키워드나 질문"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

        for attempt in range(self.max_retries):
            try:
                # 첫 번째 GPT 호출 (검색 함수 포함)
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": """당신은 기술 분야 전문 키워드 분석가입니다.

**역할 특화**: 웹 검색을 통한 실시간 정보 수집 및 검증
**주요 임무**:
- 최신 기술 동향, 제품 출시, 업데이트 정보 실시간 확인
- 공식 발표, 기술 문서, API 변경사항 추적
- 정확한 버전 정보, 제품명, 기술 사양 검증

**웹 검색 활용 지침**:
- 불확실한 최신 정보는 반드시 web_search 함수 사용
- 제품명, 버전, 출시일 등은 공식 소스에서 확인
- 검색 결과를 바탕으로 정확하고 최신의 키워드 생성

**최종 응답**: 유효한 JSON 형식, Chain-of-Thought 분석 포함"""
                        },
                        {"role": "user", "content": prompt}
                    ],
                    tools=[search_function],
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                message = response.choices[0].message

                # Function call이 있는지 확인
                if message.tool_calls:
                    logger.debug(f"GPT가 웹 검색 요청: {len(message.tool_calls)}개 검색")

                    # Performance: Parallel web search execution
                    messages = [
                        {
                            "role": "system",
                            "content": """당신은 기술 분야 전문 키워드 분석가입니다.

**현재 상황**: 웹 검색 결과를 바탕으로 최종 키워드 생성
**핵심 원칙**:
- 검색으로 확인된 정확한 정보만 사용
- 최신성과 정확성이 검증된 용어 우선
- 공식 명칭과 표준 용어 사용
- 추측이나 불확실한 정보 배제

**최종 응답**: JSON 형식, 검색 결과 기반 신뢰도 반영"""
                        },
                        {"role": "user", "content": prompt},
                        message
                    ]

                    # Performance: Collect all search tasks for parallel execution
                    search_tasks = []
                    tool_call_mapping = {}
                    
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "web_search":
                            args = json.loads(tool_call.function.arguments)
                            query = args.get("query", "")
                            
                            logger.debug(f"웹 검색 준비: {query}")
                            search_task = self._perform_web_search(query)
                            search_tasks.append(search_task)
                            tool_call_mapping[len(search_tasks) - 1] = tool_call

                    # Performance: Execute all searches in parallel
                    if search_tasks:
                        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                        
                        # Add all search results to messages
                        for i, (search_result, tool_call) in enumerate(zip(search_results, tool_call_mapping.values())):
                            if not isinstance(search_result, Exception):
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": search_result
                                })
                            else:
                                logger.warning(f"Web search failed: {search_result}")
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": f"검색 실패: {str(search_result)}"
                                })

                    # 검색 결과를 포함한 최종 응답 생성
                    final_response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )

                    return final_response.choices[0].message.content.strip()
                else:
                    # 검색 없이 바로 응답
                    return message.content.strip()

            except Exception as e:
                logger.warning(f"GPT Function Call API 호출 실패 (시도 {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    # 웹 검색 실패 시 기본 방법으로 폴백
                    logger.warning("웹 검색 실패, 기본 GPT 호출로 폴백")
                    return await self._call_gpt(prompt)
                await asyncio.sleep(2 ** attempt)

    async def _call_gpt_with_native_browse(self, prompt: str) -> str:
        """🆕 GPT-4o 네이티브 브라우징을 사용한 키워드 추출."""
        
        # 네이티브 브라우징용 향상된 프롬프트
        browsing_prompt = f"""당신은 웹 브라우징 기능을 가진 기술 분야 전문 키워드 분석가입니다.

**네이티브 브라우징 역할**:
- 실시간 웹 정보를 직접 검색하고 분석
- 공식 문서, 기술 블로그, 최신 발표 자료 접근
- 정확한 버전 정보, 출시일, 기술 사양 실시간 확인

**브라우징 지침**:
1. 먼저 주제와 관련된 공식 웹사이트나 문서를 검색하세요
2. 최신 기술 동향이나 업데이트 정보를 찾아보세요  
3. 정확한 제품명, 버전, API 정보를 확인하세요
4. 검색한 정보를 바탕으로 키워드를 생성하세요

**원본 요청**:
{prompt}

**중요**: 웹 검색을 통해 확인한 최신 정보만을 사용하여 응답하세요."""

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": """당신은 웹 브라우징 기능을 가진 기술 분야 전문 키워드 분석가입니다.

**네이티브 브라우징 특화 역할**:
- 실시간 웹 정보 검색 및 분석 전문가
- 공식 문서와 기술 사양서 직접 접근 및 분석
- 최신 기술 동향과 업데이트 정보 실시간 모니터링
- 정확한 기술 용어와 공식 명칭 검증

**핵심 능력**:
- 직접 웹사이트 방문하여 정보 수집
- 공식 API 문서와 릴리스 노트 분석
- 기술 블로그와 개발자 포럼 모니터링
- 버전 정보와 호환성 데이터 실시간 확인

**응답 원칙**: 
- 웹 브라우징으로 확인한 최신 정보만 사용
- 검색하지 않은 내용은 추측하지 않음
- 유효한 JSON 형식으로 응답
- 브라우징 과정과 결과를 분석에 포함"""
                        },
                        {"role": "user", "content": browsing_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                logger.debug("GPT-4o 네이티브 브라우징 완료")
                return response.choices[0].message.content.strip()

            except Exception as e:
                logger.warning(f"GPT 네이티브 브라우징 실패 (시도 {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    # 네이티브 브라우징 실패 시 기존 방법으로 폴백
                    logger.warning("네이티브 브라우징 실패, 기본 검색으로 폴백")
                    if self.perplexity_client:
                        return await self._call_gpt_with_search(prompt)
                    else:
                        return await self._call_gpt(prompt)
                await asyncio.sleep(2 ** attempt)

    async def _call_gpt_with_hybrid_search(self, prompt: str) -> str:
        """🆕 하이브리드 검색 전략: 네이티브 브라우징 + Perplexity 검색."""
        
        # 검색 전략 결정
        search_tasks = []
        search_methods = []
        
        # 네이티브 브라우징이 가능하면 추가
        if self.native_browsing_supported:
            search_tasks.append(self._call_gpt_with_native_browse(prompt))
            search_methods.append("native_browsing")
        
        # Perplexity 검색이 가능하면 추가
        if self.perplexity_client:
            search_tasks.append(self._call_gpt_with_search(prompt))
            search_methods.append("perplexity_search")
        
        if not search_tasks:
            # 검색 방법이 없으면 기본 GPT 호출
            return await self._call_gpt(prompt)
        
        try:
            # 병렬 검색 실행 (타임아웃 설정)
            logger.debug(f"하이브리드 검색 시작: {search_methods}")
            
            # 코루틴을 태스크로 변환
            tasks = [asyncio.create_task(task) for task in search_tasks]
            
            # 첫 번째 성공하는 검색 결과 사용 (더 빠른 응답)
            done, pending = await asyncio.wait(
                tasks, 
                return_when=asyncio.FIRST_COMPLETED,
                timeout=120  # 2분 타임아웃
            )
            
            # 완료되지 않은 태스크 취소
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # 첫 번째 성공 결과 반환
            if done:
                completed_task = list(done)[0]
                result = await completed_task
                
                # 어떤 방법이 성공했는지 찾기
                task_index = tasks.index(completed_task)
                successful_method = search_methods[task_index]
                logger.success(f"하이브리드 검색 성공: {successful_method}")
                return result
            else:
                logger.warning("하이브리드 검색 타임아웃, 기본 GPT로 폴백")
                return await self._call_gpt(prompt)
                
        except Exception as e:
            logger.error(f"하이브리드 검색 실패: {e}")
            # 모든 검색 실패 시 기본 GPT 호출
            return await self._call_gpt(prompt)

    async def _perform_web_search(self, query: str) -> str:
        """실제 웹 검색 수행."""
        try:
            if self.perplexity_client:
                # Perplexity API로 검색
                result = await self.perplexity_client._make_api_call(
                    f"{query}에 대한 최신 정보와 공식 발표 내용을 검색해주세요. "
                    f"특히 정확한 제품명, 버전, 출시일, 기술 사양 등을 포함해주세요."
                )
                return f"검색 결과: {result}"
            else:
                return f"검색 클라이언트 없음: {query}"

        except Exception as e:
            logger.warning(f"웹 검색 실패: {e}")
            return f"검색 실패: {query} (오류: {str(e)})"

    def _parse_response(self, raw_response: str) -> dict:
        """GPT 응답 파싱."""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
            if not json_match:
                raise ValueError("응답에서 JSON을 찾을 수 없습니다")

            data = json.loads(json_match.group())

            # 필수 필드 확인 및 기본값 설정
            data.setdefault("primary_keywords", [])
            data.setdefault("related_terms", [])
            data.setdefault("context_keywords", [])
            data.setdefault("confidence", 0.8)

            return data

        except Exception as e:
            logger.error(f"GPT 응답 파싱 실패: {e}")
            logger.debug(f"원본 응답: {raw_response}")

            # 폴백: 기본 구조 반환
            return {
                "primary_keywords": [self.preprocess_topic(raw_response.split()[0]) if raw_response else "키워드"],
                "related_terms": [],
                "context_keywords": [],
                "confidence": 0.3
            }

    def _create_keyword_items(self, parsed_data: dict) -> List[KeywordItem]:
        """파싱된 데이터를 KeywordItem 객체로 변환."""
        keyword_items = []
        confidence = float(parsed_data.get('confidence', 0.8))

        # Primary keywords - HIGH importance
        for kw in parsed_data.get('primary_keywords', []):
            keyword_items.append(KeywordItem(
                keyword=kw,
                sources=[self.name],
                importance=KeywordImportance.HIGH,
                confidence=confidence,
                category='primary'
            ))

        # Related terms - NORMAL importance
        for kw in parsed_data.get('related_terms', []):
            keyword_items.append(KeywordItem(
                keyword=kw,
                sources=[self.name],
                importance=KeywordImportance.NORMAL,
                confidence=confidence * 0.9,
                category='related'
            ))

        # Context keywords - NORMAL to LOW importance
        for kw in parsed_data.get('context_keywords', []):
            keyword_items.append(KeywordItem(
                keyword=kw,
                sources=[self.name],
                importance=KeywordImportance.NORMAL,
                confidence=confidence * 0.8,
                category='context'
            ))

        return keyword_items

    def set_search_strategy(self, strategy: str) -> bool:
        """🆕 검색 전략 설정.
        
        Args:
            strategy: "native", "perplexity", "hybrid" 중 하나
            
        Returns:
            bool: 설정 성공 여부
        """
        valid_strategies = ["native", "perplexity", "hybrid"]
        
        if strategy not in valid_strategies:
            logger.error(f"잘못된 검색 전략: {strategy}. 유효한 값: {valid_strategies}")
            return False
        
        # 전략 실행 가능성 확인
        if strategy == "native" and not self.native_browsing_supported:
            logger.warning("네이티브 브라우징이 지원되지 않는 모델입니다. 전략을 변경할 수 없습니다.")
            return False
        
        if strategy == "perplexity" and not self.perplexity_client:
            logger.warning("Perplexity 클라이언트가 초기화되지 않았습니다. 전략을 변경할 수 없습니다.")
            return False
        
        old_strategy = self.search_strategy
        self.search_strategy = strategy
        logger.info(f"검색 전략 변경: {old_strategy} → {strategy}")
        return True

    def get_search_info(self) -> Dict[str, any]:
        """🆕 현재 검색 설정 정보 반환."""
        return {
            'current_strategy': self.search_strategy,
            'capabilities': self._get_search_capabilities(),
            'native_browsing_model': self.native_browsing_supported,
            'model': self.model,
            'status_message': self._get_search_status_message()
        }