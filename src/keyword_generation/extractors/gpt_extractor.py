"""
GPT 기반 키워드 추출기 - 웹 검색 기능 추가 버전.
"""

import asyncio
import json
import re
import time
from typing import List, Optional
from loguru import logger
from openai import AsyncOpenAI

from src.config import config
from ..base import BaseKeywordExtractor, KeywordExtractionResult, KeywordItem, KeywordImportance


class GPTKeywordExtractor(BaseKeywordExtractor):
    """OpenAI GPT를 사용한 키워드 추출기 - 웹 검색 기능 포함."""

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

        self.is_initialized = True
        logger.info(f"GPT 키워드 추출기 초기화 완료 (모델: {self.model}, 웹 검색: {'활성화' if self.perplexity_client else '비활성화'})")

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

            # 웹 검색 기능을 포함한 API 호출
            if self.perplexity_client:
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
                    'web_search_available': self.perplexity_client is not None
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
        """GPT 프롬프트 생성."""
        base_prompt = f"""주제 "{topic}"에 대한 전문적이고 포괄적인 키워드를 생성해주세요.

**목적**: 이 키워드들은 최신 뉴스, 기술 문서, 연구 논문에서 관련 정보를 찾는 데 사용됩니다.

**키워드 생성 지침**:
- 최신 기술, 제품, 트렌드에 대해서는 2024-2025년 현재 정보를 반영하세요
- 검증 가능한 실제 용어만 생성하세요
- 추측이나 창작은 금지됩니다

**키워드 카테고리**:
1. **핵심 키워드 (Primary)**: 주제의 본질을 나타내는 가장 중요한 용어 (5-7개)
2. **관련 용어 (Related)**: 구체적인 제품명, 기술명, 회사명 등 (5-7개)
3. **맥락 키워드 (Context)**: 산업, 트렌드, 응용 분야 등 (5-7개)

**응답 형식 (JSON)**:
{{
    "primary_keywords": ["키워드1", "키워드2", ...],
    "related_terms": ["용어1", "용어2", ...],
    "context_keywords": ["맥락1", "맥락2", ...],
    "confidence": 0.0-1.0
}}"""

        if context:
            base_prompt += f"\n\n**추가 맥락**: {context}"

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
                            "content": "당신은 특정 주제에 대한 전문적인 키워드를 생성하는 전문가입니다. 항상 유효한 JSON 형식으로 응답하세요."
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
                            "content": """당신은 키워드 생성 전문가입니다. 
최신 정보가 필요한 경우 web_search 함수를 사용하여 정확한 정보를 확인하세요.
최종적으로는 반드시 유효한 JSON 형식으로 키워드를 응답하세요."""
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
                            "content": """당신은 키워드 생성 전문가입니다. 
검색 결과를 참고하여 정확하고 최신의 키워드를 JSON 형식으로 생성하세요."""
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