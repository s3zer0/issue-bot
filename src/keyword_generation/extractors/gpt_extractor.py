"""
GPT 기반 키워드 추출기 (기존 코드 리팩토링).
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
    """OpenAI GPT를 사용한 키워드 추출기."""

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

        self.is_initialized = True
        logger.info(f"GPT 키워드 추출기 초기화 완료 (모델: {self.model})")

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

            # API 호출
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
                metadata={'model': self.model}
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

**키워드 카테고리**:
1. **핵심 키워드 (Primary)**: 주제의 본질을 나타내는 가장 중요한 용어 (5-7개)
2. **관련 용어 (Related)**: 구체적인 제품명, 기술명, 회사명 등 (5-7개)
3. **맥락 키워드 (Context)**: 산업, 트렌드, 응용 분야 등 (5-7개)
# ==========================================================
# ✨ [추가] 신뢰 도메인 요청
# ==========================================================
4. **신뢰 출처 (Trusted Domains)**: 이 주제에 대해 가장 권위있는 공식 웹사이트 도메인 (3-5개)
# ==========================================================

**응답 형식 (JSON)**:
{{
    "primary_keywords": ["키워드1", "키워드2", ...],
    "related_terms": ["용어1", "용어2", ...],
    "context_keywords": ["맥락1", "맥락2", ...],
    "trusted_domains": ["official-site.com", "trusted-source.org", ...],
    "confidence": 0.0-1.0
}}

주의: 실재하는 검증 가능한 용어와 도메인만 생성하세요."""

        if context:
            base_prompt += f"\n\n**추가 맥락**: {context}"

        return base_prompt

    async def _call_gpt(self, prompt: str) -> str:
        """GPT API 호출."""
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

    def _parse_response(self, raw_response: str) -> dict:
        """GPT 응답 파싱."""
        try:
            # JSON 추출
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
            if not json_match:
                raise ValueError("응답에서 JSON을 찾을 수 없습니다")
            data = json.loads(json_match.group())
            data.setdefault("trusted_domains", [])
            return data

        except Exception as e:
            logger.error(f"GPT 응답 파싱 실패: {e}")
            # 폴백: 기본 구조 반환
            return {
                "primary_keywords": [self.preprocess_topic(raw_response.split()[0])],
                "related_terms": [],
                "context_keywords": [],
                "trusted_domains": [],  # 폴백 시 빈 리스트
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
