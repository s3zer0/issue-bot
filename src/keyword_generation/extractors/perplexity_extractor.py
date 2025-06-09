"""
Perplexity 기반 키워드 추출기.
"""

import asyncio
import time
import re
from typing import List, Optional, Dict
from loguru import logger

from src.clients.perplexity_client import PerplexityClient
from ..base import BaseKeywordExtractor, KeywordExtractionResult, KeywordItem, KeywordImportance


class PerplexityKeywordExtractor(BaseKeywordExtractor):
    """Perplexity API를 사용한 웹 기반 키워드 추출기."""

    def __init__(self, api_key: Optional[str] = None):
        """Perplexity 추출기 초기화."""
        super().__init__("Perplexity", api_key)

        # 기존 PerplexityClient 재사용
        self.client = PerplexityClient(api_key)
        self.is_initialized = True
        logger.info("Perplexity 키워드 추출기 초기화 완료")

    async def extract_keywords(
        self,
        topic: str,
        context: Optional[str] = None,
        max_keywords: int = 20
    ) -> KeywordExtractionResult:
        """Perplexity를 사용하여 웹 기반 키워드를 추출합니다."""
        start_time = time.time()
        logger.info(f"Perplexity 키워드 추출 시작: '{topic}'")

        try:
            # 웹 검색 기반 키워드 추출 프롬프트
            prompt = self._build_prompt(topic, context, max_keywords)

            # Perplexity API 호출
            response = await self.client._make_api_call(prompt)
            content = response['choices'][0]['message']['content']

            # 키워드 추출
            keywords = self._extract_keywords_from_content(content, topic)

            return KeywordExtractionResult(
                keywords=keywords,
                source_name=self.name,
                extraction_time=time.time() - start_time,
                raw_response=content,
                metadata={'model': self.client.model}
            )

        except Exception as e:
            logger.error(f"Perplexity 키워드 추출 실패: {e}")
            return KeywordExtractionResult(
                keywords=[],
                source_name=self.name,
                extraction_time=time.time() - start_time,
                error=str(e)
            )

    def _build_prompt(self, topic: str, context: Optional[str], max_keywords: int) -> str:
        """Perplexity용 프롬프트 생성."""
        prompt = f"""주제 "{topic}"와 관련하여 현재 웹에서 가장 많이 언급되고 있는 키워드를 찾아주세요.

다음 카테고리로 분류하여 키워드를 추출해주세요:

1. **핵심 키워드**: 최신 뉴스와 기사에서 자주 언급되는 주요 용어
2. **트렌딩 키워드**: 소셜미디어나 포럼에서 화제가 되는 용어
3. **전문 용어**: 학술 논문이나 기술 문서에서 사용되는 전문 용어

각 카테고리별로 5-7개씩, 총 {max_keywords}개 이내로 추출해주세요.

출력 형식:
### 핵심 키워드
- 키워드1
- 키워드2
...

### 트렌딩 키워드
- 키워드1
- 키워드2
...

### 전문 용어
- 용어1
- 용어2
..."""

        if context:
            prompt += f"\n\n추가 맥락: {context}"

        return prompt

    def _extract_keywords_from_content(self, content: str, topic: str) -> List[KeywordItem]:
        """Perplexity 응답에서 키워드 추출."""
        keyword_items = []

        # 섹션별로 키워드 추출
        sections = {
            '핵심 키워드': ('primary', KeywordImportance.HIGH, 0.9),
            '트렌딩 키워드': ('trending', KeywordImportance.HIGH, 0.85),
            '전문 용어': ('technical', KeywordImportance.NORMAL, 0.8)
        }

        for section_name, (category, importance, confidence) in sections.items():
            # 섹션 찾기
            section_pattern = rf'#*\s*{section_name}.*?(?=#{2,}|\Z)'
            section_match = re.search(section_pattern, content, re.DOTALL | re.IGNORECASE)

            if section_match:
                section_text = section_match.group()

                # 키워드 추출 (- 또는 * 로 시작하는 항목)
                keyword_pattern = r'[-*]\s*(.+?)(?:\n|$)'
                keywords = re.findall(keyword_pattern, section_text)

                for kw in keywords:
                    kw = kw.strip()
                    if kw and len(kw) > 1:
                        keyword_items.append(KeywordItem(
                            keyword=kw,
                            sources=[self.name],
                            importance=importance,
                            confidence=confidence,
                            category=category,
                            metadata={'source_type': 'web_search'}
                        ))

        # 키워드를 찾지 못한 경우 폴백
        if not keyword_items:
            # 간단한 휴리스틱으로 키워드 추출
            words = re.findall(r'\b\w+\b', content)
            word_freq = {}
            for word in words:
                if len(word) > 3 and word.lower() != topic.lower():
                    word_freq[word.lower()] = word_freq.get(word.lower(), 0) + 1

            # 빈도 상위 키워드 추출
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            for word, freq in top_words:
                keyword_items.append(KeywordItem(
                    keyword=word,
                    sources=[self.name],
                    importance=KeywordImportance.NORMAL if freq > 2 else KeywordImportance.LOW,
                    confidence=0.6,
                    category='extracted',
                    metadata={'frequency': freq}
                ))

        return keyword_items