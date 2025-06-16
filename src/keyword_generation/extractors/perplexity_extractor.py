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
        """역할 특화된 Perplexity용 고급 프롬프트 생성."""
        
        prompt = f"""당신은 웹 정보 수집과 분석을 전문으로 하는 연구 사서(Research Librarian)입니다.

**전문 영역**: 
- 실시간 웹 정보 모니터링 및 분석
- 다양한 온라인 소스에서 신뢰할 수 있는 정보 식별
- 학술 자료, 공식 문서, 기술 블로그, 뉴스 등 다각적 정보 수집

**현재 임무**: "{topic}"에 대한 웹 전반의 키워드 동향 분석

**단계별 분석 수행**:

1. **정보 소스 분석**: 먼저 이 주제와 관련된 주요 정보 소스들을 파악해보세요.
   - 공식 웹사이트나 문서
   - 최신 뉴스 기사
   - 기술 블로그나 포럼 게시물
   - 학술 논문이나 연구 자료

2. **키워드 카테고리별 수집**:

**📰 웹 트렌드 키워드 (Primary)**:
- 최신 뉴스와 기사에서 현재 가장 자주 언급되는 용어
- 공식 발표나 보도자료에 등장하는 핵심 용어
- 웹 검색량이 급증하고 있는 용어
- (5-7개)

**💬 커뮤니티 활성 키워드 (Related)**:
- 개발자 커뮤니티, 포럼에서 활발히 논의되는 용어
- GitHub, Stack Overflow 등에서 언급되는 기술 용어
- 소셜미디어에서 화제가 되는 관련 용어
- (5-7개)

**📚 전문 문서 키워드 (Context)**:
- 공식 문서, API 레퍼런스에서 사용되는 정확한 용어
- 학술 논문이나 기술 백서의 전문 용어
- 업계 표준이나 사양서에 명시된 용어
- (5-7개)

**품질 기준 (Negative Prompting)**:
❌ **제외해야 할 키워드**:
- 과장된 마케팅 표현 ("혁신적", "차세대", "최고")
- 너무 일반적인 형용사 ("좋은", "나쁜", "큰", "작은")
- 초보자 대상 용어 ("입문", "기초", "튜토리얼")
- 확인되지 않은 루머나 추측성 정보

✅ **포함해야 할 키워드**:
- 실제 웹에서 검증 가능한 정확한 용어
- 공식적으로 발표되거나 문서화된 용어
- 전문가들이 실제로 사용하는 기술 용어
- 현재 활발히 논의되고 있는 실용적 용어

**출력 형식**:
### 웹 트렌드 키워드
- [실제 뉴스/공식 발표에서 확인된 용어]
- [웹 검색 트렌드 상위 용어]
...

### 커뮤니티 활성 키워드  
- [개발자 커뮤니티에서 논의되는 용어]
- [GitHub/Stack Overflow 인기 용어]
...

### 전문 문서 키워드
- [공식 문서/API에서 사용되는 정확한 용어]
- [학술 논문/기술 백서의 전문 용어]
..."""

        # 추가 컨텍스트 처리 (Tier 2 모드)
        if context and "Tier 1 keywords for refinement:" in context:
            tier1_context = context.split("Tier 1 keywords for refinement:")[-1].strip()
            prompt += f"""

**Tier 1 정제 모드**: 이전에 생성된 키워드들을 웹 정보로 보완하고 검증하세요.

**참고할 Tier 1 키워드**: {tier1_context}

**웹 정보 기반 개선 지침**:
- Tier 1 키워드의 정확성을 웹 정보로 검증
- 누락된 웹 트렌드 키워드 추가 발굴
- 최신 웹 동향을 반영한 키워드 업데이트
- 웹에서 실제 사용되는 표현으로 정제"""

        elif context:
            prompt += f"\n\n**추가 컨텍스트**: {context}"

        prompt += f"""

**중요**: 모든 키워드는 실제 웹에서 확인 가능한 것만 포함하고, 총 {max_keywords}개 이내로 제한하세요."""

        return prompt

    def _extract_keywords_from_content(self, content: str, topic: str) -> List[KeywordItem]:
        """Perplexity 응답에서 키워드 추출."""
        keyword_items = []

        # 섹션별로 키워드 추출 (새로운 프롬프트 형식에 맞춤)
        sections = {
            '웹 트렌드 키워드': ('web_trending', KeywordImportance.HIGH, 0.9),
            '커뮤니티 활성 키워드': ('community_active', KeywordImportance.HIGH, 0.85),
            '전문 문서 키워드': ('professional_docs', KeywordImportance.NORMAL, 0.8),
            # 기존 형식 호환성 유지
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