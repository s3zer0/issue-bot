# src/clients/perplexity_client.py
"""
Perplexity AI API 클라이언트 모듈 - 개선된 프롬프트 버전
"""

import asyncio
import httpx
from typing import Dict, Any, List, Optional
from loguru import logger
from src.config import config


class PerplexityClient:
    """Perplexity API 통신을 전담하는 클라이언트"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.get_perplexity_api_key()
        if not self.api_key:
            raise ValueError("Perplexity API 키가 설정되지 않았습니다")

        self.base_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-large-128k-online"
        self.timeout = 100
        self.max_retries = 3
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"PerplexityClient 초기화 완료 (모델: {self.model})")

    async def _make_api_call(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """API 호출 공통 로직"""
        if system_prompt is None:
            system_prompt = "You are a precise and objective information analysis expert specializing in finding and analyzing the latest technical information, research papers, and industry news."

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 10000,
            "temperature": 0.3
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(self.base_url, headers=self.headers, json=payload)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as e:
                    logger.error(f"API HTTP Error (Status: {e.response.status_code}): {e.response.text}")
                    if e.response.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
                except httpx.RequestError as e:
                    logger.error(f"API Request Error (Attempt {attempt + 1}): {e}")
                    if attempt == self.max_retries - 1:
                        raise
        raise ValueError("API call failed after multiple retries.")

    async def search_issues(self, keywords: List[str], time_period: str, max_results: int) -> Dict[str, Any]:
        """이슈 목록 검색을 위한 상세한 프롬프트 생성 및 API 호출"""

        # 시스템 프롬프트를 전문 검색에 특화
        system_prompt = """You are an expert information analyst specializing in finding cutting-edge technical information, breaking news, and industry developments. 
        You have access to the latest web content and prioritize accuracy, technical depth, and relevance.
        Always base your responses on actual content from specific articles, papers, or documents - never summarize from website homepages or article listings."""

        prompt = f"""Search for and analyze up to {max_results} individual articles, technical documents, research papers, or news items related to the keywords: '{", ".join(keywords)}' published during '{time_period}'.

**CRITICAL REQUIREMENTS:**

1. **Source Quality**:
   - Prioritize primary sources: official announcements, technical blogs, research papers, security advisories
   - Include sources from: major tech companies, research institutions, security vendors, open-source projects
   - Avoid: aggregator sites, forums (unless highly technical), marketing materials

2. **Content Depth**:
   - Each item must be based on the actual content of a specific article/document
   - Include technical specifications, version numbers, dates, and quantitative data
   - Capture unique insights, findings, or announcements from each source

3. **Information Extraction**:
   - Technical details: specifications, architectures, algorithms, vulnerabilities
   - Concrete facts: numbers, dates, names, versions, CVE IDs
   - Impact analysis: who is affected, scale of impact, criticality
   - Novel aspects: what's new, what changed, why it matters

**OUTPUT FORMAT (strictly follow for each item):**

## **[Specific, descriptive title of the article/finding]**
**요약**: [Detailed summary including key technical points, findings, and implications - minimum 3-4 sentences]
**기술적 핵심**: [Key technical details, specifications, or metrics]
**출처**: [Exact source name and/or URL]
**발행일**: [YYYY-MM-DD format]
**카테고리**: [Research/Security/News/Technical/Industry/Policy]
**중요도**: [Critical/High/Medium/Low - with brief justification]
**관련 키워드**: [Additional relevant terms found in the article]

**QUALITY CHECKS:**
- Is this from an actual article, not a website listing?
- Does it contain specific, verifiable information?
- Would a technical expert find this valuable?
- Is it relevant to the search timeframe?"""

        return await self._make_api_call(prompt, system_prompt)

    async def collect_detailed_information(self, issue_title: str) -> Dict[str, Any]:
        """상세 정보 수집을 위한 심층적인 프롬프트 생성 및 API 호출"""

        system_prompt = """You are a senior technical analyst providing in-depth analysis of technology issues, security vulnerabilities, and industry developments.
        Your analysis should be suitable for technical professionals and decision-makers who need comprehensive understanding of complex topics.
        Always provide accurate, verifiable information with proper technical context."""

        prompt = f"""Provide a comprehensive, expert-level analysis in Korean for: **{issue_title}**

**ANALYSIS FRAMEWORK:**

### 1. 핵심 기술 분석 (Core Technical Analysis)
- **작동 원리**: 기술적 메커니즘과 아키텍처 상세 설명
- **구현 세부사항**: 코드 레벨, API, 프로토콜 등 구체적 정보
- **기술 사양**: 버전, 의존성, 시스템 요구사항
- **성능 지표**: 벤치마크, 처리량, 지연시간 등 정량적 데이터

### 2. 배경 및 맥락 (Background Context)
- **역사적 발전**: 이 기술/이슈가 등장하게 된 배경과 진화 과정
- **문제 정의**: 해결하려는 근본적인 문제나 니즈
- **선행 기술**: 관련된 이전 기술들과의 관계
- **산업 동향**: 현재 시장 상황과 기술 트렌드

### 3. 심층 영향 분석 (Deep Impact Analysis)
- **기술적 영향**: 
  - 어떤 시스템/플랫폼이 영향을 받는가?
  - 기존 인프라에 미치는 변화
  - 보안 implications (취약점, 위협, 방어)
- **비즈니스 영향**:
  - 비용 절감 또는 추가 비용
  - 생산성 및 효율성 변화
  - 경쟁 우위 요소
- **사용자 영향**:
  - 최종 사용자 경험 변화
  - 학습 곡선 및 적응 필요성

### 4. 실무 적용 가이드 (Practical Implementation)
- **도입 전략**: 단계별 구현 방법
- **모범 사례**: 업계 best practices
- **주의사항**: 흔한 실수와 함정
- **리소스**: 도구, 라이브러리, 문서 링크

### 5. 전문가 관점 (Expert Perspectives)
- **기술 커뮤니티 반응**: 주요 의견과 논쟁점
- **장단점 분석**: 객관적인 강점과 약점
- **대안 기술**: 경쟁 또는 보완 솔루션
- **미래 전망**: 발전 방향과 잠재력

### 6. 추가 정보 (Additional Intelligence)
- **관련 표준/규격**: ISO, IEEE, RFC 등
- **법적/규제 고려사항**: 컴플라이언스, 라이선스
- **참고 자료**: 논문, 공식 문서, 깃허브 저장소

**작성 지침:**
- 기술 용어는 정확히 사용하되, 필요시 한글 설명 병기
- 추측보다는 검증된 사실과 데이터 중심
- 구체적인 예시와 사례 포함
- 실무자가 즉시 활용 가능한 정보 제공
- 비판적 시각과 균형잡힌 분석 유지"""

        logger.info(f"Requesting detailed analysis for: {issue_title[:50]}...")
        return await self._make_api_call(prompt, system_prompt)