"""
프로젝트에서 사용하는 모든 데이터 모델(dataclasses)을 정의하는 중앙 모듈.

이 파일은 프로젝트 전반의 데이터 구조를 일관되게 관리하며, 타입 힌팅을 활용하여
코드의 안정성, 가독성, 유지보수성을 향상시킵니다. 모든 데이터 모델은 dataclasses를
사용하여 간결하고 명확한 정의를 제공합니다.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


# --- Keyword Generation Models ---
@dataclass
class KeywordResult:
    """키워드 생성 결과를 구조화된 형태로 저장하는 데이터 클래스.

    키워드 생성 모듈에서 생성된 결과를 저장하며, 주제와 관련된 다양한 키워드
    카테고리 및 메타데이터를 포함합니다.

    Attributes:
        topic (str): 키워드 생성의 기반이 되는 주제.
        primary_keywords (List[str]): 주제의 핵심을 포착하는 주요 키워드.
        related_terms (List[str]): 고유명사, 제품명, 최신 용어 등 관련 용어.
        context_keywords (List[str]): 주제의 산업, 정책, 사회적 맥락 관련 키워드.
        confidence_score (float): 생성된 키워드의 신뢰도 점수 (0.0 ~ 1.0).
        generation_time (float): 키워드 생성에 소요된 시간 (초).
        raw_response (str): LLM의 원본 응답 데이터.
    """
    topic: str
    primary_keywords: List[str]
    related_terms: List[str]
    context_keywords: List[str]
    confidence_score: float
    generation_time: float
    raw_response: str
    trusted_domains: List[str] = field(default_factory=list)


# --- Hallucination Detection Models ---
@dataclass
class RePPLScore:
    """RePPL 기반 환각 탐지 분석 결과를 저장하는 데이터 클래스.

    텍스트의 반복성, 퍼플렉시티, 의미적 엔트로피를 분석하여 LLM 출력의
    신뢰성을 평가합니다.

    Attributes:
        repetition_score (float): 텍스트 내 반복 패턴의 점수 (높을수록 반복적).
        perplexity (float): 텍스트의 퍼플렉시티 (낮을수록 예측 가능).
        semantic_entropy (float): 의미적 불확실성 점수 (높을수록 모호).
        confidence (float): 분석 결과의 신뢰도 점수 (0.0 ~ 1.0).
        repeated_phrases (List[str]): 탐지된 반복 구문 목록.
        analysis_details (Dict[str, Any]): 추가 분석 메타데이터.
    """
    repetition_score: float
    perplexity: float
    semantic_entropy: float
    confidence: float
    repeated_phrases: List[str]
    analysis_details: Dict[str, Any]

# --- Issue Search Models ---
@dataclass
class IssueItem:
    """개별 이슈의 정보를 구조화된 형태로 저장하는 데이터 클래스.

    이슈 검색 모듈에서 생성된 단일 이슈의 세부 정보를 저장하며, 필수 필드와
    선택적 상세 정보 필드를 포함합니다. 필드 순서는 필수 필드를 먼저,
    선택적 필드를 나중에 배치하여 가독성을 높였습니다.

    Attributes:
        title (str): 이슈의 제목.
        summary (str): 이슈의 요약 내용.
        source (str): 이슈의 출처 (예: 뉴스, 보고서 등).
        published_date (Optional[str]): 이슈 발행일 (형식 미정, None 가능).
        relevance_score (float): 이슈의 관련성 점수 (0.0 ~ 1.0).
        category (str): 이슈의 카테고리 (예: 'news', 'tech').
        content_snippet (str): 이슈 내용의 짧은 발췌 (일반적으로 요약의 일부).
        technical_core (Optional[str]): 이슈의 기술적 핵심 내용 (선택적).
        importance (Optional[str]): 이슈의 중요도 수준 (예: 'Critical', 'High').
        related_keywords (Optional[str]): 이슈와 관련된 키워드 문자열.
        technical_analysis (Optional[str]): 기술적 분석 상세 내용.
        detailed_content (Optional[str]): 이슈의 전체 상세 내용.
        background_context (Optional[str]): 이슈의 배경 및 맥락 설명.
        impact_analysis (Optional[Any]): 이슈의 영향 분석 내용 (형식 미정).
        detail_collection_time (Optional[float]): 상세 정보 수집 소요 시간 (초).
        detail_confidence (Optional[float]): 상세 정보의 신뢰도 점수 (0.0 ~ 1.0).

    Notes:
        - 환각 탐지 결과는 `setattr`을 통해 동적으로 추가되므로, 이 클래스 정의에는 포함되지 않음.
        - 필드 순서는 유지보수와 가독성을 위해 필수 필드를 먼저 배치.
    """
    # --- 필수 필드 (기본값 없음) ---
    title: str
    summary: str
    source: str
    published_date: Optional[str]
    relevance_score: float
    category: str
    content_snippet: str

    # --- 선택적 필드 (기본값 있음) ---
    technical_core: Optional[str] = None
    importance: Optional[str] = None
    related_keywords: Optional[str] = None
    technical_analysis: Optional[str] = None
    detailed_content: Optional[str] = None
    background_context: Optional[str] = None
    impact_analysis: Optional[Any] = None
    detail_collection_time: Optional[float] = None
    detail_confidence: Optional[float] = None


@dataclass
class SearchResult:
    """이슈 검색의 전체 결과를 구조화된 형태로 저장하는 데이터 클래스.

    이슈 검색 모듈에서 반환된 검색 결과와 메타데이터를 포함하며, 여러 이슈와
    관련 통계를 저장합니다.

    Attributes:
        query_keywords (List[str]): 검색에 사용된 키워드 리스트.
        total_found (int): 검색된 이슈의 총 개수.
        issues (List[IssueItem]): 검색된 이슈 객체 리스트.
        search_time (float): 전체 검색에 소요된 시간 (초).
        api_calls_used (int): 검색 중 사용된 API 호출 횟수.
        confidence_score (float): 검색 결과의 전체 신뢰도 점수 (0.0 ~ 1.0).
        time_period (str): 검색 대상 기간 (예: '최근 1주일').
        raw_responses (List[str]): API 호출의 원본 응답 데이터.
        detailed_issues_count (int): 상세 정보가 수집된 이슈 개수 (기본값 0).
        total_detail_collection_time (float): 상세 정보 수집에 소요된 총 시간 (기본값 0.0).
        average_detail_confidence (float): 상세 정보의 평균 신뢰도 점수 (기본값 0.0).
    """
    query_keywords: List[str]
    total_found: int
    issues: List[IssueItem]
    search_time: float
    api_calls_used: int
    confidence_score: float
    time_period: str
    raw_responses: List[str]
    detailed_issues_count: int = 0
    total_detail_collection_time: float = 0.0
    average_detail_confidence: float = 0.0