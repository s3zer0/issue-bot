# src/models.py
"""
프로젝트에서 사용하는 모든 데이터 모델(dataclasses)을 정의하는 모듈
"""

from dataclasses import dataclass, field
from typing import List, Optional

# --- Keyword Generation Models ---
@dataclass
class KeywordResult:
    """키워드 생성 결과를 담는 데이터 클래스"""
    topic: str
    primary_keywords: List[str]
    related_terms: List[str]
    context_keywords: List[str]
    confidence_score: float
    generation_time: float
    raw_response: str

# --- Issue Search Models ---
@dataclass
class EntityInfo:
    """관련 인물/기관 정보"""
    name: str
    role: str
    relevance: float
    entity_type: str
    description: str

@dataclass
class ImpactAnalysis:
    """영향도 분석 정보"""
    impact_level: str
    impact_score: float
    affected_sectors: List[str]
    geographic_scope: str
    time_sensitivity: str
    reasoning: str

@dataclass
class TimelineEvent:
    """시간순 이벤트 정보"""
    date: str
    event_type: str
    description: str
    importance: float
    source: str

@dataclass
class IssueItem:
    """개별 이슈 정보를 담는 데이터 클래스"""
    title: str
    summary: str
    source: str
    published_date: Optional[str]
    relevance_score: float
    category: str
    content_snippet: str
    detailed_content: Optional[str] = None
    related_entities: List[EntityInfo] = field(default_factory=list)
    impact_analysis: Optional[ImpactAnalysis] = None
    timeline_events: List[TimelineEvent] = field(default_factory=list)
    background_context: Optional[str] = None
    detail_collection_time: Optional[float] = None
    detail_confidence: Optional[float] = None

@dataclass
class SearchResult:
    """이슈 검색 결과를 담는 데이터 클래스"""
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