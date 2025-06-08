"""
환각 탐지 모듈 패키지.

이 패키지는 LLM이 생성한 콘텐츠의 신뢰성을 검증하기 위한 다양한 탐지 방법론을 제공합니다.
주요 구성 요소:
- RePPL (Repetition as Pre-Perplexity) 탐지기
- 자기 일관성 검사기 (Self-Consistency Checker)
- 통합 환각 탐지 검색기
"""

from .models import (
    HallucinationScore,
    RePPLScore,
    ConsistencyScore,
    CombinedHallucinationScore
)
from .reppl_detector import RePPLDetector
from .consistency_checker import SelfConsistencyChecker
from .enhanced_searcher import EnhancedIssueSearcher

__all__ = [
    # 데이터 모델
    'HallucinationScore',
    'RePPLScore',
    'ConsistencyScore',
    'CombinedHallucinationScore',
    # 탐지기
    'RePPLDetector',
    'SelfConsistencyChecker',
    # 통합 검색기
    'EnhancedIssueSearcher'
]