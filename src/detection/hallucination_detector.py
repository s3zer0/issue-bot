"""
환각 탐지 모듈 (리팩토링 후 호환성 유지).

이 파일은 기존 코드와의 호환성을 위해 유지되며,
실제 구현은 hallucination_detection 패키지로 이동했습니다.
"""

# 기존 import를 유지하면서 새로운 모듈로 리다이렉트
from src.hallucination_detection import (
    RePPLDetector,
    EnhancedIssueSearcher,
    RePPLScore,
    HallucinationScore,
    CombinedHallucinationScore
)

# 기존 클래스명 호환성 유지
RePPLHallucinationDetector = RePPLDetector
RePPLEnhancedIssueSearcher = EnhancedIssueSearcher

# 기존 코드와의 호환성을 위한 래퍼 함수들
def create_reppl_detector(model_name=None):
    """기존 코드 호환성을 위한 팩토리 함수."""
    return RePPLDetector(model_name)

def create_enhanced_searcher(api_key=None):
    """기존 코드 호환성을 위한 팩토리 함수."""
    return EnhancedIssueSearcher(api_key)

# 기존 코드에서 사용하던 모든 심볼 export
__all__ = [
    'RePPLHallucinationDetector',
    'RePPLEnhancedIssueSearcher',
    'RePPLScore',
    'HallucinationScore',
    'CombinedHallucinationScore',
    'create_reppl_detector',
    'create_enhanced_searcher'
]