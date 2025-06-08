"""
환각 탐지 관련 데이터 모델 정의.

각 탐지 방법론의 결과를 구조화하고, 통합 점수를 관리하는 데이터 클래스들을 정의합니다.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


@dataclass
class HallucinationScore(ABC):
    """환각 탐지 점수의 기본 추상 클래스."""

    confidence: float  # 최종 신뢰도 점수 (0.0 ~ 1.0)
    method_name: str = ""  # 탐지 방법 이름 (기본값 추가)
    analysis_details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """신뢰도 점수 범위 검증."""
        self.confidence = max(0.0, min(1.0, self.confidence))

    @abstractmethod
    def get_summary(self) -> str:
        """분석 결과 요약 문자열 반환."""
        pass


@dataclass
class RePPLScore(HallucinationScore):
    """RePPL 기반 환각 탐지 분석 결과."""

    repetition_score: float = 0.0
    perplexity: float = 0.0
    semantic_entropy: float = 0.0
    repeated_phrases: List[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        self.method_name = "RePPL"

    def get_summary(self) -> str:
        """RePPL 분석 결과 요약."""
        return (
            f"RePPL 분석 - 신뢰도: {self.confidence:.2f} "
            f"(반복성: {self.repetition_score:.2f}, "
            f"PPL: {self.perplexity:.1f}, "
            f"엔트로피: {self.semantic_entropy:.2f})"
        )


@dataclass
class ConsistencyScore(HallucinationScore):
    """자기 일관성 검사 결과."""

    consistency_rate: float = 0.0  # 응답 간 일치율 (0.0 ~ 1.0)
    num_queries: int = 0  # 검사에 사용된 쿼리 수
    num_consistent: int = 0  # 일관된 응답 수
    variations: List[str] = field(default_factory=list)  # 서로 다른 응답 변형들
    common_elements: List[str] = field(default_factory=list)  # 공통 요소들
    divergent_elements: List[str] = field(default_factory=list)  # 불일치 요소들

    def __post_init__(self):
        super().__post_init__()
        self.method_name = "Self-Consistency"

    def get_summary(self) -> str:
        """자기 일관성 검사 결과 요약."""
        return (
            f"자기 일관성 - 신뢰도: {self.confidence:.2f} "
            f"(일치율: {self.consistency_rate:.2f}, "
            f"검사: {self.num_consistent}/{self.num_queries})"
        )


@dataclass
class CombinedHallucinationScore:
    """여러 환각 탐지 방법의 결과를 통합한 최종 점수."""

    final_confidence: float = 0.0  # 통합 신뢰도 점수
    individual_scores: Dict[str, HallucinationScore] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)  # 각 방법의 가중치
    passed_threshold: bool = False  # 신뢰도 임계값 통과 여부
    recommendation: str = ""  # 권장 조치

    def __post_init__(self):
        """최종 점수 계산 및 권장 조치 설정."""
        if not self.weights:
            # 기본 가중치 설정
            self.weights = {
                "RePPL": 0.5,
                "Self-Consistency": 0.5
            }

        # 가중 평균으로 최종 점수 계산
        if self.individual_scores and self.final_confidence == 0.0:
            total_weight = 0
            weighted_sum = 0

            for method, score in self.individual_scores.items():
                weight = self.weights.get(method, 0.5)
                weighted_sum += score.confidence * weight
                total_weight += weight

            self.final_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

        # 임계값 확인 및 권장 조치 설정
        self.passed_threshold = self.final_confidence >= 0.5

        if self.final_confidence >= 0.8:
            self.recommendation = "높은 신뢰도 - 사용 권장"
        elif self.final_confidence >= 0.5:
            self.recommendation = "중간 신뢰도 - 주의하여 사용"
        else:
            self.recommendation = "낮은 신뢰도 - 재검증 필요"

    def get_detailed_report(self) -> str:
        """상세 분석 보고서 생성."""
        report = f"=== 환각 탐지 종합 보고서 ===\n"
        report += f"최종 신뢰도: {self.final_confidence:.2f}\n"
        report += f"판정: {self.recommendation}\n\n"

        for method, score in self.individual_scores.items():
            report += f"{score.get_summary()}\n"

        return report