"""
환각 탐지 신뢰도 임계값 관리 모듈.

각 탐지 방법별 임계값을 중앙에서 관리하고,
신뢰도 등급을 분류하는 기능을 제공합니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from loguru import logger

from src.config import config


class ConfidenceLevel(Enum):
    """신뢰도 등급 정의."""
    VERY_HIGH = "매우 높음"  # 0.85 이상
    HIGH = "높음"  # 0.70 ~ 0.85
    MODERATE = "보통"  # 0.50 ~ 0.70
    LOW = "낮음"  # 0.30 ~ 0.50
    VERY_LOW = "매우 낮음"  # 0.30 미만


@dataclass
class ThresholdConfig:
    """환각 탐지 임계값 설정."""

    # 전체 시스템 임계값
    min_confidence_threshold: float = 0.5  # 최소 허용 신뢰도

    # 개별 탐지기 임계값
    reppl_threshold: float = 0.5  # RePPL 최소 신뢰도
    consistency_threshold: float = 0.6  # 자기 일관성 최소 신뢰도
    llm_judge_threshold: float = 0.7  # LLM Judge 최소 신뢰도

    # 단계별 적용 임계값 (다음 탐지기로 진행하기 위한 최소값)
    proceed_to_consistency: float = 0.4  # 일관성 검사 진행 임계값
    proceed_to_llm_judge: float = 0.5  # LLM Judge 진행 임계값

    # 신뢰도 등급 경계값
    very_high_boundary: float = 0.85
    high_boundary: float = 0.70
    moderate_boundary: float = 0.50
    low_boundary: float = 0.30

    # 가중치 설정 (기본값)
    default_weights: Dict[str, float] = field(default_factory=lambda: {
        "RePPL": 0.3,
        "Self-Consistency": 0.3,
        "LLM-Judge": 0.4
    })

    # 보고서 생성 옵션
    include_low_confidence: bool = False  # 낮은 신뢰도 이슈 포함 여부
    detailed_analysis_threshold: float = 0.6  # 상세 분석을 포함할 최소 신뢰도


class ThresholdManager:
    """환각 탐지 임계값 관리자."""

    def __init__(self, config_override: Optional[ThresholdConfig] = None):
        """
        임계값 관리자를 초기화합니다.

        Args:
            config_override: 사용자 정의 임계값 설정 (없으면 환경 변수에서 로드)
        """
        if config_override:
            self.thresholds = config_override
        else:
            self.thresholds = self._load_from_env()

        logger.info(f"임계값 관리자 초기화 완료 - 최소 신뢰도: {self.thresholds.min_confidence_threshold}")

    def _load_from_env(self) -> ThresholdConfig:
        """환경 변수에서 임계값 설정을 로드합니다."""
        try:
            return ThresholdConfig(
                min_confidence_threshold=float(config.get('MIN_CONFIDENCE_THRESHOLD', 0.5)),
                reppl_threshold=float(config.get('REPPL_THRESHOLD', 0.5)),
                consistency_threshold=float(config.get('CONSISTENCY_THRESHOLD', 0.6)),
                llm_judge_threshold=float(config.get('LLM_JUDGE_THRESHOLD', 0.7)),
                proceed_to_consistency=float(config.get('PROCEED_TO_CONSISTENCY', 0.4)),
                proceed_to_llm_judge=float(config.get('PROCEED_TO_LLM_JUDGE', 0.5)),
                very_high_boundary=float(config.get('VERY_HIGH_BOUNDARY', 0.85)),
                high_boundary=float(config.get('HIGH_BOUNDARY', 0.70)),
                moderate_boundary=float(config.get('MODERATE_BOUNDARY', 0.50)),
                low_boundary=float(config.get('LOW_BOUNDARY', 0.30)),
                include_low_confidence=config.get('INCLUDE_LOW_CONFIDENCE', 'false').lower() == 'true',
                detailed_analysis_threshold=float(config.get('DETAILED_ANALYSIS_THRESHOLD', 0.6))
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"환경 변수 로드 실패, 기본값 사용: {e}")
            return ThresholdConfig()

    def classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """
        신뢰도 점수를 등급으로 분류합니다.

        Args:
            confidence: 신뢰도 점수 (0.0 ~ 1.0)

        Returns:
            ConfidenceLevel: 신뢰도 등급
        """
        if confidence >= self.thresholds.very_high_boundary:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= self.thresholds.high_boundary:
            return ConfidenceLevel.HIGH
        elif confidence >= self.thresholds.moderate_boundary:
            return ConfidenceLevel.MODERATE
        elif confidence >= self.thresholds.low_boundary:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def should_proceed_to_next_detector(self, current_confidence: float, next_detector: str) -> bool:
        """
        현재 신뢰도를 기반으로 다음 탐지기로 진행할지 결정합니다.

        Args:
            current_confidence: 현재까지의 신뢰도
            next_detector: 다음 탐지기 이름

        Returns:
            bool: 진행 여부
        """
        if next_detector == "Self-Consistency":
            return current_confidence >= self.thresholds.proceed_to_consistency
        elif next_detector == "LLM-Judge":
            return current_confidence >= self.thresholds.proceed_to_llm_judge
        return True

    def filter_issues_by_confidence(self, issues: List) -> Tuple[List, List, List]:
        """
        신뢰도에 따라 이슈를 분류합니다.

        Args:
            issues: 이슈 리스트 (hallucination_confidence 속성 필요)

        Returns:
            Tuple[List, List, List]: (높은 신뢰도, 중간 신뢰도, 낮은 신뢰도) 이슈들
        """
        high_confidence = []
        moderate_confidence = []
        low_confidence = []

        for issue in issues:
            confidence = getattr(issue, 'hallucination_confidence', 0.0)
            level = self.classify_confidence(confidence)

            if level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
                high_confidence.append(issue)
            elif level == ConfidenceLevel.MODERATE:
                moderate_confidence.append(issue)
            else:
                low_confidence.append(issue)

        logger.info(
            f"이슈 분류 완료 - 높음: {len(high_confidence)}, "
            f"보통: {len(moderate_confidence)}, 낮음: {len(low_confidence)}"
        )

        return high_confidence, moderate_confidence, low_confidence

    def get_confidence_summary(self, confidence: float) -> Dict[str, any]:
        """
        신뢰도에 대한 상세 요약 정보를 반환합니다.

        Args:
            confidence: 신뢰도 점수

        Returns:
            Dict: 신뢰도 요약 정보
        """
        level = self.classify_confidence(confidence)

        return {
            "score": confidence,
            "level": level,
            "level_text": level.value,
            "passed": confidence >= self.thresholds.min_confidence_threshold,
            "recommendation": self._get_recommendation(level),
            "color": self._get_color_code(level)
        }

    def _get_recommendation(self, level: ConfidenceLevel) -> str:
        """신뢰도 등급에 따른 권장 사항을 반환합니다."""
        recommendations = {
            ConfidenceLevel.VERY_HIGH: "신뢰할 수 있는 정보입니다. 안심하고 사용하세요.",
            ConfidenceLevel.HIGH: "대체로 신뢰할 수 있습니다. 중요한 결정에는 추가 확인을 권장합니다.",
            ConfidenceLevel.MODERATE: "주의가 필요합니다. 반드시 다른 출처와 교차 확인하세요.",
            ConfidenceLevel.LOW: "신뢰도가 낮습니다. 참고용으로만 사용하고 반드시 검증하세요.",
            ConfidenceLevel.VERY_LOW: "매우 신뢰도가 낮습니다. 사용을 권장하지 않습니다."
        }
        return recommendations.get(level, "신뢰도를 평가할 수 없습니다.")

    def _get_color_code(self, level: ConfidenceLevel) -> str:
        """Discord 임베드용 색상 코드를 반환합니다."""
        colors = {
            ConfidenceLevel.VERY_HIGH: 0x00FF00,  # 초록
            ConfidenceLevel.HIGH: 0x90EE90,  # 연초록
            ConfidenceLevel.MODERATE: 0xFFFF00,  # 노랑
            ConfidenceLevel.LOW: 0xFFA500,  # 주황
            ConfidenceLevel.VERY_LOW: 0xFF0000  # 빨강
        }
        return colors.get(level, 0x808080)  # 기본: 회색

    def should_include_in_report(self, confidence: float) -> bool:
        """
        해당 신뢰도의 이슈를 보고서에 포함할지 결정합니다.

        Args:
            confidence: 신뢰도 점수

        Returns:
            bool: 포함 여부
        """
        if self.thresholds.include_low_confidence:
            return True  # 모든 이슈 포함

        return confidence >= self.thresholds.min_confidence_threshold

    def should_include_detailed_analysis(self, confidence: float) -> bool:
        """
        상세 분석을 포함할지 결정합니다.

        Args:
            confidence: 신뢰도 점수

        Returns:
            bool: 상세 분석 포함 여부
        """
        return confidence >= self.thresholds.detailed_analysis_threshold

    def get_weights_for_confidence(self, current_confidence: float) -> Dict[str, float]:
        """
        현재 신뢰도에 따라 동적으로 가중치를 조정합니다.

        낮은 신뢰도일수록 LLM Judge의 가중치를 높입니다.

        Args:
            current_confidence: 현재 신뢰도

        Returns:
            Dict[str, float]: 조정된 가중치
        """
        weights = self.thresholds.default_weights.copy()

        # 신뢰도가 낮을수록 LLM Judge 가중치 증가
        if current_confidence < 0.5:
            weights["LLM-Judge"] = 0.5
            weights["RePPL"] = 0.25
            weights["Self-Consistency"] = 0.25

        return weights


# 전역 임계값 관리자 인스턴스
threshold_manager = ThresholdManager()