# tests/test_threshold_manager.py
"""
ThresholdManager 클래스에 대한 단위 테스트.

이 테스트는 신뢰도 점수에 따른 등급 분류, 다음 탐지기 진행 여부 결정 등
임계값과 관련된 핵심 로직의 정확성을 검증합니다.
"""

import pytest
import sys
import os

# 테스트 실행을 위해 프로젝트 루트 경로를 sys.path에 추가합니다.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 테스트 대상 모듈 임포트
from src.hallucination_detection.threshold_manager import ThresholdManager, ConfidenceLevel, ThresholdConfig

class TestThresholdManager:
    """ThresholdManager의 다양한 시나리오를 테스트하는 클래스."""

    def test_initialization_with_defaults(self):
        """기본값으로 ThresholdManager가 올바르게 초기화되는지 테스트합니다."""
        manager = ThresholdManager()
        assert manager.thresholds is not None
        assert isinstance(manager.thresholds, ThresholdConfig)
        assert manager.thresholds.min_confidence_threshold == 0.5

    def test_initialization_with_override(self):
        """사용자 정의 설정으로 ThresholdManager가 올바르게 초기화되는지 테스트합니다."""
        custom_config = ThresholdConfig(min_confidence_threshold=0.8)
        manager = ThresholdManager(config_override=custom_config)
        assert manager.thresholds.min_confidence_threshold == 0.8

    @pytest.mark.parametrize("score, expected_level", [
        (0.9, ConfidenceLevel.VERY_HIGH),
        (0.85, ConfidenceLevel.VERY_HIGH), # 경계값 테스트
        (0.8, ConfidenceLevel.HIGH),
        (0.70, ConfidenceLevel.HIGH),      # 경계값 테스트
        (0.6, ConfidenceLevel.MODERATE),
        (0.50, ConfidenceLevel.MODERATE),  # 경계값 테스트
        (0.4, ConfidenceLevel.LOW),
        (0.30, ConfidenceLevel.LOW),       # 경계값 테스트
        (0.2, ConfidenceLevel.VERY_LOW),
        (0.0, ConfidenceLevel.VERY_LOW),
        (1.0, ConfidenceLevel.VERY_HIGH),
    ])
    def test_classify_confidence(self, score, expected_level):
        """
        다양한 신뢰도 점수에 대해 classify_confidence 메서드가
        정확한 등급을 반환하는지 테스트합니다.
        """
        manager = ThresholdManager()
        assert manager.classify_confidence(score) == expected_level

    @pytest.mark.parametrize("current_confidence, detector_name, expected_result", [
        # Self-Consistency 검사 진행 여부 테스트
        (0.3, "Self-Consistency", False), # 임계값(0.4) 미만, 진행 불가
        (0.4, "Self-Consistency", True),  # 임계값(0.4) 이상, 진행 가능
        (0.45, "Self-Consistency", True),
        # LLM-Judge 검사 진행 여부 테스트
        (0.4, "LLM-Judge", False),        # 임계값(0.5) 미만, 진행 불가
        (0.5, "LLM-Judge", True),         # 임계값(0.5) 이상, 진행 가능
        (0.6, "LLM-Judge", True),
        # 알 수 없는 탐지기 이름에 대해서는 항상 True 반환
        (0.1, "Unknown-Detector", True)
    ])
    def test_should_proceed_to_next_detector(self, current_confidence, detector_name, expected_result):
        """
        should_proceed_to_next_detector 메서드가 현재 신뢰도와 다음 탐지기 이름을
        기반으로 진행 여부를 정확히 판단하는지 테스트합니다.
        """
        manager = ThresholdManager()
        assert manager.should_proceed_to_next_detector(current_confidence, detector_name) == expected_result

    def test_get_confidence_summary(self):
        """
        get_confidence_summary 메서드가 신뢰도 점수에 대해
        올바른 요약 정보를 반환하는지 테스트합니다.
        """
        manager = ThresholdManager()
        summary = manager.get_confidence_summary(0.75) # HIGH 등급
        assert summary['score'] == 0.75
        assert summary['level'] == ConfidenceLevel.HIGH
        assert "대체로 신뢰할 수 있습니다" in summary['recommendation']
        assert summary['color'] == 0x90EE90 # 연초록

