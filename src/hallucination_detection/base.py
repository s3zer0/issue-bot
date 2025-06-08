"""
환각 탐지기의 기본 인터페이스 및 공통 기능 정의.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
from loguru import logger

from src.models import IssueItem
from .models import HallucinationScore


class BaseHallucinationDetector(ABC):
    """환각 탐지기의 기본 추상 클래스."""

    def __init__(self, name: str):
        """
        환각 탐지기 기본 초기화.

        Args:
            name (str): 탐지기 이름
        """
        self.name = name
        self.is_initialized = False
        logger.info(f"{self.name} 환각 탐지기 초기화 중...")

    @abstractmethod
    async def analyze_text(self, text: str, context: Optional[str] = None) -> HallucinationScore:
        """
        텍스트의 환각 현상을 분석합니다.

        Args:
            text (str): 분석할 텍스트
            context (Optional[str]): 텍스트의 맥락 정보

        Returns:
            HallucinationScore: 분석 결과
        """
        pass

    async def analyze_issue(self, issue: IssueItem, topic: str) -> HallucinationScore:
        """
        이슈 아이템의 환각 현상을 분석합니다.

        Args:
            issue (IssueItem): 분석할 이슈
            topic (str): 이슈의 주제

        Returns:
            HallucinationScore: 분석 결과
        """
        # 분석할 텍스트 구성
        text_to_analyze = self._prepare_issue_text(issue)

        # 환각 탐지 수행
        score = await self.analyze_text(text_to_analyze, context=topic)

        # 분석 메타데이터 추가
        score.analysis_details.update({
            "issue_title": issue.title,
            "issue_source": issue.source,
            "analyzed_text_length": len(text_to_analyze)
        })

        return score

    def _prepare_issue_text(self, issue: IssueItem) -> str:
        """
        이슈 객체에서 분석할 텍스트를 추출하고 준비합니다.

        Args:
            issue (IssueItem): 이슈 객체

        Returns:
            str: 분석할 텍스트
        """
        text_parts = [f"제목: {issue.title}", f"요약: {issue.summary}"]

        if issue.detailed_content:
            # 상세 내용이 너무 길면 일부만 사용
            content_preview = issue.detailed_content[:1000]
            text_parts.append(f"상세내용: {content_preview}")

        return "\n".join(text_parts)

    def validate_confidence_threshold(self, confidence: float, threshold: float = 0.5) -> bool:
        """
        신뢰도가 임계값을 통과하는지 확인합니다.

        Args:
            confidence (float): 신뢰도 점수
            threshold (float): 임계값 (기본: 0.5)

        Returns:
            bool: 임계값 통과 여부
        """
        passed = confidence >= threshold
        logger.debug(f"{self.name} - 신뢰도: {confidence:.2f} {'통과' if passed else '실패'} (임계값: {threshold})")
        return passed