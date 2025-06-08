"""
환각 탐지가 통합된 향상된 이슈 검색기.

기본 이슈 검색 후 여러 환각 탐지 방법을 적용하여
신뢰도 높은 결과만을 필터링합니다.
"""

import asyncio
from typing import Optional, List, Dict
from loguru import logger

from src.models import KeywordResult, IssueItem, SearchResult
from src.issue_searcher import create_issue_searcher
from src.keyword_generator import generate_keywords_for_topic
from .reppl_detector import RePPLDetector
from .consistency_checker import SelfConsistencyChecker
from .models import CombinedHallucinationScore, HallucinationScore


class EnhancedIssueSearcher:
    """여러 환각 탐지 방법이 통합된 이슈 검색기."""

    def __init__(
            self,
            api_key: Optional[str] = None,
            enable_reppl: bool = True,
            enable_consistency: bool = True,
            enable_llm_judge: bool = True,  # 새로 추가
            min_confidence_threshold: float = 0.5,
            consistency_check_threshold: float = 0.6,
            llm_judge_threshold: float = 0.7  # 새로 추가
    ):
        """
        환각 탐지 기능이 강화된 이슈 검색기를 초기화합니다.

        Args:
            api_key (Optional[str]): API 키
            enable_reppl (bool): RePPL 탐지 활성화 여부
            enable_consistency (bool): 자기 일관성 검사 활성화 여부
            enable_llm_judge (bool): LLM Judge 활성화 여부
            min_confidence_threshold (float): 최소 신뢰도 임계값
            consistency_check_threshold (float): 일관성 검사를 적용할 최소 신뢰도
            llm_judge_threshold (float): LLM Judge를 적용할 최소 신뢰도
        """
        # 기본 이슈 검색기
        self.base_searcher = create_issue_searcher(api_key)

        # 환각 탐지기들
        self.detectors = {}
        if enable_reppl:
            self.detectors['RePPL'] = RePPLDetector()
        if enable_consistency:
            self.detectors['Self-Consistency'] = SelfConsistencyChecker()
        if enable_llm_judge:
            from .llm_judge import LLMJudgeDetector
            self.detectors['LLM-Judge'] = LLMJudgeDetector()

        # 설정
        self.min_confidence_threshold = min_confidence_threshold
        self.consistency_check_threshold = consistency_check_threshold
        self.llm_judge_threshold = llm_judge_threshold

        # 탐지 방법별 가중치 (LLM Judge 추가)
        self.detector_weights = {
            'RePPL': 0.3,
            'Self-Consistency': 0.3,
            'LLM-Judge': 0.4  # LLM Judge에 더 높은 가중치
        }

        logger.info(
            f"향상된 이슈 검색기 초기화 완료 "
            f"(활성 탐지기: {list(self.detectors.keys())})"
        )

    async def search_with_validation(
            self,
            keyword_result: KeywordResult,
            time_period: str,
            max_retries: int = 2
    ) -> SearchResult:
        """
        키워드로 이슈를 검색하고 환각 탐지를 수행합니다.

        Args:
            keyword_result (KeywordResult): 검색 키워드
            time_period (str): 검색 기간
            max_retries (int): 최대 재시도 횟수

        Returns:
            SearchResult: 검증된 이슈 검색 결과
        """
        current_keywords = keyword_result

        for attempt in range(max_retries):
            logger.info(f"이슈 검색 시도 {attempt + 1}/{max_retries}")

            # 1. 기본 검색 수행
            search_result = await self.base_searcher.search_issues_from_keywords(
                current_keywords, time_period, collect_details=True
            )

            if not search_result.issues:
                logger.warning("검색 결과가 없습니다.")
                continue

            # 2. 각 이슈에 대해 환각 탐지 수행
            validated_issues = await self._validate_issues(
                search_result.issues,
                current_keywords.topic
            )

            # 3. 충분한 수의 신뢰할 수 있는 이슈를 찾았는지 확인
            if len(validated_issues) >= 3 or attempt == max_retries - 1:
                search_result.issues = validated_issues
                search_result.total_found = len(validated_issues)

                # 환각 탐지 메타데이터 추가
                search_result.detailed_issues_count = len([
                    i for i in validated_issues
                    if hasattr(i, 'hallucination_analysis')
                ])

                return search_result

            # 4. 결과가 부족하면 키워드 재생성
            logger.info(
                f"신뢰도 높은 이슈 부족 ({len(validated_issues)}개), "
                f"키워드 재생성 중..."
            )
            current_keywords = await generate_keywords_for_topic(
                f"{current_keywords.topic}의 다른 측면"
            )

        return search_result

    async def _validate_issues(
            self,
            issues: List[IssueItem],
            topic: str
    ) -> List[IssueItem]:
        """
        이슈 목록에 대해 환각 탐지를 수행합니다.

        Args:
            issues (List[IssueItem]): 검증할 이슈 목록
            topic (str): 이슈 주제

        Returns:
            List[IssueItem]: 검증된 이슈 목록
        """
        # 병렬로 모든 이슈 검증
        validation_tasks = [
            self._validate_single_issue(issue, topic)
            for issue in issues
        ]

        validation_results = await asyncio.gather(*validation_tasks)

        # 검증을 통과한 이슈만 필터링
        validated_issues = [
            issue for issue in validation_results
            if issue is not None
        ]

        logger.info(
            f"환각 탐지 완료: {len(issues)}개 중 "
            f"{len(validated_issues)}개 통과"
        )

        return validated_issues

    async def _validate_single_issue(
            self,
            issue: IssueItem,
            topic: str
    ) -> Optional[IssueItem]:
        """
        단일 이슈에 대해 환각 탐지를 수행합니다.

        Args:
            issue (IssueItem): 검증할 이슈
            topic (str): 이슈 주제

        Returns:
            Optional[IssueItem]: 검증 통과 시 이슈, 실패 시 None
        """
        try:
            # 1. 각 탐지기로 분석 수행
            individual_scores = {}

            # RePPL 분석
            if 'RePPL' in self.detectors:
                reppl_score = await self.detectors['RePPL'].analyze_issue(issue, topic)
                individual_scores['RePPL'] = reppl_score

            # 자기 일관성 검사 (RePPL 점수가 임계값 이상일 때만)
            if ('Self-Consistency' in self.detectors and
                    (not individual_scores or
                     min(s.confidence for s in individual_scores.values()) >= self.consistency_check_threshold)):
                consistency_score = await self.detectors['Self-Consistency'].analyze_text(
                    issue.summary,
                    context=issue.title
                )
                individual_scores['Self-Consistency'] = consistency_score

            # LLM Judge 검사 (이전 점수들이 임계값 이상일 때만)
            if ('LLM-Judge' in self.detectors and
                    (not individual_scores or
                     min(s.confidence for s in individual_scores.values()) >= self.llm_judge_threshold)):
                # 상세 내용이 있으면 포함하여 평가
                text_to_judge = issue.summary
                if issue.detailed_content:
                    text_to_judge = f"{issue.summary}\n\n{issue.detailed_content[:1000]}"

                llm_judge_score = await self.detectors['LLM-Judge'].analyze_text(
                    text_to_judge,
                    context=f"주제: {topic}, 제목: {issue.title}"
                )
                individual_scores['LLM-Judge'] = llm_judge_score

            # 2. 점수 통합
            combined_score = CombinedHallucinationScore(
                individual_scores=individual_scores,
                weights=self.detector_weights,
                final_confidence=0  # __post_init__에서 계산됨
            )

            # 3. 임계값 확인
            if combined_score.final_confidence >= self.min_confidence_threshold:
                # 분석 결과를 이슈에 추가
                setattr(issue, 'hallucination_analysis', combined_score)
                setattr(issue, 'hallucination_confidence', combined_score.final_confidence)

                logger.debug(
                    f"이슈 '{issue.title[:30]}...' 검증 통과 "
                    f"(신뢰도: {combined_score.final_confidence:.2f})"
                )

                return issue
            else:
                logger.warning(
                    f"이슈 '{issue.title[:30]}...' 제외됨 - "
                    f"신뢰도: {combined_score.final_confidence:.2f} < {self.min_confidence_threshold}"
                )

                # LLM Judge의 문제 영역 정보 로그
                if 'LLM-Judge' in individual_scores:
                    judge_score = individual_scores['LLM-Judge']
                    if hasattr(judge_score, 'problematic_areas') and judge_score.problematic_areas:
                        logger.warning(f"LLM Judge가 발견한 문제점: {judge_score.problematic_areas[0]['issue']}")

                return None

        except Exception as e:
            logger.error(f"이슈 검증 중 오류: {e}")
            return None

    def update_detector_weights(self, weights: Dict[str, float]):
        """
        탐지기별 가중치를 업데이트합니다.

        Args:
            weights (Dict[str, float]): 새로운 가중치
        """
        self.detector_weights.update(weights)
        logger.info(f"탐지기 가중치 업데이트: {self.detector_weights}")

    def set_confidence_threshold(self, threshold: float):
        """
        최소 신뢰도 임계값을 설정합니다.

        Args:
            threshold (float): 새로운 임계값
        """
        self.min_confidence_threshold = threshold
        logger.info(f"신뢰도 임계값 변경: {threshold}")

    async def analyze_text_with_all_detectors(
            self,
            text: str,
            context: Optional[str] = None
    ) -> CombinedHallucinationScore:
        """
        텍스트에 대해 모든 활성 탐지기를 실행합니다.

        Args:
            text (str): 분석할 텍스트
            context (Optional[str]): 텍스트 맥락

        Returns:
            CombinedHallucinationScore: 통합 분석 결과
        """
        individual_scores = {}

        # 모든 탐지기로 분석
        for name, detector in self.detectors.items():
            try:
                score = await detector.analyze_text(text, context)
                individual_scores[name] = score
            except Exception as e:
                logger.error(f"{name} 분석 실패: {e}")

        # 결과 통합
        return CombinedHallucinationScore(
            individual_scores=individual_scores,
            weights=self.detector_weights,
            final_confidence=0  # __post_init__에서 계산됨
        )