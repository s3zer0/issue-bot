"""
환각 탐지가 통합된 향상된 이슈 검색기 (임계값 관리자 통합 버전).

기본 이슈 검색 후 여러 환각 탐지 방법을 적용하여
신뢰도 높은 결과만을 필터링합니다.
"""

import asyncio
from typing import Optional, List, Dict
from loguru import logger

from src.models import KeywordResult, IssueItem, SearchResult
from src.issue_searcher import create_issue_searcher
from src.keyword_generator import generate_keywords_for_topic
from src.hallucination_detection.threshold_manager import ThresholdManager
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
            enable_llm_judge: bool = True,
            threshold_manager: Optional[ThresholdManager] = None
    ):
        """
        환각 탐지 기능이 강화된 이슈 검색기를 초기화합니다.

        Args:
            api_key (Optional[str]): API 키
            enable_reppl (bool): RePPL 탐지 활성화 여부
            enable_consistency (bool): 자기 일관성 검사 활성화 여부
            enable_llm_judge (bool): LLM Judge 활성화 여부
            threshold_manager (Optional[ThresholdManager]): 임계값 관리자
        """
        # 기본 이슈 검색기
        self.base_searcher = create_issue_searcher(api_key)

        # 임계값 관리자
        self.threshold_manager = threshold_manager or ThresholdManager()

        # 환각 탐지기들
        self.detectors = {}
        if enable_reppl:
            self.detectors['RePPL'] = RePPLDetector()
        if enable_consistency:
            self.detectors['Self-Consistency'] = SelfConsistencyChecker()
        if enable_llm_judge:
            from .llm_judge import LLMJudgeDetector
            self.detectors['LLM-Judge'] = LLMJudgeDetector()

        logger.info(
            f"향상된 이슈 검색기 초기화 완료 "
            f"(활성 탐지기: {list(self.detectors.keys())}, "
            f"최소 신뢰도: {self.threshold_manager.thresholds.min_confidence_threshold})"
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
        all_attempts_issues = []  # 모든 시도의 이슈를 저장

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

            all_attempts_issues.extend(validated_issues)

            # 3. 충분한 수의 신뢰할 수 있는 이슈를 찾았는지 확인
            high_confidence_issues = [
                issue for issue in validated_issues
                if getattr(issue, 'hallucination_confidence', 0) >=
                   self.threshold_manager.thresholds.high_boundary
            ]

            if len(high_confidence_issues) >= 3 or attempt == max_retries - 1:
                # 최종 결과 구성
                search_result.issues = all_attempts_issues
                search_result.total_found = len(all_attempts_issues)

                # 환각 탐지 메타데이터 추가
                search_result.detailed_issues_count = len([
                    i for i in all_attempts_issues
                    if hasattr(i, 'hallucination_analysis')
                ])

                # 신뢰도별 통계 추가
                high, moderate, low = self.threshold_manager.filter_issues_by_confidence(
                    all_attempts_issues
                )
                setattr(search_result, 'confidence_distribution', {
                    'high': len(high),
                    'moderate': len(moderate),
                    'low': len(low)
                })

                return search_result

            # 4. 결과가 부족하면 키워드 재생성
            logger.info(
                f"높은 신뢰도 이슈 부족 ({len(high_confidence_issues)}개), "
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

        # 신뢰도별 분류 로깅
        high, moderate, low = self.threshold_manager.filter_issues_by_confidence(
            [i for i in validation_results if i is not None]
        )

        logger.info(
            f"환각 탐지 완료: {len(issues)}개 중 "
            f"{len(validated_issues)}개 통과 "
            f"(높음: {len(high)}, 보통: {len(moderate)}, 낮음: {len(low)})"
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
            current_confidence = 1.0  # 초기 신뢰도

            # RePPL 분석
            if 'RePPL' in self.detectors:
                reppl_score = await self.detectors['RePPL'].analyze_issue(issue, topic)
                individual_scores['RePPL'] = reppl_score
                current_confidence = reppl_score.confidence

            # 자기 일관성 검사 (조건부 실행)
            if ('Self-Consistency' in self.detectors and
                    self.threshold_manager.should_proceed_to_next_detector(
                        current_confidence, 'Self-Consistency')):
                consistency_score = await self.detectors['Self-Consistency'].analyze_text(
                    issue.summary,
                    context=issue.title
                )
                individual_scores['Self-Consistency'] = consistency_score
                current_confidence = min(current_confidence, consistency_score.confidence)

            # LLM Judge 검사 (조건부 실행)
            if ('LLM-Judge' in self.detectors and
                    self.threshold_manager.should_proceed_to_next_detector(
                        current_confidence, 'LLM-Judge')):
                # 상세 내용이 있으면 포함하여 평가
                text_to_judge = issue.summary
                if issue.detailed_content:
                    text_to_judge = f"{issue.summary}\n\n{issue.detailed_content[:1000]}"

                llm_judge_score = await self.detectors['LLM-Judge'].analyze_text(
                    text_to_judge,
                    context=f"주제: {topic}, 제목: {issue.title}"
                )
                individual_scores['LLM-Judge'] = llm_judge_score

            # 2. 가중치 동적 조정
            weights = self.threshold_manager.get_weights_for_confidence(current_confidence)

            # 3. 점수 통합
            combined_score = CombinedHallucinationScore(
                individual_scores=individual_scores,
                weights=weights,
                final_confidence=0  # __post_init__에서 계산됨
            )

            # 4. 분석 결과를 이슈에 추가
            setattr(issue, 'hallucination_analysis', combined_score)
            setattr(issue, 'hallucination_confidence', combined_score.final_confidence)

            # 5. 보고서 포함 여부 결정
            if self.threshold_manager.should_include_in_report(combined_score.final_confidence):
                # 신뢰도 레벨 추가
                confidence_level = self.threshold_manager.classify_confidence(
                    combined_score.final_confidence
                )
                setattr(issue, 'confidence_level', confidence_level)

                logger.debug(
                    f"이슈 '{issue.title[:30]}...' 검증 통과 "
                    f"(신뢰도: {combined_score.final_confidence:.2f}, "
                    f"등급: {confidence_level.value})"
                )

                return issue
            else:
                logger.warning(
                    f"이슈 '{issue.title[:30]}...' 제외됨 - "
                    f"신뢰도: {combined_score.final_confidence:.2f} < "
                    f"{self.threshold_manager.thresholds.min_confidence_threshold}"
                )

                # LLM Judge의 문제 영역 정보 로그
                if 'LLM-Judge' in individual_scores:
                    judge_score = individual_scores['LLM-Judge']
                    if hasattr(judge_score, 'problematic_areas') and judge_score.problematic_areas:
                        logger.warning(
                            f"LLM Judge가 발견한 문제점: "
                            f"{judge_score.problematic_areas[0]['issue']}"
                        )

                return None

        except Exception as e:
            logger.error(f"이슈 검증 중 오류: {e}")
            # 오류 발생 시에도 이슈를 포함하되 낮은 신뢰도 부여
            setattr(issue, 'hallucination_confidence', 0.0)
            setattr(issue, 'hallucination_error', str(e))
            return issue

    def update_threshold_config(self, new_config: ThresholdManager):
        """
        임계값 설정을 업데이트합니다.

        Args:
            new_config (ThresholdManager): 새로운 임계값 설정
        """
        self.threshold_manager = new_config
        logger.info(f"임계값 설정 업데이트 완료")

    def get_detector_status(self) -> Dict[str, bool]:
        """
        각 탐지기의 활성화 상태를 반환합니다.

        Returns:
            Dict[str, bool]: 탐지기별 활성화 상태
        """
        return {
            name: detector.is_initialized
            for name, detector in self.detectors.items()
        }

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

        # 현재 신뢰도에 따른 가중치 조정
        avg_confidence = sum(s.confidence for s in individual_scores.values()) / len(individual_scores)
        weights = self.threshold_manager.get_weights_for_confidence(avg_confidence)

        # 결과 통합
        return CombinedHallucinationScore(
            individual_scores=individual_scores,
            weights=weights,
            final_confidence=0  # __post_init__에서 계산됨
        )