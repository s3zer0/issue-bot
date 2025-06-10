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
            # ✨ [수정] topic 대신 keyword_result 객체 전체를 전달
            validated_issues = await self._validate_issues(
                search_result.issues,
                current_keywords
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
            keyword_result: KeywordResult
    ) -> List[IssueItem]:
        """
        이슈 목록에 대해 환각 탐지를 수행합니다. (세마포어를 통한 동시 실행 제한 추가)
        """
        if not issues:
            return []

        logger.info(f"환각 탐지 시작: {len(issues)}개 이슈 병렬 처리")
        start_time = asyncio.get_event_loop().time()

        # 세마포어를 사용하여 동시 실행 수 제한 (API 요금 제한 고려)
        semaphore = asyncio.Semaphore(5)  # 최대 5개 이슈 동시 처리

        async def validate_with_semaphore(issue):
            async with semaphore:
                return await self._validate_single_issue(issue, keyword_result)

        # 모든 이슈에 대해 병렬 검증 실행
        validation_tasks = [
            validate_with_semaphore(issue)
            for issue in issues
        ]

        # 결과 수집
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # 성공한 결과만 필터링
        validated_issues = []
        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                logger.error(f"이슈 {i} 검증 중 예외 발생: {result}")
            elif result is not None:
                validated_issues.append(result)

        # 성능 및 통계 로깅
        total_time = asyncio.get_event_loop().time() - start_time

        if validated_issues:
            high, moderate, low = self.threshold_manager.filter_issues_by_confidence(
                validated_issues
            )

            logger.info(
                f"환각 탐지 완료 ({total_time:.2f}초): {len(issues)}개 중 {len(validated_issues)}개 통과 "
                f"(높음: {len(high)}, 보통: {len(moderate)}, 낮음: {len(low)})"
            )

            # 탐지기별 성능 통계 로깅
            detector_stats = {}
            for issue in validated_issues:
                analysis = getattr(issue, 'hallucination_analysis', None)
                if analysis and hasattr(analysis, 'individual_scores'):
                    for detector, score in analysis.individual_scores.items():
                        if detector not in detector_stats:
                            detector_stats[detector] = []
                        detector_stats[detector].append(score.confidence)

            # 평균 성능 로깅
            for detector, scores in detector_stats.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    logger.info(f"{detector} 평균 신뢰도: {avg_score:.2f} ({len(scores)}개 분석)")
        else:
            logger.warning("검증을 통과한 이슈가 없습니다")

        return validated_issues

    async def _validate_single_issue(
            self,
            issue: IssueItem,
            keyword_result: KeywordResult
    ) -> Optional[IssueItem]:
        """
        단일 이슈에 대해 환각 탐지를 병렬로 수행합니다. (개선된 버전)

        Args:
            issue (IssueItem): 검증할 이슈
            keyword_result (KeywordResult): 주제 및 동적 신뢰 출처 리스트를 포함한 키워드 결과 객체

        Returns:
            Optional[IssueItem]: 검증 통과 시 이슈, 실패 시 None
        """
        topic = keyword_result.topic

        try:
            # 1. 모든 탐지기를 병렬로 실행하기 위한 태스크 생성
            detection_tasks = {}

            # RePPL 분석 태스크
            if 'RePPL' in self.detectors:
                detection_tasks['RePPL'] = asyncio.create_task(
                    self.detectors['RePPL'].analyze_issue(issue, topic),
                    name=f"RePPL-{issue.title[:20]}"
                )

            # 자기 일관성 검사 태스크
            if 'Self-Consistency' in self.detectors:
                detection_tasks['Self-Consistency'] = asyncio.create_task(
                    self.detectors['Self-Consistency'].analyze_text(
                        issue.summary,
                        context=issue.title
                    ),
                    name=f"Consistency-{issue.title[:20]}"
                )

            # LLM Judge 검사 태스크
            if 'LLM-Judge' in self.detectors:
                text_to_judge = issue.summary
                if issue.detailed_content:
                    text_to_judge = f"{issue.summary}\n\n{issue.detailed_content[:1000]}"

                detection_tasks['LLM-Judge'] = asyncio.create_task(
                    self.detectors['LLM-Judge'].analyze_text(
                        text_to_judge,
                        context=f"주제: {topic}, 제목: {issue.title}"
                    ),
                    name=f"LLMJudge-{issue.title[:20]}"
                )

            # 2. 모든 탐지기를 병렬로 실행하고 결과 수집
            individual_scores = {}

            if detection_tasks:
                # 모든 태스크를 병렬로 실행 (30초 타임아웃)
                done, pending = await asyncio.wait(
                    detection_tasks.values(),
                    return_when=asyncio.ALL_COMPLETED,
                    timeout=30.0
                )

                # 완료된 태스크 결과 수집
                for task in done:
                    task_name = task.get_name()
                    try:
                        result = await task
                        # 태스크 이름에서 탐지기 유형 추출
                        detector_type = task_name.split('-')[0]
                        if detector_type == 'Consistency':
                            detector_type = 'Self-Consistency'
                        elif detector_type == 'LLMJudge':
                            detector_type = 'LLM-Judge'

                        individual_scores[detector_type] = result
                        logger.debug(f"{detector_type} 탐지 완료: {result.confidence:.2f}")

                    except Exception as e:
                        detector_type = task_name.split('-')[0]
                        logger.warning(f"{detector_type} 탐지 실패: {e}")

                # 미완료 태스크 취소
                for task in pending:
                    task.cancel()
                    detector_type = task.get_name().split('-')[0]
                    logger.warning(f"{detector_type} 탐지 타임아웃으로 취소됨")

            # 3. 결과가 없으면 None 반환
            if not individual_scores:
                logger.warning(f"이슈 '{issue.title}': 모든 환각 탐지기 실패")
                return None

            # 4. 가중치 동적 조정 (최고 신뢰도 기준)
            max_confidence = max(score.confidence for score in individual_scores.values())
            weights = self.threshold_manager.get_weights_for_confidence(max_confidence)

            # 5. 점수 통합
            combined_score = CombinedHallucinationScore(
                individual_scores=individual_scores,
                weights=weights,
                final_confidence=0  # __post_init__에서 계산됨
            )

            # 6. 분석 결과를 이슈에 추가
            setattr(issue, 'hallucination_analysis', combined_score)
            setattr(issue, 'hallucination_confidence', combined_score.final_confidence)

            # 7. 최소 임계값 검사
            if combined_score.final_confidence < self.threshold_manager.thresholds.min_confidence_threshold:
                logger.debug(
                    f"이슈 '{issue.title}' 신뢰도 부족: "
                    f"{combined_score.final_confidence:.2f} < "
                    f"{self.threshold_manager.thresholds.min_confidence_threshold:.2f}"
                )
                return None

            # 8. 성능 메트릭 로깅
            detection_summary = ", ".join([
                f"{k}: {v.confidence:.2f}"
                for k, v in individual_scores.items()
            ])

            logger.debug(
                f"이슈 '{issue.title}' 검증 완료 - "
                f"개별 점수: [{detection_summary}], "
                f"최종 신뢰도: {combined_score.final_confidence:.2f}"
            )

            return issue

        except asyncio.CancelledError:
            logger.warning(f"이슈 '{issue.title}' 검증이 취소되었습니다")
            return None
        except Exception as e:
            logger.error(f"이슈 '{issue.title}' 검증 중 오류 발생: {e}")
            return None

    # 추가로 _validate_issues 메서드도 다음과 같이 개선할 수 있습니다:

    async def _validate_issues(
            self,
            issues: List[IssueItem],
            keyword_result: KeywordResult
    ) -> List[IssueItem]:
        """
        이슈 목록에 대해 환각 탐지를 수행합니다. (세마포어를 통한 동시 실행 제한 추가)

        Args:
            issues (List[IssueItem]): 검증할 이슈 목록
            keyword_result (KeywordResult): 주제 및 신뢰 출처를 포함한 키워드 결과

        Returns:
            List[IssueItem]: 검증된 이슈 목록
        """
        if not issues:
            return []

        logger.info(f"환각 탐지 시작: {len(issues)}개 이슈 병렬 처리")

        # 세마포어를 사용하여 동시 실행 수 제한 (API 요금 제한 고려)
        semaphore = asyncio.Semaphore(5)  # 최대 5개 이슈 동시 처리

        async def validate_with_semaphore(issue):
            async with semaphore:
                return await self._validate_single_issue(issue, keyword_result)

        # 모든 이슈에 대해 병렬 검증 실행
        validation_tasks = [
            validate_with_semaphore(issue)
            for issue in issues
        ]

        # 결과 수집
        validation_results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # 성공한 결과만 필터링
        validated_issues = []
        for i, result in enumerate(validation_results):
            if isinstance(result, Exception):
                logger.error(f"이슈 {i} 검증 중 예외 발생: {result}")
            elif result is not None:
                validated_issues.append(result)

        # 신뢰도별 분류 로깅
        if validated_issues:
            high, moderate, low = self.threshold_manager.filter_issues_by_confidence(
                validated_issues
            )

            logger.info(
                f"환각 탐지 완료: {len(issues)}개 중 {len(validated_issues)}개 통과 "
                f"(높음: {len(high)}, 보통: {len(moderate)}, 낮음: {len(low)})"
            )

            # 탐지기별 성능 통계 로깅
            detector_stats = {}
            for issue in validated_issues:
                analysis = getattr(issue, 'hallucination_analysis', None)
                if analysis and hasattr(analysis, 'individual_scores'):
                    for detector, score in analysis.individual_scores.items():
                        if detector not in detector_stats:
                            detector_stats[detector] = []
                        detector_stats[detector].append(score.confidence)

            # 평균 성능 로깅
            for detector, scores in detector_stats.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    logger.info(f"{detector} 평균 신뢰도: {avg_score:.2f} ({len(scores)}개 분석)")
        else:
            logger.warning("검증을 통과한 이슈가 없습니다")

        return validated_issues

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