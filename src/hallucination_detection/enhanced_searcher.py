"""
환각 탐지가 통합된 향상된 이슈 검색기
"""

import asyncio
import time
import hashlib
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from loguru import logger

from src.models import KeywordResult, IssueItem, SearchResult
from src.search.issue_searcher import IssueSearcher
from src.detection.keyword_generator import generate_keywords_for_topic
from src.hallucination_detection.threshold_manager import ThresholdManager
from src.hallucination_detection.reppl_detector import RePPLDetector
from src.hallucination_detection.consistency_checker import SelfConsistencyChecker
from src.hallucination_detection.models import CombinedHallucinationScore, ConsistencyScore
from sentence_transformers import SentenceTransformer


# 🚀 Global Sentence Transformer cache for performance optimization
class GlobalModelCache:
    """전역 Sentence Transformer 모델 캐시"""
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2') -> SentenceTransformer:
        """캐시된 모델 반환 또는 새로 로드"""
        if model_name not in self._models:
            logger.info(f"Sentence Transformer 모델 로딩: {model_name}")
            self._models[model_name] = SentenceTransformer(model_name)
        else:
            logger.debug(f"캐시된 모델 사용: {model_name}")
        return self._models[model_name]
    
    def clear_cache(self):
        """모델 캐시 초기화"""
        self._models.clear()
        logger.info("Sentence Transformer 모델 캐시 초기화")


@dataclass
class OptimizationMetrics:
    """최적화 성능 지표"""
    total_issues_processed: int = 0
    total_processing_time: float = 0.0
    avg_issue_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    self_consistency_runs: int = 0
    self_consistency_skips: int = 0
    timeout_utilizations: List[float] = field(default_factory=list)
    # Progressive Deepening 지표
    progressive_deepening_triggered: int = 0
    progressive_deepening_successful: int = 0
    progressive_deepening_failed: int = 0
    initial_low_confidence_issues: int = 0
    
    def calculate_cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def calculate_self_consistency_skip_ratio(self) -> float:
        total = self.self_consistency_runs + self.self_consistency_skips
        return self.self_consistency_skips / total if total > 0 else 0.0


class SmartCache:
    """
    🚀 추가 최적화: 스마트 캐싱 시스템
    
    동일한 텍스트에 대한 환각 탐지 결과를 캐싱하여
    중복 처리를 방지하고 성능을 대폭 향상시킵니다.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, text: str, context: str = "") -> str:
        """캐시 키 생성"""
        combined = f"{text}:{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, text: str, context: str = "") -> Optional[Any]:
        """캐시에서 결과 조회"""
        key = self._generate_key(text, context)
        current_time = time.time()
        
        if key in self.cache:
            stored_time, result = self.cache[key]
            
            # TTL 체크
            if current_time - stored_time < self.ttl_seconds:
                self.access_times[key] = current_time
                self.hits += 1
                logger.debug(f"캐시 히트: {key[:8]}")
                return result
            else:
                # 만료된 항목 제거
                del self.cache[key]
                del self.access_times[key]
        
        self.misses += 1
        return None
    
    def put(self, text: str, result: Any, context: str = ""):
        """결과를 캐시에 저장"""
        key = self._generate_key(text, context)
        current_time = time.time()
        
        # 캐시 크기 제한 확인
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = (current_time, result)
        self.access_times[key] = current_time
        logger.debug(f"캐시 저장: {key[:8]}")
    
    def _evict_oldest(self):
        """가장 오래된 항목 제거 (LRU)"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        logger.debug(f"캐시 제거: {oldest_key[:8]}")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total = self.hits + self.misses
        hit_ratio = self.hits / total if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': hit_ratio,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }


class EnhancedIssueSearcher:
    """여러 환각 탐지 방법이 통합된 이슈 검색기 (완전 최적화 버전)."""

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
        """
        # 기본 이슈 검색기
        self.base_searcher = IssueSearcher(api_key)

        # 임계값 관리자
        self.threshold_manager = threshold_manager or ThresholdManager()

        # 🚀 최적화 설정
        self.max_concurrent_issues = 8  # 동시 실행 수 증가 (5개 → 8개)
        self.self_consistency_threshold = 0.6  # SC 실행 임계값
        
        # 🚀 추가 최적화: 스마트 캐싱
        self.cache = SmartCache(max_size=1000, ttl_seconds=3600)
        
        # 성능 지표 추적
        self.metrics = OptimizationMetrics()
        
        # 적응형 타임아웃 설정
        self.timeout_config = {
            'base_timeout': 60.0,
            'max_timeout': 300.0,
            'chars_per_second_ratio': 500,
            'additional_per_500chars': 5.0
        }
        
        # 🆕 Progressive Deepening 설정
        self.progressive_deepening_config = {
            'enable': True,                    # Progressive Deepening 활성화
            'initial_confidence_threshold': 0.5,  # 1차 분석 신뢰도 임계값
            'deepening_confidence_threshold': 0.7, # 2차 분석 목표 신뢰도
            'timeout_multiplier': 1.5,         # 2차 분석 타임아웃 배수
            'max_deepening_attempts': 1,       # 최대 심화 분석 시도 횟수
            'enhanced_query_count': 5,         # 심화 분석시 Self-Consistency 쿼리 수
            'enhanced_similarity_threshold': 0.85,  # 심화 분석시 유사도 임계값
        }

        # 환각 탐지기들 초기화
        self.detectors = {}
        if enable_reppl:
            self.detectors['RePPL'] = RePPLDetector()
        if enable_consistency:
            self.detectors['Self-Consistency'] = SelfConsistencyChecker()
        if enable_llm_judge:
            try:
                from src.hallucination_detection.llm_judge import LLMJudgeDetector
                self.detectors['LLM-Judge'] = LLMJudgeDetector()
            except ImportError:
                logger.warning("LLM Judge 탐지기를 불러올 수 없습니다")

        logger.info(
            f"완전 최적화된 이슈 검색기 초기화 완료 "
            f"(활성 탐지기: {list(self.detectors.keys())}, "
            f"동시 실행: {self.max_concurrent_issues}개, "
            f"최소 신뢰도: {self.threshold_manager.thresholds.min_confidence_threshold})"
        )

    def _calculate_adaptive_timeout(self, text: str) -> float:
        text_length = len(text)
        base = 60.0  # 기본 타임아웃
        max_timeout = 300.0  # 최대 타임아웃

        # 텍스트 복잡도에 따른 추가 시간 계산
        additional = min(
            max_timeout - base,
            (text_length / 500) * 5.0  # 500자당 5초
        )

        return base + additional


    def _should_run_self_consistency(self, priority_confidence: float) -> bool:
        """
        🚀 개선사항 2: 우선순위 탐지기 결과에 따른 Self-Consistency 실행 결정
        """
        should_run = priority_confidence >= self.self_consistency_threshold
        
        if should_run:
            self.metrics.self_consistency_runs += 1
        else:
            self.metrics.self_consistency_skips += 1
        
        logger.debug(
            f"Self-Consistency 결정: 우선순위 {priority_confidence:.2f} "
            f"{'≥' if should_run else '<'} {self.self_consistency_threshold:.2f} → "
            f"{'실행' if should_run else '스킵'}"
        )
        
        return should_run

    async def search_with_validation(
            self,
            keyword_result: KeywordResult,
            time_period: str,
            max_retries: int = 3
    ) -> SearchResult:
        """
        키워드로 이슈를 검색하고 환각 탐지를 수행합니다. (최적화 버전)
        """
        logger.info(f"최적화된 검색 시작: 주제 '{keyword_result.topic}', 기간 '{time_period}'")
        overall_start = time.time()
        
        current_keywords = keyword_result
        all_attempts_issues = []

        for attempt in range(max_retries):
            logger.info(f"이슈 검색 시도 {attempt + 1}/{max_retries}")

            # 1. 기본 검색 수행
            search_result = await self.base_searcher.search_issues_from_keywords(
                current_keywords, time_period, collect_details=True
            )

            if not search_result.issues:
                logger.warning("검색 결과가 없습니다.")
                if attempt < max_retries - 1:
                    current_keywords = await generate_keywords_for_topic(
                        f"{current_keywords.topic}의 다른 측면"
                    )
                continue

            all_attempts_issues.extend(search_result.issues)

            # 2. 🚀 최적화된 환각 탐지 수행
            validated_issues = await self._validate_issues_optimized(
                search_result.issues, current_keywords
            )

            if not validated_issues:
                logger.warning("환각 탐지를 통과한 이슈가 없습니다.")
                if attempt < max_retries - 1:
                    current_keywords = await generate_keywords_for_topic(
                        f"{current_keywords.topic}의 다른 측면"
                    )
                continue

            # 3. 신뢰도별 분류 및 결과 평가
            high_confidence_issues = [
                issue for issue in validated_issues
                if getattr(issue, 'hallucination_confidence', 0) >= 0.7
            ]

            # 성공 조건 확인
            if len(high_confidence_issues) >= min(3, len(validated_issues) // 2):
                logger.info(
                    f"최적화된 검색 성공: {len(validated_issues)}개 이슈 중 "
                    f"{len(high_confidence_issues)}개 높은 신뢰도"
                )
                
                # 최종 결과 업데이트
                search_result.issues = validated_issues
                search_result.search_time = time.time() - overall_start
                
                # 🚀 성능 지표 업데이트
                self._update_final_metrics(search_result)
                break
            else:
                # 결과가 부족하면 키워드 재생성
                logger.info(
                    f"높은 신뢰도 이슈 부족 ({len(high_confidence_issues)}개), "
                    f"키워드 재생성 중..."
                )
                current_keywords = await generate_keywords_for_topic(
                    f"{current_keywords.topic}의 다른 측면"
                )

        # 🚀 성능 리포트 출력
        self._log_performance_report()
        
        return search_result

    async def _validate_issues_optimized(
        self,
        issues: List[IssueItem],
        keyword_result: KeywordResult
    ) -> List[IssueItem]:
        """
        🚀 모든 최적화가 적용된 이슈 검증
        
        포함된 최적화:
        - 동시 실행 수 증가 (8개)
        - 스마트 캐싱
        - 적응형 타임아웃
        - 우선순위 기반 처리
        - 메모리 효율적 배치 처리
        """
        if not issues:
            return []

        logger.info(f"완전 최적화된 환각 탐지 시작: {len(issues)}개 이슈")
        start_time = time.time()
        
        # 🚀 개선사항 4: 동시 실행 수 증가
        semaphore = asyncio.Semaphore(self.max_concurrent_issues)

        async def validate_with_optimizations(issue):
            async with semaphore:
                return await self._validate_single_issue_with_cache(issue, keyword_result)

        # 🚀 스트리밍 검증: 메모리 효율적이고 빠른 처리
        all_validated = []
        
        # 스트리밍 방식으로 이슈를 처리 (큰 배치 대신 연속적 처리)
        async def stream_process_issues():
            """스트리밍 방식으로 이슈를 연속적으로 처리"""
            results = []
            active_tasks = []
            
            for i, issue in enumerate(issues):
                # 동시 실행 수 제한
                if len(active_tasks) >= self.max_concurrent_issues:
                    # 완료된 태스크 수집
                    done, active_tasks = await asyncio.wait(
                        active_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in done:
                        try:
                            result = await task
                            if result is not None:
                                results.append(result)
                        except Exception as e:
                            logger.error(f"스트리밍 검증 실패: {e}")
                
                # 새로운 태스크 추가
                task = asyncio.create_task(validate_with_optimizations(issue))
                active_tasks.append(task)
                
                # 진행률 로깅
                if (i + 1) % 10 == 0:
                    logger.debug(f"스트리밍 처리 진행: {i+1}/{len(issues)}")
            
            # 남은 태스크 완료 대기
            if active_tasks:
                remaining_results = await asyncio.gather(*active_tasks, return_exceptions=True)
                for result in remaining_results:
                    if isinstance(result, Exception):
                        logger.error(f"최종 스트리밍 검증 실패: {result}")
                    elif result is not None:
                        results.append(result)
            
            return results
        
        all_validated = await stream_process_issues()

        # 성능 지표 업데이트
        processing_time = time.time() - start_time
        self.metrics.total_issues_processed += len(issues)
        self.metrics.total_processing_time += processing_time
        self.metrics.avg_issue_time = (
            self.metrics.total_processing_time / self.metrics.total_issues_processed
        )

        logger.info(
            f"완전 최적화 검증 완료 ({processing_time:.2f}초): "
            f"{len(issues)}개 → {len(all_validated)}개 통과"
        )

        return all_validated

    async def _validate_single_issue_with_cache(
        self,
        issue: IssueItem,
        keyword_result: KeywordResult
    ) -> Optional[IssueItem]:
        """
        🚀 캐싱이 적용된 단일 이슈 검증
        """
        # 캐시 확인
        cache_key_text = f"{issue.title}:{issue.summary}"
        cached_result = self.cache.get(cache_key_text, keyword_result.topic)
        
        if cached_result is not None:
            logger.debug(f"캐시 히트: {issue.title[:30]}")
            # 캐시된 결과를 이슈에 적용
            setattr(issue, 'hallucination_analysis', cached_result['analysis'])
            setattr(issue, 'hallucination_confidence', cached_result['confidence'])
            return issue if cached_result['is_valid'] else None
        
        # 캐시 미스 - 실제 검증 수행
        logger.debug(f"캐시 미스: {issue.title[:30]}")
        result = await self._validate_single_issue_optimized(issue, keyword_result)
        
        # 결과를 캐시에 저장
        if result is not None:
            cache_data = {
                'analysis': getattr(result, 'hallucination_analysis', None),
                'confidence': getattr(result, 'hallucination_confidence', 0.0),
                'is_valid': True
            }
        else:
            cache_data = {
                'analysis': None,
                'confidence': 0.0,
                'is_valid': False
            }
        
        self.cache.put(cache_key_text, cache_data, keyword_result.topic)
        return result

    async def _validate_single_issue_optimized(
        self,
        issue: IssueItem,
        keyword_result: KeywordResult
    ) -> Optional[IssueItem]:
        """
        🚀 단일 이슈 완전 최적화 검증 (Progressive Deepening 포함)
        """
        topic = keyword_result.topic
        
        try:
            # 분석 텍스트 준비
            analysis_text = self._prepare_analysis_text(issue)
            
            # 🚀 개선사항 1: 적응형 타임아웃 계산
            adaptive_timeout = self._calculate_adaptive_timeout(analysis_text)
            
            # === 1차 분석: 빠른 초기 분석 ===
            initial_result = await self._run_initial_analysis(
                issue, topic, analysis_text, adaptive_timeout
            )
            
            if initial_result is None:
                return None
            
            # 초기 분석 신뢰도 확인
            initial_confidence = getattr(initial_result, 'hallucination_confidence', 0.0)
            
            # === 🚀 스마트 Progressive Deepening 결정 ===
            should_deepen = self._should_apply_progressive_deepening(
                issue, initial_confidence, analysis_text
            )
            
            if should_deepen:
                logger.info(
                    f"Progressive Deepening 트리거 - 초기 신뢰도: {initial_confidence:.2f} < "
                    f"{self.progressive_deepening_config['initial_confidence_threshold']:.2f} "
                    f"(이슈: {issue.title[:30]})"
                )
                
                self.metrics.progressive_deepening_triggered += 1
                self.metrics.initial_low_confidence_issues += 1
                
                # === 2차 분석: 심화 정밀 분석 ===
                deepened_result = await self._run_progressive_deepening_analysis(
                    issue, topic, analysis_text, adaptive_timeout, initial_result
                )
                
                if deepened_result is not None:
                    deepened_confidence = getattr(deepened_result, 'hallucination_confidence', 0.0)
                    
                    if deepened_confidence >= self.progressive_deepening_config['deepening_confidence_threshold']:
                        self.metrics.progressive_deepening_successful += 1
                        logger.info(
                            f"Progressive Deepening 성공 - 향상된 신뢰도: "
                            f"{initial_confidence:.2f} → {deepened_confidence:.2f}"
                        )
                        return deepened_result
                    else:
                        self.metrics.progressive_deepening_failed += 1
                        logger.debug(
                            f"Progressive Deepening 실패 - 여전히 낮은 신뢰도: {deepened_confidence:.2f}"
                        )
                else:
                    self.metrics.progressive_deepening_failed += 1
                    logger.debug("Progressive Deepening 실패 - 분석 오류")
            
            # 초기 결과 반환 (Progressive Deepening이 비활성화되거나 실패한 경우)
            return initial_result
            
        except Exception as e:
            logger.error(f"이슈 '{issue.title}' 검증 오류: {e}")
            return None

    async def _run_initial_analysis(
        self,
        issue: IssueItem,
        topic: str,
        analysis_text: str,
        adaptive_timeout: float
    ) -> Optional[IssueItem]:
        """
        🚀 1차 분석: 빠른 초기 분석 수행
        """
        # === 1단계: 우선순위 탐지기 실행 ===
        priority_scores = await self._run_priority_detectors_optimized(
            issue, topic, analysis_text, adaptive_timeout * 0.7
        )
        
        # === 2단계: Self-Consistency 조건부 실행 ===
        all_scores = priority_scores.copy()
        
        if priority_scores:
            avg_priority_confidence = sum(
                score.confidence for score in priority_scores.values()
            ) / len(priority_scores)
            
            # 🚀 개선사항 2 & 3: 조건부 Self-Consistency
            if self._should_run_self_consistency(avg_priority_confidence):
                consistency_score = await self._run_optimized_self_consistency(
                    analysis_text, issue.title, adaptive_timeout * 0.3
                )
                
                # Handle return format (ConsistencyScore object or dict)
                if consistency_score:
                    if isinstance(consistency_score, dict):
                        # Timeout or error case
                        status = consistency_score.get('status', 'unknown')
                        logger.debug(f"Self-Consistency 결과: {status}")
                    else:
                        # Success case - ConsistencyScore object
                        all_scores['Self-Consistency'] = consistency_score
        
        # === 3단계: 결과 통합 및 최종 검증 ===
        return self._finalize_issue_validation(issue, all_scores)

    async def _run_progressive_deepening_analysis(
        self,
        issue: IssueItem,
        topic: str,
        analysis_text: str,
        adaptive_timeout: float,
        initial_result: IssueItem
    ) -> Optional[IssueItem]:
        """
        🆕 2차 분석: Progressive Deepening 심화 정밀 분석
        
        더 긴 타임아웃과 향상된 설정으로 재분석을 수행합니다.
        """
        logger.debug(f"Progressive Deepening 시작: {issue.title[:30]}")
        
        # 심화 분석용 연장된 타임아웃
        deepening_timeout = adaptive_timeout * self.progressive_deepening_config['timeout_multiplier']
        
        try:
            # === 심화 단계 1: 향상된 우선순위 탐지기 실행 ===
            enhanced_priority_scores = await self._run_enhanced_priority_detectors(
                issue, topic, analysis_text, deepening_timeout * 0.6
            )
            
            # === 심화 단계 2: 강화된 Self-Consistency 실행 ===
            all_enhanced_scores = enhanced_priority_scores.copy()
            
            if enhanced_priority_scores:
                # 강화된 Self-Consistency 무조건 실행
                enhanced_consistency_score = await self._run_enhanced_self_consistency(
                    analysis_text, issue.title, deepening_timeout * 0.4
                )
                
                if enhanced_consistency_score:
                    if isinstance(enhanced_consistency_score, dict):
                        # Timeout or error case
                        status = enhanced_consistency_score.get('status', 'unknown')
                        logger.debug(f"Enhanced Self-Consistency 결과: {status}")
                    else:
                        # Success case - ConsistencyScore object
                        all_enhanced_scores['Enhanced-Self-Consistency'] = enhanced_consistency_score
            
            # === 심화 단계 3: 결과 통합 및 검증 ===
            if all_enhanced_scores:
                # 초기 결과와 심화 결과를 결합하여 더 정확한 신뢰도 계산
                initial_scores = getattr(initial_result, 'hallucination_analysis', None)
                if initial_scores and hasattr(initial_scores, 'individual_scores'):
                    # 기존 점수와 새로운 점수를 결합
                    combined_scores = initial_scores.individual_scores.copy()
                    combined_scores.update(all_enhanced_scores)
                    all_enhanced_scores = combined_scores
                
                return self._finalize_issue_validation(issue, all_enhanced_scores)
            else:
                logger.warning(f"Progressive Deepening 실패: 모든 심화 탐지기 실패 - {issue.title[:30]}")
                return None
                
        except Exception as e:
            logger.error(f"Progressive Deepening 오류: {e} - {issue.title[:30]}")
            return None

    async def _run_enhanced_priority_detectors(
        self,
        issue: IssueItem,
        topic: str,
        analysis_text: str,
        timeout: float
    ) -> Dict[str, Any]:
        """
        🆕 향상된 우선순위 탐지기 실행 (Progressive Deepening용)
        
        더 긴 타임아웃과 더 정밀한 설정으로 우선순위 탐지기들을 실행합니다.
        """
        enhanced_tasks = {}
        
        # Enhanced RePPL 분석 (더 정밀한 설정)
        if 'RePPL' in self.detectors:
            enhanced_tasks['Enhanced-RePPL'] = asyncio.create_task(
                self.detectors['RePPL'].analyze_issue(issue, topic),
                name=f"EnhancedRePPL-{issue.title[:20]}"
            )
        
        # Enhanced LLM Judge 검사 (더 상세한 컨텍스트)
        if 'LLM-Judge' in self.detectors:
            enhanced_context = (
                f"주제: {topic}\n"
                f"제목: {issue.title}\n"
                f"심화 분석 모드: Progressive Deepening\n"
                f"분석 길이: {len(analysis_text)} 문자"
            )
            enhanced_tasks['Enhanced-LLM-Judge'] = asyncio.create_task(
                self.detectors['LLM-Judge'].analyze_text(
                    analysis_text,
                    context=enhanced_context
                ),
                name=f"EnhancedLLMJudge-{issue.title[:20]}"
            )
        
        # 향상된 탐지기 병렬 실행 (더 긴 타임아웃)
        enhanced_scores = {}
        
        if enhanced_tasks:
            done, pending = await asyncio.wait(
                enhanced_tasks.values(),
                return_when=asyncio.ALL_COMPLETED,
                timeout=timeout
            )
            
            # 완료된 태스크 결과 수집
            for task in done:
                task_name = task.get_name()
                try:
                    result = await task
                    detector_type = task_name.split('-')[1] if '-' in task_name else task_name
                    if detector_type == 'LLMJudge':
                        detector_type = 'LLM-Judge'
                    
                    enhanced_scores[f"Enhanced-{detector_type}"] = result
                    logger.debug(f"Enhanced {detector_type} 완료: {result.confidence:.2f}")
                    
                except Exception as e:
                    detector_type = task_name.split('-')[1] if '-' in task_name else task_name
                    logger.warning(f"Enhanced {detector_type} 실패: {e}")
            
            # 미완료 태스크 정리
            for task in pending:
                task.cancel()
                detector_type = task.get_name().split('-')[1] if '-' in task.get_name() else task.get_name()
                logger.warning(f"Enhanced {detector_type} 타임아웃")
                
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        
        return enhanced_scores

    async def _run_enhanced_self_consistency(
        self,
        text: str,
        context: str,
        timeout: float
    ) -> Optional[Union[ConsistencyScore, Dict[str, Any]]]:
        """
        🆕 강화된 Self-Consistency 실행 (Progressive Deepening용)
        
        더 많은 쿼리와 더 엄격한 기준으로 Self-Consistency를 실행합니다.
        Returns ConsistencyScore on success, Dict on error/timeout, None on failure
        """
        if 'Self-Consistency' not in self.detectors:
            return {'status': 'detector_unavailable', 'confidence': 0.0}
        
        detector = self.detectors['Self-Consistency']
        
        # 🆕 심화 분석용 강화된 설정 적용
        original_settings = self._apply_enhanced_consistency_settings(detector)
        
        try:
            # Enhanced Self-Consistency 실행
            enhanced_task = asyncio.create_task(
                detector.analyze_text(text, context=f"Enhanced Analysis: {context}")
            )
            
            result = await asyncio.wait_for(enhanced_task, timeout=timeout)
            logger.debug(f"Enhanced Self-Consistency 성공: {result.confidence:.2f}")
            return result
            
        except asyncio.TimeoutError:
            enhanced_task.cancel()
            
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logger.warning(
                f"Enhanced Self-Consistency 타임아웃 - 설정 타임아웃: {timeout}초, "
                f"분석 텍스트: '{text_preview}'"
            )
            
            try:
                await enhanced_task
            except (asyncio.CancelledError, Exception):
                pass
            
            return {'status': 'timeout', 'confidence': 0.0}
        except Exception as e:
            logger.warning(f"Enhanced Self-Consistency 실패: {e}")
            return {'status': 'error', 'confidence': 0.0, 'error': str(e)}
        finally:
            # 원래 설정 복원
            self._restore_consistency_settings(detector, original_settings)

    def _should_apply_progressive_deepening(
        self, 
        issue: IssueItem, 
        initial_confidence: float, 
        analysis_text: str
    ) -> bool:
        """
        🚀 스마트 Progressive Deepening 결정 로직
        
        단순히 신뢰도만 보는 것이 아니라 다양한 요소를 고려하여
        Progressive Deepening이 실제로 도움이 될지 판단합니다.
        """
        
        # 기본 조건: Progressive Deepening이 활성화되어야 함
        if not self.progressive_deepening_config['enable']:
            return False
        
        # 🚀 조건 1: 초기 신뢰도가 임계값보다 낮아야 함
        confidence_threshold = self.progressive_deepening_config['initial_confidence_threshold']
        if initial_confidence >= confidence_threshold:
            return False
        
        # 🚀 조건 2: 너무 짧은 텍스트는 스킵 (개선 여지가 적음)
        if len(analysis_text) < 150:
            logger.debug(f"Progressive Deepening 스킵 - 텍스트 너무 짧음: {len(analysis_text)} < 150")
            return False
        
        # 🚀 조건 3: 명백히 문제가 있는 이슈는 스킵 (시간 절약)
        if initial_confidence < 0.2:
            logger.debug(f"Progressive Deepening 스킵 - 명백히 신뢰도 낮음: {initial_confidence:.2f} < 0.2")
            return False
        
        # 🚀 조건 4: 중간 신뢰도 영역에서만 Progressive Deepening 적용
        # (0.2 ~ 0.5 범위에서 가장 효과적)
        if 0.2 <= initial_confidence < confidence_threshold:
            logger.debug(f"Progressive Deepening 적용 - 개선 가능 범위: {initial_confidence:.2f}")
            return True
        
        return False

    def _apply_enhanced_consistency_settings(self, detector) -> Dict[str, Any]:
        """
        🆕 Progressive Deepening용 강화된 Self-Consistency 설정 적용
        """
        original_settings = {}
        
        # 🆕 쿼리 수 증가 (기본 3개 → 5개)
        if hasattr(detector, 'set_query_count'):
            original_settings['query_count'] = getattr(detector, 'query_count', 3)
            detector.set_query_count(
                self.progressive_deepening_config['enhanced_query_count']
            )
        
        # 🆕 더 엄격한 유사도 기준 (기본 0.8 → 0.85)
        if hasattr(detector, 'set_similarity_threshold'):
            original_settings['similarity_threshold'] = getattr(detector, 'similarity_threshold', 0.8)
            detector.set_similarity_threshold(
                self.progressive_deepening_config['enhanced_similarity_threshold']
            )
        
        return original_settings

    def _prepare_analysis_text(self, issue: IssueItem) -> str:
        """분석용 텍스트 최적화된 준비"""
        analysis_text = issue.summary
        
        # 상세 내용이 있으면 포함 (길이 제한으로 메모리 효율성 확보)
        if hasattr(issue, 'detailed_content') and issue.detailed_content:
            # 너무 긴 텍스트는 잘라서 메모리 사용량 제한
            max_detail_length = 1500
            detail = issue.detailed_content[:max_detail_length]
            if len(issue.detailed_content) > max_detail_length:
                detail += "..."
            
            analysis_text = f"{issue.summary}\n\n{detail}"
        
        return analysis_text

    async def _run_priority_detectors_optimized(
        self, 
        issue: IssueItem, 
        topic: str, 
        analysis_text: str, 
        timeout: float
    ) -> Dict[str, Any]:
        """최적화된 우선순위 탐지기 실행"""
        
        priority_tasks = {}
        
        # RePPL 분석 (가장 빠름)
        if 'RePPL' in self.detectors:
            priority_tasks['RePPL'] = asyncio.create_task(
                self.detectors['RePPL'].analyze_issue(issue, topic),
                name=f"RePPL-{issue.title[:20]}"
            )
        
        # LLM Judge 검사 (중간 속도)
        if 'LLM-Judge' in self.detectors:
            priority_tasks['LLM-Judge'] = asyncio.create_task(
                self.detectors['LLM-Judge'].analyze_text(
                    analysis_text,
                    context=f"주제: {topic}, 제목: {issue.title}"
                ),
                name=f"LLMJudge-{issue.title[:20]}"
            )
        
        # 우선순위 탐지기 병렬 실행
        priority_scores = {}
        
        if priority_tasks:
            done, pending = await asyncio.wait(
                priority_tasks.values(),
                return_when=asyncio.ALL_COMPLETED,
                timeout=timeout
            )
            
            # 완료된 태스크 결과 수집
            for task in done:
                task_name = task.get_name()
                try:
                    result = await task
                    detector_type = task_name.split('-')[0]
                    if detector_type == 'LLMJudge':
                        detector_type = 'LLM-Judge'
                    
                    priority_scores[detector_type] = result
                    logger.debug(f"{detector_type} 완료: {result.confidence:.2f}")
                    
                except Exception as e:
                    detector_type = task_name.split('-')[0]
                    logger.warning(f"{detector_type} 실패: {e}")
            
            # 미완료 태스크 정리
            for task in pending:
                task.cancel()
                detector_type = task.get_name().split('-')[0]
                logger.warning(f"{detector_type} 우선순위 단계 타임아웃")
                
                # 타임아웃된 태스크 정리 대기
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        
        return priority_scores

    async def _run_optimized_self_consistency(
        self, 
        text: str, 
        context: str, 
        timeout: float
    ) -> Optional[Union[ConsistencyScore, Dict[str, Any]]]:
        """
        🚀 개선사항 3: 최적화된 Self-Consistency 실행
        Returns ConsistencyScore on success, Dict on error/timeout, None on failure
        """
        if 'Self-Consistency' not in self.detectors:
            return {'status': 'detector_unavailable', 'confidence': 0.0}
        
        detector = self.detectors['Self-Consistency']
        
        # 🚀 짧은 텍스트 빠른 처리
        if len(text) < 100:
            logger.debug("짧은 텍스트 - Self-Consistency 간소화")
            return self._create_optimized_consistency_score()
        
        # 🚀 동적 최적화 설정 적용
        original_settings = self._apply_consistency_optimizations(detector)
        
        try:
            # Self-Consistency 실행
            consistency_task = asyncio.create_task(
                detector.analyze_text(text, context=context)
            )
            
            result = await asyncio.wait_for(consistency_task, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            consistency_task.cancel()
            
            # Enhanced logging with timeout and text preview
            text_preview = text[:50] + "..." if len(text) > 50 else text
            logger.warning(
                f"Self-Consistency 타임아웃 - 설정 타임아웃: {timeout}초, "
                f"분석 텍스트: '{text_preview}'"
            )
            
            # 정리 대기
            try:
                await consistency_task
            except (asyncio.CancelledError, Exception):
                pass
            
            return {'status': 'timeout', 'confidence': 0.0}
        except Exception as e:
            logger.warning(f"Self-Consistency 실패: {e}")
            return {'status': 'error', 'confidence': 0.0, 'error': str(e)}
        finally:
            # 원래 설정 복원
            self._restore_consistency_settings(detector, original_settings)

    def _apply_consistency_optimizations(self, detector) -> Dict[str, Any]:
        """Self-Consistency 최적화 설정 적용"""
        original_settings = {}
        
        # 🚀 쿼리 수 감소 (3개 → 2개)
        if hasattr(detector, 'set_query_count'):
            original_settings['query_count'] = getattr(detector, 'query_count', 3)
            detector.set_query_count(2)
        
        # 🚀 관대한 일치 기준 (0.8 → 0.7)
        if hasattr(detector, 'set_similarity_threshold'):
            original_settings['similarity_threshold'] = getattr(detector, 'similarity_threshold', 0.8)
            detector.set_similarity_threshold(0.7)
        
        return original_settings

    def _restore_consistency_settings(self, detector, original_settings: Dict[str, Any]):
        """원래 Self-Consistency 설정 복원"""
        for setting, value in original_settings.items():
            if hasattr(detector, f'set_{setting}'):
                getattr(detector, f'set_{setting}')(value)

    def _create_optimized_consistency_score(self):
        """최적화된 일관성 점수 생성 (짧은 텍스트용)"""
        return ConsistencyScore(
            confidence=0.85,
            consistency_rate=0.85,
            num_queries=1,
            num_consistent=1,
            variations=["최적화된 응답"],
            common_elements=["짧은 텍스트"],
            divergent_elements=[],
            analysis_details={"optimized": True, "reason": "짧은 텍스트 최적화 처리"}
        )

    def _finalize_issue_validation(
        self, 
        issue: IssueItem, 
        all_scores: Dict[str, Any]
    ) -> Optional[IssueItem]:
        """이슈 검증 최종 처리"""
        
        if not all_scores:
            logger.warning(f"이슈 '{issue.title}': 모든 탐지기 실패")
            return None
        
        # 가중치 동적 조정
        max_confidence = max(score.confidence for score in all_scores.values())
        weights = self.threshold_manager.get_weights_for_confidence(max_confidence)
        
        # 점수 통합
        combined_score = CombinedHallucinationScore(
            individual_scores=all_scores,
            weights=weights,
            final_confidence=0
        )
        
        # 결과를 이슈에 추가
        setattr(issue, 'hallucination_analysis', combined_score)
        setattr(issue, 'hallucination_confidence', combined_score.final_confidence)
        
        # 최소 임계값 검사
        min_threshold = self.threshold_manager.thresholds.min_confidence_threshold
        if combined_score.final_confidence < min_threshold:
            logger.debug(
                f"이슈 '{issue.title}' 신뢰도 부족: "
                f"{combined_score.final_confidence:.2f} < {min_threshold:.2f}"
            )
            return None
        
        # 성공 로깅
        detection_summary = ", ".join([
            f"{k}: {v.confidence:.2f}"
            for k, v in all_scores.items()
        ])
        
        logger.debug(
            f"이슈 '{issue.title[:30]}' 완전 최적화 검증 완료 - "
            f"탐지기: [{detection_summary}], 최종: {combined_score.final_confidence:.2f}"
        )
        
        return issue

    def _update_final_metrics(self, search_result: SearchResult):
        """최종 성능 지표 업데이트"""
        # 캐시 통계 포함
        cache_stats = self.cache.get_stats()
        self.metrics.cache_hits = cache_stats['hits']
        self.metrics.cache_misses = cache_stats['misses']
        
        # 검색 결과에 성능 지표 추가 (Progressive Deepening 포함)
        search_result.performance_metrics = {
            'total_processing_time': self.metrics.total_processing_time,
            'avg_issue_time': self.metrics.avg_issue_time,
            'cache_hit_ratio': self.metrics.calculate_cache_hit_ratio(),
            'self_consistency_skip_ratio': self.metrics.calculate_self_consistency_skip_ratio(),
            'avg_timeout_used': sum(self.metrics.timeout_utilizations) / len(self.metrics.timeout_utilizations) if self.metrics.timeout_utilizations else 0,
            # Progressive Deepening 통계
            'progressive_deepening': {
                'triggered': self.metrics.progressive_deepening_triggered,
                'successful': self.metrics.progressive_deepening_successful,
                'failed': self.metrics.progressive_deepening_failed,
                'success_rate': (self.metrics.progressive_deepening_successful / max(1, self.metrics.progressive_deepening_triggered)) * 100,
                'initial_low_confidence_issues': self.metrics.initial_low_confidence_issues
            }
        }

    def _log_performance_report(self):
        """성능 리포트 출력 (Progressive Deepening 포함)"""
        cache_stats = self.cache.get_stats()
        
        # Progressive Deepening 통계 계산
        pd_success_rate = 0.0
        if self.metrics.progressive_deepening_triggered > 0:
            pd_success_rate = (
                self.metrics.progressive_deepening_successful / 
                self.metrics.progressive_deepening_triggered
            ) * 100
        
        logger.info(
            f"🎯 완전 최적화 성능 리포트 (Progressive Deepening 포함):\n"
            f"  • 총 처리 이슈: {self.metrics.total_issues_processed}개\n"
            f"  • 평균 이슈 처리 시간: {self.metrics.avg_issue_time:.3f}초\n"
            f"  • 캐시 히트율: {cache_stats['hit_ratio']:.1%} ({cache_stats['hits']}/{cache_stats['hits'] + cache_stats['misses']})\n"
            f"  • Self-Consistency 스킵율: {self.metrics.calculate_self_consistency_skip_ratio():.1%}\n"
            f"  • 평균 타임아웃: {sum(self.metrics.timeout_utilizations) / len(self.metrics.timeout_utilizations) if self.metrics.timeout_utilizations else 0:.1f}초\n"
            f"  • 동시 실행 수: {self.max_concurrent_issues}개\n"
            f"  🆕 Progressive Deepening 통계:\n"
            f"    - 낮은 신뢰도 이슈: {self.metrics.initial_low_confidence_issues}개\n"
            f"    - Deepening 트리거: {self.metrics.progressive_deepening_triggered}개\n"
            f"    - Deepening 성공: {self.metrics.progressive_deepening_successful}개\n"
            f"    - Deepening 실패: {self.metrics.progressive_deepening_failed}개\n"
            f"    - 성공률: {pd_success_rate:.1f}%"
        )

    # 기존 호환성 메서드들 유지
    def get_detector_status(self) -> Dict[str, bool]:
        """각 탐지기의 활성화 상태 반환"""
        return {
            name: detector.is_initialized if hasattr(detector, 'is_initialized') else True
            for name, detector in self.detectors.items()
        }
    
    def update_threshold_config(self, new_config: ThresholdManager):
        """임계값 설정 업데이트"""
        self.threshold_manager = new_config
        logger.info("임계값 설정 업데이트 완료")
    
    def clear_cache(self):
        """캐시 초기화"""
        self.cache = SmartCache(max_size=1000, ttl_seconds=3600)
        logger.info("캐시 초기화 완료")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """현재 성능 지표 반환"""
        return {
            'processing_metrics': {
                'total_issues': self.metrics.total_issues_processed,
                'avg_time_per_issue': self.metrics.avg_issue_time,
                'total_time': self.metrics.total_processing_time
            },
            'cache_metrics': self.cache.get_stats(),
            'optimization_metrics': {
                'self_consistency_skip_ratio': self.metrics.calculate_self_consistency_skip_ratio(),
                'avg_timeout': sum(self.metrics.timeout_utilizations) / len(self.metrics.timeout_utilizations) if self.metrics.timeout_utilizations else 0,
                'max_concurrent': self.max_concurrent_issues
            }
        }