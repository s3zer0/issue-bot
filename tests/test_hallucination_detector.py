import pytest
import numpy as np
import math
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import os

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 테스트 대상 모듈 임포트 - 새로운 경로 사용
from src.hallucination_detection.reppl_detector import RePPLDetector
from src.hallucination_detection.enhanced_searcher import EnhancedIssueSearcher
from src.models import KeywordResult, IssueItem, SearchResult


# --- 픽스처(Fixtures) ---
@pytest.fixture
def detector():
    """테스트용 RePPLDetector 인스턴스 픽스처"""
    with patch('src.hallucination_detection.reppl_detector.SentenceTransformer'), \
            patch('src.hallucination_detection.reppl_detector.config'):
        return RePPLDetector()


@pytest.fixture
def sample_keyword_result():
    """테스트용 KeywordResult 픽스처"""
    return KeywordResult(
        topic="AI",
        primary_keywords=["인공지능"],
        related_terms=["머신러닝"],
        context_keywords=[],
        confidence_score=0.9,
        generation_time=1.0,
        raw_response=""
    )


# --- RePPLDetector 단위 테스트 ---
class TestRePPLHallucinationDetector:
    """RePPLDetector의 각 계산 메서드를 단위 테스트합니다."""

    def test_analyze_repetition(self, detector):
        """반복성 점수 계산 로직을 테스트합니다."""
        text_normal = "이것은 정상적인 문장입니다. 반복이 없습니다."
        text_repeated = "중요한 내용은 반복됩니다. 중요한 내용은 반복됩니다."

        score_normal, phrases_normal = detector._analyze_repetition(text_normal)
        score_repeated, phrases_repeated = detector._analyze_repetition(text_repeated)

        assert score_normal == 0.0
        assert len(phrases_normal) == 0

        assert score_repeated >= 0.5
        assert "중요한 내용은 반복됩니다" in phrases_repeated

    @pytest.mark.asyncio
    async def test_calculate_perplexity_with_gpt(self, detector):
        """GPT API를 이용한 퍼플렉시티 계산 로직을 Mocking하여 테스트합니다."""
        mock_logprobs = [MagicMock(logprob=-0.5), MagicMock(logprob=-1.5)]
        expected_perplexity = math.exp(1.0)

        mock_openai_response = MagicMock()
        mock_openai_response.choices[0].logprobs.content = mock_logprobs

        with patch.object(detector.client.chat.completions, 'create',
                          AsyncMock(return_value=mock_openai_response)) as mock_create:
            perplexity = await detector._calculate_perplexity_with_gpt("테스트 텍스트")

            mock_create.assert_awaited_once()
            assert perplexity == pytest.approx(expected_perplexity)

    def test_calculate_semantic_entropy(self, detector):
        """의미적 엔트로피 계산 로직을 테스트합니다."""
        mock_embeddings_similar = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.301]])
        mock_embeddings_diverse = np.array([[0.1, 0.2, 0.3], [0.8, 0.1, 0.2], [-0.5, 0.4, 0.1]])

        # 1. 유사한 문장들로 구성된 텍스트 -> 낮은 엔트로피
        detector.sentence_model.encode = MagicMock(return_value=mock_embeddings_similar)
        entropy_low = detector._calculate_semantic_entropy("이것은 문장입니다. 이것도 문장입니다.")

        # 2. 다양한 문장들로 구성된 텍스트 -> 높은 엔트로피
        detector.sentence_model.encode = MagicMock(return_value=mock_embeddings_diverse)
        entropy_high = detector._calculate_semantic_entropy("이것은 문장입니다. 저것은 다른 문장입니다. 완전히 새로운 내용입니다.")

        assert entropy_high > entropy_low

    def test_calculate_confidence_score(self, detector):
        """최종 신뢰도 점수 계산 로직을 테스트합니다."""
        confidence_good = detector._calculate_confidence_score(0.05, 10, 0.8)
        assert confidence_good > 0.7

        confidence_bad = detector._calculate_confidence_score(0.8, 200, 0.1)
        assert confidence_bad < 0.3

        confidence_high_ppl = detector._calculate_confidence_score(0.1, 500, 0.5)
        assert confidence_high_ppl >= 0


# --- EnhancedIssueSearcher 통합 테스트 ---
@pytest.mark.asyncio
class TestRePPLEnhancedIssueSearcher:
    """EnhancedIssueSearcher의 전체 흐름을 테스트합니다."""

    @patch('src.hallucination_detection.enhanced_searcher.create_issue_searcher')
    @patch('src.hallucination_detection.enhanced_searcher.RePPLDetector')
    async def test_search_with_validation_success(self, mock_detector_class, mock_searcher_class,
                                                  sample_keyword_result):
        """모든 이슈가 검증을 통과하는 시나리오를 테스트합니다."""
        # Mock 설정
        mock_issue = IssueItem("테스트 이슈", "요약", "출처", None, 0.8, "news", "...", detailed_content="상세 내용")
        mock_base_search_result = SearchResult(
            query_keywords=["AI"], total_found=1, issues=[mock_issue],
            search_time=1.0, api_calls_used=1, confidence_score=0.8, time_period="1일", raw_responses=[]
        )

        mock_base_searcher = AsyncMock()
        mock_base_searcher.search_issues_from_keywords.return_value = mock_base_search_result
        mock_searcher_class.return_value = mock_base_searcher

        # Mock RePPL 분석 결과
        from src.hallucination_detection.models import RePPLScore
        mock_reppl_score = RePPLScore(
            confidence=0.9,
            repetition_score=0.1,
            perplexity=20.0,
            semantic_entropy=0.7
        )

        mock_detector_instance = AsyncMock()
        mock_detector_instance.analyze_issue.return_value = mock_reppl_score
        mock_detector_class.return_value = mock_detector_instance

        # 테스트 실행
        enhanced_searcher = EnhancedIssueSearcher(enable_consistency=False)  # 일관성 검사 비활성화
        final_result = await enhanced_searcher.search_with_validation(sample_keyword_result, "1일")

        # 검증
        assert final_result.total_found == 1
        assert final_result.issues[0].title == "테스트 이슈"
        assert hasattr(final_result.issues[0], 'hallucination_confidence')
        assert final_result.issues[0].hallucination_confidence == 0.9

    @patch('src.hallucination_detection.enhanced_searcher.create_issue_searcher')
    @patch('src.hallucination_detection.enhanced_searcher.RePPLDetector')
    @patch('src.hallucination_detection.enhanced_searcher.generate_keywords_for_topic')
    async def test_search_with_validation_retry(self, mock_regen_keywords, mock_detector_class, mock_searcher_class,
                                                sample_keyword_result):
        """검증 실패로 인해 키워드 재생성 및 재시도가 일어나는지 테스트합니다."""
        # Mock 설정
        mock_issue = IssueItem("신뢰도 낮은 이슈", "요약", "출처", None, 0.8, "news", "...")
        mock_base_search_result = SearchResult(
            query_keywords=["AI"], total_found=1, issues=[mock_issue],
            search_time=1.0, api_calls_used=1, confidence_score=0.8, time_period="1일", raw_responses=[]
        )

        # 첫 번째 검색은 성공하지만, 두 번째 검색 결과는 없다고 가정
        mock_base_searcher = AsyncMock()
        mock_base_searcher.search_issues_from_keywords.side_effect = [
            mock_base_search_result,
            SearchResult(query_keywords=[], total_found=0, issues=[], search_time=1, api_calls_used=1,
                         confidence_score=0, time_period="1일", raw_responses=[])
        ]
        mock_searcher_class.return_value = mock_base_searcher

        # RePPL 분석은 항상 낮은 점수를 반환하도록 설정
        from src.hallucination_detection.models import RePPLScore
        mock_reppl_score = RePPLScore(
            confidence=0.1,  # 임계값(0.5) 이하
            repetition_score=0.8,
            perplexity=100.0,
            semantic_entropy=0.2
        )

        mock_detector_instance = AsyncMock()
        mock_detector_instance.analyze_issue.return_value = mock_reppl_score
        mock_detector_class.return_value = mock_detector_instance

        # 키워드 재생성 Mock
        mock_regen_keywords.return_value = sample_keyword_result

        # 테스트 실행
        enhanced_searcher = EnhancedIssueSearcher(enable_consistency=False)
        final_result = await enhanced_searcher.search_with_validation(sample_keyword_result, "1일")

        # 검증
        assert mock_base_searcher.search_issues_from_keywords.call_count == 2  # 검색이 2번 일어났는지
        mock_regen_keywords.assert_awaited_once()  # 키워드 재생성이 1번 일어났는지
        assert final_result.total_found == 0  # 최종적으로 통과한 이슈는 없는지