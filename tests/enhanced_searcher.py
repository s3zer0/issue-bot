"""
EnhancedIssueSearcher 클래스에 대한 통합 테스트.

이 테스트는 여러 환각 탐지기를 조율하고, 신뢰도에 따라 재시도하는
복잡한 비즈니스 로직을 검증하는 데 중점을 둡니다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# 테스트 대상 및 관련 데이터 모델 임포트
from src.models import KeywordResult, SearchResult, IssueItem
from src.hallucination_detection.enhanced_searcher import EnhancedIssueSearcher
from src.hallucination_detection.threshold_manager import ThresholdManager
from src.hallucination_detection.models import RePPLScore, ConsistencyScore, LLMJudgeScore


@pytest.fixture
def searcher():
    """모든 외부 의존성이 모의(mock) 처리된 EnhancedIssueSearcher 픽스처."""
    # 의존하는 모든 클래스와 함수를 patch하여 격리된 테스트 환경을 만듭니다.
    with patch('src.hallucination_detection.enhanced_searcher.create_issue_searcher') as mock_base_searcher, \
            patch('src.hallucination_detection.enhanced_searcher.RePPLDetector') as mock_reppl, \
            patch('src.hallucination_detection.enhanced_searcher.SelfConsistencyChecker') as mock_consistency, \
            patch('src.hallucination_detection.enhanced_searcher.LLMJudgeDetector') as mock_llm_judge:
        searcher_instance = EnhancedIssueSearcher(threshold_manager=ThresholdManager())

        # 각 모의 객체에 대한 참조를 searcher 인스턴스에 저장하여 테스트에서 접근 가능하게 합니다.
        searcher_instance.mock_base_searcher = mock_base_searcher.return_value
        searcher_instance.mock_reppl = mock_reppl.return_value
        searcher_instance.mock_consistency = mock_consistency.return_value
        searcher_instance.mock_llm_judge = mock_llm_judge.return_value

        yield searcher_instance


@pytest.mark.asyncio
async def test_search_with_validation_success_first_try(searcher):
    """첫 시도에서 높은 신뢰도의 이슈를 충분히 발견하는 성공 경로를 테스트합니다."""
    # --- 모의 데이터 설정 ---
    # KeywordResult 모의
    keywords = KeywordResult(topic="AI", primary_keywords=["AI"], related_terms=[], context_keywords=[],
                             confidence_score=0.9, generation_time=1, raw_response="")

    # 높은 신뢰도 이슈 3개 생성
    issues = []
    for i in range(3):
        issue = IssueItem(title=f"High Confidence Issue {i + 1}", summary="...", source="...",
                          published_date="2024-01-01", relevance_score=0.9, category="news", content_snippet="...")
        # 각 이슈에 동적으로 환각 탐지 점수 속성 추가
        setattr(issue, 'hallucination_confidence', 0.9)
        issues.append(issue)

    # 기본 검색기의 search_issues_from_keywords가 위 이슈들을 반환하도록 설정
    searcher.mock_base_searcher.search_issues_from_keywords.return_value = SearchResult(query_keywords=["AI"],
                                                                                        total_found=3, issues=issues,
                                                                                        search_time=1, api_calls_used=1,
                                                                                        confidence_score=0.9,
                                                                                        time_period="1주")

    # 각 탐지기의 분석 결과 모의 처리 (높은 점수 반환)
    searcher.mock_reppl.analyze_issue.return_value = RePPLScore(confidence=0.9)
    searcher.mock_consistency.analyze_text.return_value = ConsistencyScore(confidence=0.9)
    searcher.mock_llm_judge.analyze_text.return_value = LLMJudgeScore(confidence=0.9)

    # --- 실행 ---
    final_result = await searcher.search_with_validation(keywords, "1주")

    # --- 검증 ---
    # 기본 검색기가 1번만 호출되었는지 확인 (재시도 없음)
    searcher.mock_base_searcher.search_issues_from_keywords.assert_awaited_once()
    # 최종 결과에 3개의 이슈가 포함되었는지 확인
    assert len(final_result.issues) == 3
    assert final_result.issues[0].title == "High Confidence Issue 1"
    # 모든 이슈의 신뢰도가 높은지 확인
    assert all(getattr(i, 'hallucination_confidence', 0) > 0.8 for i in final_result.issues)


@pytest.mark.asyncio
@patch('src.hallucination_detection.enhanced_searcher.generate_keywords_for_topic')
async def test_search_with_validation_retry_logic(mock_generate_keywords, searcher):
    """신뢰도 낮은 이슈만 발견 시, 키워드 재생성 및 재검색 로직을 테스트합니다."""
    # --- 첫 번째 시도 모의 설정 (낮은 신뢰도 결과) ---
    low_conf_issue = IssueItem(title="Low Confidence", summary="...", source="...", published_date=None,
                               relevance_score=0.5, category="tech", content_snippet="...")
    setattr(low_conf_issue, 'hallucination_confidence', 0.2)
    first_search_result = SearchResult(query_keywords=["keyword1"], total_found=1, issues=[low_conf_issue],
                                       search_time=1.0, api_calls_used=1, confidence_score=0.2, time_period="1주")

    # --- 두 번째 시도 모의 설정 (높은 신뢰도 결과) ---
    high_conf_issue = IssueItem(title="High Confidence", summary="...", source="...", published_date=None,
                                relevance_score=0.9, category="tech", content_snippet="...")
    setattr(high_conf_issue, 'hallucination_confidence', 0.9)
    second_search_result = SearchResult(query_keywords=["keyword2"], total_found=1, issues=[high_conf_issue],
                                        search_time=1.0, api_calls_used=1, confidence_score=0.9, time_period="1주")

    # base_searcher가 순서대로 낮은/높은 신뢰도 결과를 반환하도록 설정
    searcher.mock_base_searcher.search_issues_from_keywords.side_effect = [first_search_result, second_search_result]

    # 각 탐지기의 분석 결과도 순서대로 낮은/높은 신뢰도를 반환하도록 설정
    searcher.mock_reppl.analyze_issue.side_effect = [RePPLScore(confidence=0.2), RePPLScore(confidence=0.9)]
    searcher.mock_consistency.analyze_text.side_effect = [ConsistencyScore(confidence=0.2),
                                                          ConsistencyScore(confidence=0.9)]
    searcher.mock_llm_judge.analyze_text.side_effect = [LLMJudgeScore(confidence=0.2), LLMJudgeScore(confidence=0.9)]

    # 키워드 재생성 함수가 새로운 키워드를 반환하도록 모의 처리
    mock_generate_keywords.return_value = KeywordResult(topic="test", primary_keywords=["regenerated_keyword"],
                                                        related_terms=[], context_keywords=[], confidence_score=0.9,
                                                        generation_time=0.5, raw_response="")

    # --- 실행 ---
    initial_keywords = KeywordResult(topic="test", primary_keywords=["initial_keyword"], related_terms=[],
                                     context_keywords=[], confidence_score=0.9, generation_time=0.5, raw_response="")
    final_result = await searcher.search_with_validation(initial_keywords, "1주")

    # --- 검증 ---
    # 키워드 재생성 함수가 1번 호출되었는지 확인
    mock_generate_keywords.assert_awaited_once()
    # 기본 검색 함수는 2번 호출되었는지 확인
    assert searcher.mock_base_searcher.search_issues_from_keywords.await_count == 2
    # 최종 결과에 낮은 신뢰도 이슈와 높은 신뢰도 이슈가 모두 포함되었는지 확인
    assert len(final_result.issues) == 2
    issue_titles = [issue.title for issue in final_result.issues]
    assert "Low Confidence" in issue_titles
    assert "High Confidence" in issue_titles
