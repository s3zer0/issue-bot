"""
LLM Judge 환각 탐지기 테스트.
"""

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock
import sys
import os

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.hallucination_detection.llm_judge import LLMJudgeDetector, LLMJudgeScore
from src.models import IssueItem


class TestLLMJudgeDetector:
    """LLM Judge 탐지기의 단위 테스트."""

    @pytest.fixture
    def detector(self):
        """테스트용 LLMJudgeDetector 인스턴스."""
        with patch('src.hallucination_detection.llm_judge.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_key"
            yield LLMJudgeDetector(model_name="gpt-4o")

    def test_create_evaluation_prompt(self, detector):
        """평가 프롬프트 생성 테스트."""
        text = "OpenAI는 2024년 12월에 GPT-5를 출시했습니다."
        context = "최신 AI 기술 동향"

        prompt = detector._create_evaluation_prompt(text, context)

        assert "원래 질문/주제: 최신 AI 기술 동향" in prompt
        assert text in prompt
        assert "사실적 정확성" in prompt
        assert "논리적 일관성" in prompt
        assert "맥락적 관련성" in prompt
        assert "출처 신뢰성" in prompt
        assert "JSON" in prompt

    def test_parse_evaluation_result(self, detector):
        """평가 결과 파싱 테스트."""
        mock_result = {
            "factual_accuracy": {"score": 80, "reasoning": "대부분 정확"},
            "logical_consistency": {"score": 90, "reasoning": "논리적"},
            "contextual_relevance": {"score": 85, "reasoning": "관련성 높음"},
            "source_reliability": {"score": 60, "reasoning": "출처 불명확"}
        }

        scores = detector._parse_evaluation_result(mock_result)

        assert scores["factual_accuracy"] == 0.8
        assert scores["logical_consistency"] == 0.9
        assert scores["contextual_relevance"] == 0.85
        assert scores["source_reliability"] == 0.6

    def test_calculate_weighted_score(self, detector):
        """가중 점수 계산 테스트."""
        scores = {
            "factual_accuracy": 0.8,
            "logical_consistency": 0.9,
            "contextual_relevance": 0.85,
            "source_reliability": 0.6
        }

        final_score = detector._calculate_weighted_score(scores)

        # 가중치에 따른 계산 검증
        expected = (0.8 * 0.35 + 0.9 * 0.25 + 0.85 * 0.20 + 0.6 * 0.20)
        assert final_score == pytest.approx(expected, rel=0.01)

    def test_identify_problematic_areas(self, detector):
        """문제 영역 식별 테스트."""
        mock_result = {
            "problematic_areas": [
                {"text": "2024년 12월 GPT-5 출시", "issue": "미래 날짜에 대한 확정적 주장"},
                {"text": "100% 정확도", "issue": "과도한 주장"},
                {"text": "모든 분야에서 최고", "issue": "검증 불가능한 일반화"}
            ]
        }

        areas = detector._identify_problematic_areas(mock_result)

        assert len(areas) == 3
        assert areas[0]["text"] == "2024년 12월 GPT-5 출시"
        assert "미래 날짜" in areas[0]["issue"]

    @pytest.mark.asyncio
    async def test_analyze_text_success(self, detector):
        """텍스트 분석 성공 케이스 테스트."""
        mock_llm_response = {
            "factual_accuracy": {"score": 85, "reasoning": "대체로 정확한 정보"},
            "logical_consistency": {"score": 90, "reasoning": "논리적으로 일관됨"},
            "contextual_relevance": {"score": 95, "reasoning": "주제에 매우 적합"},
            "source_reliability": {"score": 70, "reasoning": "일부 출처 누락"},
            "problematic_areas": [
                {"text": "특정 날짜", "issue": "확인 필요"}
            ],
            "overall_reasoning": "전반적으로 신뢰할 만한 텍스트"
        }

        with patch.object(detector, '_get_llm_evaluation', AsyncMock(return_value=mock_llm_response)):
            result = await detector.analyze_text(
                "AI 기술이 빠르게 발전하고 있습니다.",
                context="AI 기술 동향"
            )

            assert isinstance(result, LLMJudgeScore)
            assert result.confidence > 0.7
            assert len(result.problematic_areas) == 1
            assert "전반적으로 신뢰할 만한" in result.judge_reasoning

    @pytest.mark.asyncio
    async def test_analyze_text_low_confidence(self, detector):
        """낮은 신뢰도 텍스트 분석 테스트."""
        mock_llm_response = {
            "factual_accuracy": {"score": 30, "reasoning": "많은 사실 오류"},
            "logical_consistency": {"score": 40, "reasoning": "모순된 주장"},
            "contextual_relevance": {"score": 50, "reasoning": "주제에서 벗어남"},
            "source_reliability": {"score": 20, "reasoning": "출처 없음"},
            "problematic_areas": [
                {"text": "확인되지 않은 주장", "issue": "근거 부족"},
                {"text": "과장된 표현", "issue": "검증 불가"}
            ],
            "overall_reasoning": "신뢰도가 낮은 텍스트"
        }

        with patch.object(detector, '_get_llm_evaluation', AsyncMock(return_value=mock_llm_response)):
            result = await detector.analyze_text("의심스러운 주장들...")

            assert result.confidence < 0.5
            assert len(result.problematic_areas) == 2
            assert result.category_scores["factual_accuracy"] == 0.3

    @pytest.mark.asyncio
    async def test_analyze_text_error_handling(self, detector):
        """오류 처리 테스트."""
        with patch.object(detector, '_get_llm_evaluation', AsyncMock(side_effect=Exception("API 오류"))):
            result = await detector.analyze_text("테스트 텍스트")

            assert result.confidence == 0.5  # 중립 점수
            assert "평가 중 오류 발생" in result.judge_reasoning
            assert "API 오류" in result.analysis_details.get("error", "")

    @pytest.mark.asyncio
    async def test_analyze_claims(self, detector):
        """여러 주장 분석 테스트."""
        claims = [
            "AI는 2024년에 큰 발전을 이루었습니다.",
            "GPT-5는 1조 개의 파라미터를 가집니다.",
            "모든 기업이 AI를 도입했습니다."
        ]

        mock_scores = [0.8, 0.5, 0.3]

        with patch.object(detector, 'analyze_text', AsyncMock()) as mock_analyze:
            # 각 호출에 대해 다른 점수 반환
            mock_analyze.side_effect = [
                MagicMock(confidence=score) for score in mock_scores
            ]

            claim_scores = await detector.analyze_claims(claims, "AI 현황")

            assert len(claim_scores) == 3
            assert claim_scores[claims[0]] == 0.8
            assert claim_scores[claims[1]] == 0.5
            assert claim_scores[claims[2]] == 0.3


@pytest.mark.asyncio
class TestLLMJudgeIntegration:
    """LLM Judge의 통합 테스트."""

    async def test_llm_judge_with_issue_item(self):
        """IssueItem과의 통합 테스트."""
        with patch('src.hallucination_detection.llm_judge.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_key"

            detector = LLMJudgeDetector()

            issue = IssueItem(
                title="AI 기술의 미래",
                summary="AI가 모든 산업을 완전히 대체할 것입니다.",
                source="Tech Blog",
                published_date="2024-01-15",
                relevance_score=0.9,
                category="tech",
                content_snippet="...",
                detailed_content="AI는 10년 내에 인간의 모든 일자리를 대체하고..."
            )

            mock_response = {
                "factual_accuracy": {"score": 40, "reasoning": "과장된 주장"},
                "logical_consistency": {"score": 60, "reasoning": "일부 논리적"},
                "contextual_relevance": {"score": 80, "reasoning": "주제 관련"},
                "source_reliability": {"score": 50, "reasoning": "블로그 출처"},
                "problematic_areas": [
                    {"text": "모든 산업을 완전히 대체", "issue": "극단적 일반화"}
                ],
                "overall_reasoning": "과장된 미래 예측"
            }

            with patch.object(detector, '_get_llm_evaluation', AsyncMock(return_value=mock_response)):
                result = await detector.analyze_issue(issue, "AI 기술 동향")

                assert result.confidence < 0.6  # 낮은 신뢰도
                assert "issue_title" in result.analysis_details
                assert result.analysis_details["issue_title"] == "AI 기술의 미래"


@pytest.mark.asyncio
class TestLLMJudgeEdgeCases:
    """LLM Judge의 엣지 케이스 테스트."""

    async def test_empty_text_analysis(self):
        """빈 텍스트 분석 테스트."""
        with patch('src.hallucination_detection.llm_judge.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_key"

            detector = LLMJudgeDetector()

            with patch.object(detector, '_get_llm_evaluation', AsyncMock()) as mock_eval:
                mock_eval.return_value = {
                    "factual_accuracy": {"score": 50},
                    "logical_consistency": {"score": 50},
                    "contextual_relevance": {"score": 0},
                    "source_reliability": {"score": 0},
                    "problematic_areas": [],
                    "overall_reasoning": "빈 텍스트"
                }

                result = await detector.analyze_text("")

                assert result.confidence == pytest.approx(0.3)

    async def test_malformed_llm_response(self):
        """잘못된 형식의 LLM 응답 처리 테스트."""
        with patch('src.hallucination_detection.llm_judge.config') as mock_config:
            mock_config.get_openai_api_key.return_value = "test_key"

            detector = LLMJudgeDetector()

            # 일부 필드가 누락된 응답
            incomplete_response = {
                "factual_accuracy": {"score": 80},
                "logical_consistency": {"score": 90}
                # contextual_relevance와 source_reliability 누락
            }

            with patch.object(detector, '_get_llm_evaluation', AsyncMock(return_value=incomplete_response)):
                result = await detector.analyze_text("테스트")

                # 누락된 카테고리는 0.5로 처리되어야 함
                assert "contextual_relevance" in result.category_scores
                assert result.category_scores["contextual_relevance"] == 0.5
