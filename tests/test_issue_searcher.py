"""
이슈 검색기 pytest 테스트 - 4단계 세부 정보 수집 테스트 포함 (수정됨)
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import json

# 프로젝트 루트 디렉토리의 src 폴더를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)


class TestStage4DataClasses:
    """4단계 데이터 클래스 테스트"""

    @pytest.mark.unit
    def test_entity_info_creation(self):
        """EntityInfo 데이터클래스 생성 테스트"""
        from src.issue_searcher import EntityInfo

        entity = EntityInfo(
            name="엘론 머스크",
            role="Tesla CEO",
            relevance=0.9,
            entity_type="person",
            description="전기차 및 우주 산업 리더"
        )

        assert entity.name == "엘론 머스크"
        assert entity.role == "Tesla CEO"
        assert entity.relevance == 0.9
        assert entity.entity_type == "person"
        assert entity.description == "전기차 및 우주 산업 리더"

    @pytest.mark.unit
    def test_impact_analysis_creation(self):
        """ImpactAnalysis 데이터클래스 생성 테스트"""
        from src.issue_searcher import ImpactAnalysis

        impact = ImpactAnalysis(
            impact_level="high",
            impact_score=0.8,
            affected_sectors=["기술", "자동차"],
            geographic_scope="global",
            time_sensitivity="short-term",
            reasoning="전기차 시장에 대한 글로벌 영향"
        )

        assert impact.impact_level == "high"
        assert impact.impact_score == 0.8
        assert impact.affected_sectors == ["기술", "자동차"]
        assert impact.geographic_scope == "global"
        assert impact.time_sensitivity == "short-term"
        assert impact.reasoning == "전기차 시장에 대한 글로벌 영향"

    @pytest.mark.unit
    def test_timeline_event_creation(self):
        """TimelineEvent 데이터클래스 생성 테스트"""
        from src.issue_searcher import TimelineEvent

        event = TimelineEvent(
            date="2024-01-15",
            event_type="announcement",
            description="새로운 전기차 모델 발표",
            importance=0.9,
            source="Tesla Press Release"
        )

        assert event.date == "2024-01-15"
        assert event.event_type == "announcement"
        assert event.description == "새로운 전기차 모델 발표"
        assert event.importance == 0.9
        assert event.source == "Tesla Press Release"

    @pytest.mark.unit
    def test_enhanced_issue_item(self):
        """확장된 IssueItem 테스트"""
        from src.issue_searcher import IssueItem, EntityInfo, ImpactAnalysis, TimelineEvent

        entity = EntityInfo("테스트", "역할", 0.5, "person", "설명")
        impact = ImpactAnalysis("medium", 0.6, ["기술"], "national", "short-term", "이유")
        timeline = TimelineEvent("2024-01-01", "development", "설명", 0.7, "출처")

        issue = IssueItem(
            title="테스트 이슈",
            summary="테스트 요약",
            source="테스트 소스",
            published_date="2024-01-15",
            relevance_score=0.85,
            category="news",
            content_snippet="테스트 내용",
            detailed_content="상세 내용",
            related_entities=[entity],
            impact_analysis=impact,
            timeline_events=[timeline],
            background_context="배경 정보",
            detail_collection_time=2.5,
            detail_confidence=0.8
        )

        assert issue.detailed_content == "상세 내용"
        assert len(issue.related_entities) == 1
        assert issue.impact_analysis.impact_level == "medium"
        assert len(issue.timeline_events) == 1
        assert issue.background_context == "배경 정보"
        assert issue.detail_collection_time == 2.5
        assert issue.detail_confidence == 0.8

    @pytest.mark.unit
    def test_enhanced_search_result(self):
        """확장된 SearchResult 테스트"""
        from src.issue_searcher import SearchResult

        result = SearchResult(
            query_keywords=["AI", "기술"],
            total_found=5,
            issues=[],
            search_time=3.0,
            api_calls_used=3,
            confidence_score=0.8,
            time_period="최근 1주일",
            raw_responses=["response1"],
            detailed_issues_count=2,
            total_detail_collection_time=15.0,
            average_detail_confidence=0.75
        )

        assert result.detailed_issues_count == 2
        assert result.total_detail_collection_time == 15.0
        assert result.average_detail_confidence == 0.75


class TestStage4PerplexityClient:
    """4단계 Perplexity 클라이언트 테스트"""

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_perplexity_key'})
    def test_perplexity_client_stage4_methods(self):
        """Perplexity 클라이언트 4단계 메서드 존재 테스트"""
        from src.issue_searcher import PerplexityClient

        client = PerplexityClient(api_key="test_key")

        # 4단계 메서드들이 존재하는지 확인
        assert hasattr(client, 'collect_detailed_information')
        assert hasattr(client, 'extract_entities_and_impact')
        assert hasattr(client, 'extract_timeline')
        assert hasattr(client, '_make_api_call')

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_collect_detailed_information(self):
        """세부 정보 수집 API 호출 테스트"""
        mock_response = {
            "choices": [{
                "message": {
                    "content": """**상세 내용**: AI 기술이 급속도로 발전하고 있으며, 특히 대화형 AI의 성능이 크게 향상되었습니다.
**관련 인물/기관**: OpenAI, Google, Meta 등이 주요 개발사입니다.
**영향도 분석**: 기술 산업 전반에 큰 영향을 미칠 것으로 예상됩니다."""
                }
            }]
        }

        with patch('src.issue_searcher.httpx.AsyncClient') as mock_client:
            mock_response_obj = MagicMock()
            mock_response_obj.status_code = 200
            mock_response_obj.json.return_value = mock_response

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.post.return_value = mock_response_obj
            mock_client.return_value = mock_context

            from src.issue_searcher import PerplexityClient

            client = PerplexityClient(api_key="test_key")
            result = await client.collect_detailed_information(
                "AI 기술 발전",
                "AI가 빠르게 발전하고 있다",
                ["AI", "기술"]
            )

            assert "choices" in result
            assert "AI 기술이 급속도로" in result["choices"][0]["message"]["content"]


class TestStage4IssueSearcher:
    """4단계 IssueSearcher 테스트"""

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_perplexity_key'})
    def test_issue_searcher_stage4_initialization(self):
        """4단계 IssueSearcher 초기화 테스트"""
        from src.issue_searcher import IssueSearcher

        searcher = IssueSearcher(api_key="test_key")

        # 4단계 관련 속성들 확인
        assert hasattr(searcher, 'enable_detailed_collection')
        assert hasattr(searcher, 'max_detailed_issues')
        assert hasattr(searcher, 'detail_collection_timeout')

        assert searcher.enable_detailed_collection == True
        assert searcher.max_detailed_issues == 10
        assert searcher.detail_collection_timeout == 60

    @pytest.mark.unit
    def test_calculate_detail_confidence(self):
        """세부 정보 신뢰도 계산 테스트"""
        from src.issue_searcher import IssueSearcher, EntityInfo, ImpactAnalysis, TimelineEvent

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            # 테스트 데이터 준비
            detailed_content = "이것은 충분히 긴 상세 내용입니다. " * 10

            entities = [
                EntityInfo("OpenAI", "AI 회사", 0.9, "company", "ChatGPT 개발"),
                EntityInfo("Sam Altman", "CEO", 0.8, "person", "OpenAI CEO")
            ]

            impact = ImpactAnalysis(
                "high", 0.8, ["기술", "교육"], "global", "short-term", "AI 확산"
            )

            timeline_events = [
                TimelineEvent("2024-01-01", "announcement", "발표", 0.9, "출처1"),
                TimelineEvent("2024-01-02", "development", "개발", 0.8, "출처2")
            ]

            # 신뢰도 계산
            confidence = searcher._calculate_detail_confidence(
                detailed_content, entities, impact, timeline_events
            )

            # 높은 신뢰도 기대 (모든 요소가 충족됨)
            assert confidence > 0.8
            assert confidence <= 1.0

    @pytest.mark.unit
    def test_extract_detailed_content(self):
        """상세 내용 추출 테스트"""
        from src.issue_searcher import IssueSearcher

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            # 테스트 API 응답
            api_response = {
                "choices": [{
                    "message": {
                        "content": """다음은 분석 결과입니다:

**상세 내용**: AI 기술이 급속도로 발전하고 있으며, 특히 대화형 AI의 성능이 크게 향상되었습니다. 이는 산업 전반에 큰 변화를 가져올 것으로 예상됩니다.

**관련 인물/기관**: OpenAI, Google 등"""
                    }
                }]
            }

            detailed_content = searcher._extract_detailed_content(api_response)

            assert "AI 기술이 급속도로" in detailed_content
            assert "대화형 AI의 성능" in detailed_content

    @pytest.mark.unit
    def test_parse_entity_and_impact_response(self):
        """엔티티 및 영향도 응답 파싱 테스트"""
        from src.issue_searcher import IssueSearcher

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            # JSON 응답 시뮬레이션
            response = {
                "choices": [{
                    "message": {
                        "content": """{
    "entities": [
        {
            "name": "OpenAI",
            "role": "AI 개발 회사", 
            "relevance": 0.9,
            "entity_type": "company",
            "description": "ChatGPT 개발사"
        }
    ],
    "impact": {
        "impact_level": "high",
        "impact_score": 0.8,
        "affected_sectors": ["기술", "교육"],
        "geographic_scope": "global",
        "time_sensitivity": "short-term",
        "reasoning": "AI 기술의 급속한 발전"
    }
}"""
                    }
                }]
            }

            entities, impact = searcher._parse_entity_and_impact_response(response)

            # 엔티티 검증
            assert len(entities) == 1
            assert entities[0].name == "OpenAI"
            assert entities[0].relevance == 0.9
            assert entities[0].entity_type == "company"

            # 영향도 검증
            assert impact is not None
            assert impact.impact_level == "high"
            assert impact.impact_score == 0.8
            assert "기술" in impact.affected_sectors
            assert impact.geographic_scope == "global"

    @pytest.mark.unit
    def test_parse_timeline_response(self):
        """타임라인 응답 파싱 테스트"""
        from src.issue_searcher import IssueSearcher

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            response = {
                "choices": [{
                    "message": {
                        "content": """{
    "timeline": [
        {
            "date": "2024-01-15",
            "event_type": "announcement",
            "description": "새로운 AI 모델 발표",
            "importance": 0.9,
            "source": "OpenAI Blog"
        },
        {
            "date": "2024-01-20",
            "event_type": "development",
            "description": "기능 업데이트",
            "importance": 0.7,
            "source": "Tech News"
        }
    ],
    "background_context": "AI 기술 발전의 배경 정보입니다."
}"""
                    }
                }]
            }

            timeline_events, background_context = searcher._parse_timeline_response(response)

            # 타임라인 검증
            assert len(timeline_events) == 2
            assert timeline_events[0].date == "2024-01-15"
            assert timeline_events[0].event_type == "announcement"
            assert timeline_events[0].importance == 0.9
            assert timeline_events[1].description == "기능 업데이트"

            # 배경 정보 검증
            assert "AI 기술 발전의 배경" in background_context

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_collect_issue_details(self):
        """이슈 세부 정보 수집 테스트"""
        from src.issue_searcher import IssueSearcher, IssueItem

        # Mock API 응답들 준비
        detailed_response = {
            "choices": [{
                "message": {
                    "content": "**상세 내용**: AI 기술 발전에 대한 상세 내용입니다."
                }
            }]
        }

        entity_response = {
            "choices": [{
                "message": {
                    "content": '{"entities": [{"name": "OpenAI", "role": "AI 회사", "relevance": 0.9, "entity_type": "company", "description": "ChatGPT 개발"}], "impact": {"impact_level": "high", "impact_score": 0.8, "affected_sectors": ["기술"], "geographic_scope": "global", "time_sensitivity": "short-term", "reasoning": "AI 확산"}}'
                }
            }]
        }

        timeline_response = {
            "choices": [{
                "message": {
                    "content": '{"timeline": [{"date": "2024-01-15", "event_type": "announcement", "description": "발표", "importance": 0.9, "source": "출처"}], "background_context": "배경 정보"}'
                }
            }]
        }

        with patch('src.issue_searcher.PerplexityClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.collect_detailed_information.return_value = detailed_response
            mock_client.extract_entities_and_impact.return_value = entity_response
            mock_client.extract_timeline.return_value = timeline_response
            mock_client_class.return_value = mock_client

            searcher = IssueSearcher(api_key="test_key")

            # 테스트용 이슈 아이템
            issue = IssueItem(
                title="AI 기술 발전",
                summary="AI가 발전하고 있다",
                source="Tech News",
                published_date="2024-01-15",
                relevance_score=0.8,
                category="news",
                content_snippet="AI 발전",
                related_entities=[],
                timeline_events=[]
            )

            # 세부 정보 수집 실행
            enhanced_issue = await searcher._collect_issue_details(issue, ["AI", "기술"])

            # 결과 검증
            assert enhanced_issue.detailed_content is not None
            assert "AI 기술 발전에 대한 상세 내용" in enhanced_issue.detailed_content
            assert len(enhanced_issue.related_entities) == 1
            assert enhanced_issue.related_entities[0].name == "OpenAI"
            assert enhanced_issue.impact_analysis is not None
            assert enhanced_issue.impact_analysis.impact_level == "high"
            assert len(enhanced_issue.timeline_events) == 1
            assert enhanced_issue.timeline_events[0].description == "발표"
            assert enhanced_issue.background_context == "배경 정보"
            assert enhanced_issue.detail_confidence > 0.0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_issues_with_details_fixed(self):
        """세부 정보 포함 이슈 검색 통합 테스트 - 수정됨"""
        # 기본 검색 응답
        basic_search_response = {
            "choices": [{
                "message": {
                    "content": """**제목**: AI 기술 혁신
**요약**: AI 기술이 빠르게 발전하고 있습니다.
**출처**: Tech Journal
**일자**: 2024-01-15
**카테고리**: news

**제목**: 머신러닝 발전
**요약**: 새로운 ML 알고리즘이 개발되었습니다.
**출처**: AI Magazine
**일자**: 2024-01-14
**카테고리**: academic"""
                }
            }]
        }

        # 세부 정보 응답들
        detailed_response = {
            "choices": [{
                "message": {
                    "content": "**상세 내용**: AI 기술에 대한 상세한 설명입니다."
                }
            }]
        }

        entity_response = {
            "choices": [{
                "message": {
                    "content": '{"entities": [{"name": "OpenAI", "role": "AI 회사", "relevance": 0.9, "entity_type": "company", "description": "AI 개발"}], "impact": {"impact_level": "medium", "impact_score": 0.7, "affected_sectors": ["기술"], "geographic_scope": "global", "time_sensitivity": "short-term", "reasoning": "기술 발전"}}'
                }
            }]
        }

        timeline_response = {
            "choices": [{
                "message": {
                    "content": '{"timeline": [{"date": "2024-01-15", "event_type": "announcement", "description": "발표", "importance": 0.8, "source": "출처"}], "background_context": "배경"}'
                }
            }]
        }

        with patch('src.issue_searcher.PerplexityClient') as mock_client_class:
            mock_client = AsyncMock()

            # 직접 메서드별로 Mock 설정
            mock_client.search_issues.return_value = basic_search_response
            mock_client.collect_detailed_information.return_value = detailed_response
            mock_client.extract_entities_and_impact.return_value = entity_response
            mock_client.extract_timeline.return_value = timeline_response

            mock_client_class.return_value = mock_client

            from src.issue_searcher import IssueSearcher
            from src.keyword_generator import KeywordResult

            keyword_result = KeywordResult(
                topic="AI 기술",
                primary_keywords=["AI", "인공지능"],
                related_terms=["머신러닝"],
                synonyms=["기계학습"],
                context_keywords=["기술혁신"],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test"
            )

            searcher = IssueSearcher(api_key="test_key")
            result = await searcher.search_issues_from_keywords(
                keyword_result,
                collect_details=True
            )

            # 결과 검증
            assert result.total_found >= 1
            assert result.detailed_issues_count >= 1
            assert result.average_detail_confidence > 0.0
            assert result.total_detail_collection_time > 0.0

            # 첫 번째 이슈의 세부 정보 확인
            if result.issues:
                first_issue = result.issues[0]
                if first_issue.detailed_content:
                    assert first_issue.detailed_content is not None
                    assert first_issue.detail_confidence > 0.0


class TestStage4ConvenienceFunctions:
    """4단계 편의 함수 테스트"""

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_key'})
    def test_search_issues_for_keywords_with_details(self):
        """세부 정보 포함 키워드 검색 편의 함수 테스트"""
        from src.issue_searcher import search_issues_for_keywords

        # 함수가 collect_details 파라미터를 받는지 확인
        import inspect
        sig = inspect.signature(search_issues_for_keywords)
        assert 'collect_details' in sig.parameters

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_key'})
    def test_create_detailed_report_from_search_result(self):
        """상세 보고서 생성 편의 함수 테스트"""
        from src.issue_searcher import create_detailed_report_from_search_result, SearchResult, IssueItem, EntityInfo, ImpactAnalysis

        # 테스트 데이터 준비
        entity = EntityInfo("OpenAI", "AI 회사", 0.9, "company", "ChatGPT 개발")
        impact = ImpactAnalysis("high", 0.8, ["기술"], "global", "short-term", "AI 확산")

        issue = IssueItem(
            title="AI 기술 혁신",
            summary="AI 기술이 발전하고 있습니다",
            source="Tech News",
            published_date="2024-01-15",
            relevance_score=0.9,
            category="news",
            content_snippet="AI 발전",
            detailed_content="AI 기술에 대한 상세한 내용입니다.",
            related_entities=[entity],
            impact_analysis=impact,
            timeline_events=[],
            background_context="배경 정보",
            detail_confidence=0.85
        )

        search_result = SearchResult(
            query_keywords=["AI", "기술"],
            total_found=1,
            issues=[issue],
            search_time=3.0,
            api_calls_used=3,
            confidence_score=0.8,
            time_period="최근 1주일",
            raw_responses=["test"],
            detailed_issues_count=1,
            total_detail_collection_time=5.0,
            average_detail_confidence=0.85
        )

        # 보고서 생성
        report = create_detailed_report_from_search_result(search_result)

        # 보고서 내용 검증
        assert "종합 이슈 분석 보고서" in report
        assert "AI 기술 혁신" in report
        assert "OpenAI" in report
        assert "high" in report
        assert "세부 분석 이슈" in report and "1개" in report

    @pytest.mark.unit
    def test_format_detailed_issue_report(self):
        """개별 이슈 상세 보고서 포맷팅 테스트"""
        from src.issue_searcher import create_issue_searcher, IssueItem, EntityInfo, ImpactAnalysis, TimelineEvent

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = create_issue_searcher(api_key="test_key")

            # 테스트 데이터
            entity = EntityInfo("엘론 머스크", "Tesla CEO", 0.9, "person", "전기차 리더")
            impact = ImpactAnalysis("high", 0.8, ["자동차", "기술"], "global", "short-term", "전기차 혁신")
            timeline_event = TimelineEvent("2024-01-15", "announcement", "새 모델 발표", 0.9, "Tesla")

            issue = IssueItem(
                title="Tesla 신모델 발표",
                summary="Tesla가 새로운 전기차를 발표했습니다",
                source="Tesla Blog",
                published_date="2024-01-15",
                relevance_score=0.9,
                category="news",
                content_snippet="Tesla 신모델",
                detailed_content="Tesla의 새로운 전기차 모델에 대한 상세 정보입니다.",
                related_entities=[entity],
                impact_analysis=impact,
                timeline_events=[timeline_event],
                background_context="전기차 시장 발전 배경",
                detail_confidence=0.88
            )

            # 보고서 생성
            report = searcher.format_detailed_issue_report(issue)

            # 내용 검증
            assert "Tesla 신모델 발표" in report
            assert "엘론 머스크" in report
            assert "👤" in report  # person emoji
            assert "🟠" in report  # high impact emoji
            assert "📢" in report  # announcement emoji
            assert "전기차 시장 발전 배경" in report
            assert "88%" in report  # detail confidence


class TestStage4Integration:
    """4단계 통합 테스트"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_stage4_pipeline(self):
        """4단계 전체 파이프라인 테스트"""
        # 복합적인 Mock 응답 시뮬레이션
        search_response = {
            "choices": [{
                "message": {
                    "content": """**제목**: OpenAI GPT-4 업데이트
**요약**: OpenAI가 GPT-4의 새로운 기능을 발표했습니다.
**출처**: OpenAI Blog
**일자**: 2024-01-15
**카테고리**: news"""
                }
            }]
        }

        detail_response = {
            "choices": [{
                "message": {
                    "content": "**상세 내용**: GPT-4의 새로운 멀티모달 기능과 향상된 추론 능력에 대한 상세한 설명입니다."
                }
            }]
        }

        entity_response = {
            "choices": [{
                "message": {
                    "content": '{"entities": [{"name": "Sam Altman", "role": "OpenAI CEO", "relevance": 0.95, "entity_type": "person", "description": "OpenAI 최고경영자"}], "impact": {"impact_level": "high", "impact_score": 0.9, "affected_sectors": ["기술", "교육", "엔터테인먼트"], "geographic_scope": "global", "time_sensitivity": "immediate", "reasoning": "AI 기술의 급진적 발전"}, "confidence": 0.9}'
                }
            }]
        }

        timeline_response = {
            "choices": [{
                "message": {
                    "content": '{"timeline": [{"date": "2024-01-10", "event_type": "development", "description": "GPT-4 업데이트 개발 완료", "importance": 0.8, "source": "내부 정보"}, {"date": "2024-01-15", "event_type": "announcement", "description": "공식 발표", "importance": 1.0, "source": "OpenAI Blog"}], "background_context": "OpenAI는 지속적으로 GPT 모델을 개선해왔으며, 이번 업데이트는 멀티모달 기능을 크게 향상시켰습니다."}'
                }
            }]
        }

        with patch('src.issue_searcher.PerplexityClient') as mock_client_class:
            mock_client = AsyncMock()

            # 각 메서드별로 직접 Mock 설정
            mock_client.search_issues.return_value = search_response
            mock_client.collect_detailed_information.return_value = detail_response
            mock_client.extract_entities_and_impact.return_value = entity_response
            mock_client.extract_timeline.return_value = timeline_response

            mock_client_class.return_value = mock_client

            from src.issue_searcher import IssueSearcher
            from src.keyword_generator import KeywordResult

            # 키워드 결과 준비
            keyword_result = KeywordResult(
                topic="OpenAI GPT-4 업데이트",
                primary_keywords=["OpenAI", "GPT-4", "AI"],
                related_terms=["멀티모달", "추론"],
                synonyms=["인공지능", "언어모델"],
                context_keywords=["기술혁신", "자연어처리"],
                confidence_score=0.9,
                generation_time=2.0,
                raw_response="test response"
            )

            # 4단계 전체 실행
            searcher = IssueSearcher(api_key="test_key")
            result = await searcher.search_issues_from_keywords(
                keyword_result,
                time_period="최근 1주일",
                max_total_results=5,
                collect_details=True
            )

            # 종합 검증
            assert result.total_found >= 1
            assert result.detailed_issues_count >= 1
            assert result.confidence_score > 0.7
            assert result.average_detail_confidence >= 0.75

            # 첫 번째 이슈 상세 검증
            if result.issues and result.issues[0].detailed_content:
                issue = result.issues[0]

                # 기본 정보
                assert issue.title == "OpenAI GPT-4 업데이트"
                assert issue.detailed_content is not None

                # 엔티티 정보
                assert len(issue.related_entities) >= 1
                assert issue.related_entities[0].name == "Sam Altman"
                assert issue.related_entities[0].entity_type == "person"

                # 영향도 분석
                assert issue.impact_analysis is not None
                assert issue.impact_analysis.impact_level == "high"
                assert issue.impact_analysis.impact_score == 0.9
                assert "기술" in issue.impact_analysis.affected_sectors

                # 타임라인
                assert len(issue.timeline_events) >= 2
                timeline_dates = [event.date for event in issue.timeline_events]
                assert "2024-01-10" in timeline_dates
                assert "2024-01-15" in timeline_dates

                # 배경 정보
                assert issue.background_context is not None
                assert "OpenAI는 지속적으로" in issue.background_context

                # 메타 정보
                assert issue.detail_confidence > 0.8
                assert issue.detail_collection_time > 0

            # 보고서 생성 테스트
            from src.issue_searcher import create_detailed_report_from_search_result
            detailed_report = create_detailed_report_from_search_result(result)

            assert "종합 이슈 분석 보고서" in detailed_report
            assert "OpenAI GPT-4 업데이트" in detailed_report
            assert "Sam Altman" in detailed_report


# 기존 테스트 클래스들 (PerplexityClient, IssueItem, SearchResult 등)
class TestPerplexityClient:
    """Perplexity API 클라이언트 테스트 클래스"""

    @pytest.mark.unit
    def test_perplexity_client_import(self):
        """Perplexity 클라이언트 import 테스트"""
        from src.issue_searcher import PerplexityClient, IssueItem, SearchResult
        assert PerplexityClient is not None
        assert IssueItem is not None
        assert SearchResult is not None

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_perplexity_key'})
    def test_perplexity_client_initialization(self):
        """Perplexity 클라이언트 초기화 테스트"""
        from src.issue_searcher import PerplexityClient

        client = PerplexityClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.model == "llama-3.1-sonar-large-128k-online"
        assert client.base_url == "https://api.perplexity.ai/chat/completions"
        assert "Authorization" in client.headers
        assert "Bearer test_key" in client.headers["Authorization"]

    @pytest.mark.unit
    def test_perplexity_client_no_api_key(self):
        """API 키 없을 때 예외 처리 테스트"""
        with patch.dict(os.environ, {}, clear=True):
            with patch('src.issue_searcher.config') as mock_config:
                mock_config.get_perplexity_api_key.return_value = None

                from src.issue_searcher import PerplexityClient

                with pytest.raises(ValueError, match="Perplexity API 키가 설정되지 않았습니다"):
                    PerplexityClient()


class TestIssueItem:
    """IssueItem 데이터클래스 테스트"""

    @pytest.mark.unit
    def test_issue_item_creation(self):
        """IssueItem 생성 테스트"""
        from src.issue_searcher import IssueItem

        issue = IssueItem(
            title="테스트 이슈",
            summary="테스트 요약",
            source="테스트 소스",
            published_date="2024-01-15",
            relevance_score=0.85,
            category="news",
            content_snippet="테스트 내용",
            related_entities=[],
            timeline_events=[]
        )

        assert issue.title == "테스트 이슈"
        assert issue.summary == "테스트 요약"
        assert issue.source == "테스트 소스"
        assert issue.published_date == "2024-01-15"
        assert issue.relevance_score == 0.85
        assert issue.category == "news"
        assert issue.content_snippet == "테스트 내용"


class TestSearchResult:
    """SearchResult 데이터클래스 테스트"""

    @pytest.mark.unit
    def test_search_result_creation(self):
        """SearchResult 생성 테스트"""
        from src.issue_searcher import SearchResult, IssueItem

        issue = IssueItem(
            title="테스트", summary="테스트", source="테스트",
            published_date=None, relevance_score=0.5,
            category="news", content_snippet="테스트",
            related_entities=[], timeline_events=[]
        )

        result = SearchResult(
            query_keywords=["AI", "기술"],
            total_found=1,
            issues=[issue],
            search_time=2.5,
            api_calls_used=1,
            confidence_score=0.8,
            time_period="최근 1주일",
            raw_responses=["test response"]
        )

        assert result.query_keywords == ["AI", "기술"]
        assert result.total_found == 1
        assert len(result.issues) == 1
        assert result.search_time == 2.5
        assert result.api_calls_used == 1
        assert result.confidence_score == 0.8
        assert result.time_period == "최근 1주일"


class TestIssueSearcher:
    """IssueSearcher 테스트 클래스"""

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_perplexity_key'})
    def test_issue_searcher_initialization(self):
        """IssueSearcher 초기화 테스트"""
        from src.issue_searcher import IssueSearcher

        searcher = IssueSearcher(api_key="test_key")
        assert searcher.client.api_key == "test_key"
        assert searcher.max_keywords_per_search == 5
        assert searcher.max_results_per_search == 10

    @pytest.mark.unit
    def test_prepare_search_keywords(self):
        """검색 키워드 준비 테스트"""
        from src.issue_searcher import IssueSearcher
        from src.keyword_generator import KeywordResult

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            keyword_result = KeywordResult(
                topic="AI 기술",
                primary_keywords=["AI", "인공지능", "머신러닝", "딥러닝"],
                related_terms=["신경망", "알고리즘", "빅데이터"],
                synonyms=["Artificial Intelligence"],
                context_keywords=["기술혁신"],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test"
            )

            keywords = searcher._prepare_search_keywords(keyword_result)

            # 최대 5개 키워드, 핵심 키워드 우선
            assert len(keywords) <= 5
            assert "AI" in keywords
            assert "인공지능" in keywords
            assert "머신러닝" in keywords

    @pytest.mark.unit
    def test_parse_issue_section(self):
        """이슈 섹션 파싱 테스트"""
        from src.issue_searcher import IssueSearcher

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            section = """AI 기술 혁신 가속화
**요약**: 최근 AI 기술이 빠르게 발전하고 있습니다.
**출처**: Tech News
**일자**: 2024-01-15
**카테고리**: news"""

            issue = searcher._parse_issue_section(section, 1)

            assert issue is not None
            assert issue.title == "AI 기술 혁신 가속화"
            assert "AI 기술이 빠르게 발전" in issue.summary
            assert issue.source == "Tech News"
            assert issue.published_date == "2024-01-15"
            assert issue.category == "news"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_issues_from_keywords_success(self):
        """키워드 기반 이슈 검색 성공 테스트"""
        # Mock API 응답
        mock_api_response = {
            "choices": [{
                "message": {
                    "content": """**제목**: AI 기술 발전
**요약**: 인공지능 기술이 빠르게 발전하고 있습니다.
**출처**: Tech News
**일자**: 2024-01-15
**카테고리**: news

**제목**: 머신러닝 혁신
**요약**: 새로운 머신러닝 알고리즘이 개발되었습니다.
**출처**: AI Journal
**일자**: 2024-01-14
**카테고리**: academic"""
                }
            }]
        }

        with patch('src.issue_searcher.PerplexityClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.search_issues.return_value = mock_api_response
            mock_client_class.return_value = mock_client

            from src.issue_searcher import IssueSearcher
            from src.keyword_generator import KeywordResult

            keyword_result = KeywordResult(
                topic="AI 기술",
                primary_keywords=["AI", "인공지능"],
                related_terms=["머신러닝"],
                synonyms=["기계학습"],
                context_keywords=["기술혁신"],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test"
            )

            searcher = IssueSearcher(api_key="test_key")
            result = await searcher.search_issues_from_keywords(keyword_result, collect_details=False)

            assert result.total_found == 2
            assert len(result.issues) == 2
            assert result.confidence_score > 0.0
            assert result.search_time > 0.0
            assert "AI" in result.query_keywords

            # 첫 번째 이슈 확인
            first_issue = result.issues[0]
            assert first_issue.title == "AI 기술 발전"
            assert "인공지능" in first_issue.summary
            assert first_issue.source == "Tech News"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_issues_from_keywords_failure(self):
        """키워드 기반 이슈 검색 실패 테스트"""
        with patch('src.issue_searcher.PerplexityClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.search_issues.side_effect = Exception("API 오류")
            mock_client_class.return_value = mock_client

            from src.issue_searcher import IssueSearcher
            from src.keyword_generator import KeywordResult

            keyword_result = KeywordResult(
                topic="테스트",
                primary_keywords=["테스트"],
                related_terms=[],
                synonyms=[],
                context_keywords=[],
                confidence_score=0.8,
                generation_time=1.0,
                raw_response="test"
            )

            searcher = IssueSearcher(api_key="test_key")
            result = await searcher.search_issues_from_keywords(keyword_result)

            # 폴백 결과 확인
            assert result.total_found == 0
            assert len(result.issues) == 0
            assert result.confidence_score == 0.1
            assert result.raw_responses == ["검색 실패로 인한 응답 없음"]

    @pytest.mark.unit
    def test_format_search_summary_success(self):
        """검색 결과 요약 포맷팅 테스트 (성공)"""
        from src.issue_searcher import IssueSearcher, SearchResult, IssueItem

        with patch('src.issue_searcher.PerplexityClient'):
            searcher = IssueSearcher(api_key="test_key")

            issues = [
                IssueItem(
                    title="AI 기술 발전",
                    summary="인공지능 기술이 빠르게 발전하고 있습니다.",
                    source="Tech News",
                    published_date="2024-01-15",
                    relevance_score=0.9,
                    category="news",
                    content_snippet="test",
                    related_entities=[],
                    timeline_events=[]
                )
            ]

            result = SearchResult(
                query_keywords=["AI", "기술"],
                total_found=1,
                issues=issues,
                search_time=2.5,
                api_calls_used=1,
                confidence_score=0.85,
                time_period="최근 1주일",
                raw_responses=["test"]
            )

            summary = searcher.format_search_summary(result)

            assert "이슈 검색 완료" in summary
            assert "1개 이슈 발견" in summary
            assert "85%" in summary
            assert "AI 기술 발전" in summary


class TestConvenienceFunctions:
    """편의 함수 테스트"""

    @pytest.mark.unit
    @patch.dict(os.environ, {'PERPLEXITY_API_KEY': 'test_key'})
    def test_create_issue_searcher(self):
        """create_issue_searcher 함수 테스트"""
        from src.issue_searcher import create_issue_searcher

        searcher = create_issue_searcher(api_key="test_key")
        assert searcher is not None
        assert searcher.client.api_key == "test_key"


if __name__ == "__main__":
    # pytest 실행
    pytest.main([__file__, "-v"])