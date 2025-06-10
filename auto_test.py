"""자동화된 E2E(End-to-End) 테스트 스크립트

이 스크립트는 Discord 봇을 직접 실행하지 않고,
키워드 생성부터 이슈 검색, 환각 탐지, 보고서 생성까지의
핵심 기능 전체 흐름을 순차적으로 실행하여 검증합니다.
"""

import asyncio
import os

# --- 코어 로직 임포트 ---
try:
    from src.keyword_generator import generate_keywords_for_topic
    from src.hallucination_detector import RePPLEnhancedIssueSearcher
    from src.hallucination_detection.enhanced_reporting import EnhancedReportGenerator
    from src.hallucination_detection.threshold_manager import ThresholdManager, ConfidenceLevel
    from src.config import config  # .env 파일 로드를 위해 import
    print("✅ 자동 테스트에 필요한 모듈을 성공적으로 불러왔습니다.")
except ImportError as e:
    print(f"❌ 모듈 불러오기 실패: {e}")
    print("스크립트를 프로젝트 최상위 폴더에서 실행하고 있는지 확인해주세요.")
    exit(1)


# --- 💡 테스트 파라미터 💡 ---
TEST_TOPIC = "WWDC"
TEST_PERIOD = "최근 2일"


async def main():
    """자동 테스트의 메인 실행 함수."""
    print("-" * 60)
    print(f"🚀 '{TEST_TOPIC}' 주제에 대한 자동 테스트를 시작합니다.")
    print(f"   (기간: {TEST_PERIOD}, 3단계 환각 탐지 활성화)")
    print("-" * 60)

    # 임계값 관리자 초기화
    threshold_manager = ThresholdManager()
    print(f"\n⚙️ 환각 탐지 임계값 설정:")
    print(f"   - 최소 신뢰도: {threshold_manager.thresholds.min_confidence_threshold:.1%}")
    print(f"   - RePPL: {threshold_manager.thresholds.reppl_threshold:.1%}")
    print(f"   - 자기 일관성: {threshold_manager.thresholds.consistency_threshold:.1%}")
    print(f"   - LLM Judge: {threshold_manager.thresholds.llm_judge_threshold:.1%}")

    # 1. 키워드 생성 단계
    print("\n[1/4] 키워드를 생성합니다...")
    try:
        keyword_result = await generate_keywords_for_topic(TEST_TOPIC)
        print(f"✅ 키워드 생성 완료! (핵심 키워드 {len(keyword_result.primary_keywords)}개)")
        print(f"   - 생성된 키워드 (일부): {', '.join(keyword_result.primary_keywords[:5])}")
    except Exception as e:
        print(f"❌ 키워드 생성 중 오류 발생: {e}")
        return

    # 2. 이슈 검색 및 환각 탐지 단계
    print("\n[2/4] 생성된 키워드로 이슈를 검색하고 3단계 환각 탐지를 수행합니다...")
    print("   - RePPL: 반복성, 퍼플렉시티, 의미적 엔트로피 분석")
    print("   - Self-Consistency: 여러 응답 간 일관성 검증")
    print("   - LLM-as-Judge: GPT-4o를 통한 종합 평가")

    try:
        enhanced_searcher = RePPLEnhancedIssueSearcher(threshold_manager=threshold_manager)
        search_result = await enhanced_searcher.search_with_validation(
            keyword_result,
            TEST_PERIOD
        )

        # 신뢰도 분포 출력
        if hasattr(search_result, 'confidence_distribution'):
            dist = search_result.confidence_distribution
            print(f"\n✅ 이슈 검색 및 검증 완료!")
            print(f"   - 총 이슈: {search_result.total_found}개")
            print(f"   - 높은 신뢰도: {dist['high']}개")
            print(f"   - 중간 신뢰도: {dist['moderate']}개")
            print(f"   - 낮은 신뢰도: {dist['low']}개")
    except Exception as e:
        print(f"❌ 이슈 검색 중 오류 발생: {e}")
        return

    # 3. 검증 결과 상세 출력
    print("\n[3/4] 환각 탐지 결과 분석...")
    if search_result.issues:
        # 상위 3개 이슈의 상세 분석 출력
        for i, issue in enumerate(search_result.issues[:3], 1):
            print(f"\n📄 이슈 {i}: {issue.title[:50]}...")

            confidence = getattr(issue, 'hallucination_confidence', 0.0)
            level = threshold_manager.classify_confidence(confidence)

            print(f"   - 최종 신뢰도: {confidence:.1%} ({level.value})")

            # 개별 탐지기 점수 출력
            analysis = getattr(issue, 'hallucination_analysis', None)
            if analysis:
                for method, score in analysis.individual_scores.items():
                    print(f"   - {method}: {score.confidence:.1%}")

    # 4. 보고서 생성 및 저장
    print("\n[4/4] 최종 보고서를 생성하고 로컬에 저장합니다...")
    report_generator = EnhancedReportGenerator(threshold_manager)

    try:
        # 상세 보고서 생성
        detailed_report = report_generator.generate_detailed_report(search_result)
        file_path = report_generator.save_report_to_file(detailed_report, TEST_TOPIC)

        print(f"✅ 보고서 저장을 완료했습니다!")
        print(f"   - 파일 위치: {file_path}")
        print(f"   - 파일 크기: {os.path.getsize(file_path):,} bytes")

        # 보고서 요약 정보
        high, moderate, low = threshold_manager.filter_issues_by_confidence(search_result.issues)
        avg_confidence = sum(
            getattr(issue, 'hallucination_confidence', 0.0)
            for issue in search_result.issues
        ) / len(search_result.issues) if search_result.issues else 0.0

        print(f"\n📊 보고서 요약:")
        print(f"   - 전체 평균 신뢰도: {avg_confidence:.1%}")
        print(f"   - 권장 사항: {threshold_manager.get_confidence_summary(avg_confidence)['recommendation']}")

    except Exception as e:
        print(f"❌ 보고서 저장 중 오류 발생: {e}")

    print("\n" + "-" * 60)
    print("🎉 자동 테스트가 모두 종료되었습니다.")
    print("-" * 60)


if __name__ == "__main__":
    # 스크립트를 직접 실행할 때 이 함수가 호출됩니다.
    asyncio.run(main())