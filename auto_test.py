"""자동화된 E2E(End-to-End) 테스트 스크립트

이 스크립트는 Discord 봇을 직접 실행하지 않고,
키워드 생성부터 이슈 검색, 환각 탐지, 보고서 생성까지의
핵심 기능 전체 흐름을 순차적으로 실행하여 검증합니다.
마크다운과 PDF 보고서를 모두 생성합니다.
"""

import asyncio
import os
import sys

# --- 코어 로직 임포트 ---
try:
    from src.keyword_generator import generate_keywords_for_topic
    from src.hallucination_detector import RePPLEnhancedIssueSearcher
    from src.hallucination_detection.enhanced_reporting_with_pdf import (
        EnhancedReportGenerator, generate_all_reports
    )
    from src.hallucination_detection.threshold_manager import ThresholdManager, ConfidenceLevel
    from src.pdf_report_generator import PDFReportGenerator
    from src.config import config  # .env 파일 로드를 위해 import
    print("✅ 자동 테스트에 필요한 모듈을 성공적으로 불러왔습니다.")
except ImportError as e:
    print(f"❌ 모듈 불러오기 실패: {e}")
    print("스크립트를 프로젝트 최상위 폴더에서 실행하고 있는지 확인해주세요.")
    print("\n필요한 패키지가 설치되어 있는지 확인하세요:")
    print("pip install reportlab pillow")
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

    # --- ✨ 수정된 부분: 변수 초기화 ✨ ---
    # 오류 발생 시 UnboundLocalError를 방지하기 위해 변수를 미리 None으로 설정합니다.
    markdown_path = None
    pdf_path = None

    # OpenAI API 키 확인
    has_openai_key = config.get_openai_api_key() is not None
    if has_openai_key:
        print("✅ OpenAI API 키 감지됨 - PDF 보고서 생성이 가능합니다.")
    else:
        print("⚠️  OpenAI API 키가 없음 - 마크다운 보고서만 생성됩니다.")
        print("   PDF 보고서를 원하시면 .env 파일에 OPENAI_API_KEY를 설정하세요.")

    # 임계값 관리자 초기화
    threshold_manager = ThresholdManager()
    print(f"\n⚙️ 환각 탐지 임계값 설정:")
    print(f"   - 최소 신뢰도: {threshold_manager.thresholds.min_confidence_threshold:.1%}")
    print(f"   - RePPL: {threshold_manager.thresholds.reppl_threshold:.1%}")
    print(f"   - 자기 일관성: {threshold_manager.thresholds.consistency_threshold:.1%}")
    print(f"   - LLM Judge: {threshold_manager.thresholds.llm_judge_threshold:.1%}")

    # 1. 키워드 생성 단계
    print("\n[1/5] 키워드를 생성합니다...")
    try:
        keyword_result = await generate_keywords_for_topic(TEST_TOPIC)
        print(f"✅ 키워드 생성 완료! (핵심 키워드 {len(keyword_result.primary_keywords)}개)")
        print(f"   - 생성된 키워드 (일부): {', '.join(keyword_result.primary_keywords[:5])}")
    except Exception as e:
        print(f"❌ 키워드 생성 중 오류 발생: {e}")
        return

    # 2. 이슈 검색 및 환각 탐지 단계
    print("\n[2/5] 생성된 키워드로 이슈를 검색하고 3단계 환각 탐지를 수행합니다...")
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
    print("\n[3/5] 환각 탐지 결과 분석...")
    if search_result.issues:
        # 상위 3개 이슈의 상세 분석 출력
        for i, issue in enumerate(search_result.issues[:3], 1):
            print(f"\n📄 이슈 {i}: {issue.title[:50]}...")

            confidence = getattr(issue, 'combined_confidence', 0.5)
            level = threshold_manager.classify_confidence(confidence)

            print(f"   - 최종 신뢰도: {confidence:.1%} ({level.value})")

            # 개별 탐지기 점수 출력
            if hasattr(issue, 'hallucination_score') and issue.hallucination_score:
                score = issue.hallucination_score
                if hasattr(score, 'reppl_score') and score.reppl_score:
                    print(f"   - RePPL: {score.reppl_score.confidence:.1%}")
                if hasattr(score, 'consistency_score') and score.consistency_score:
                    print(f"   - 자기 일관성: {score.consistency_score.confidence:.1%}")
                if hasattr(score, 'llm_judge_score') and score.llm_judge_score:
                    print(f"   - LLM Judge: {score.llm_judge_score.confidence:.1%}")

    # 4. 마크다운 보고서 생성
    print("\n[4/5] 마크다운 보고서를 생성합니다...")
    try:
        # 통합 보고서 생성 함수 사용
        result_embed, markdown_path, pdf_path = await generate_all_reports(
            search_result,
            TEST_TOPIC,
            generate_pdf=has_openai_key
        )

        print(f"✅ 마크다운 보고서 저장 완료!")
        print(f"   - 파일 위치: {markdown_path}")
        print(f"   - 파일 크기: {os.path.getsize(markdown_path):,} bytes")

        # 보고서 요약 정보
        high, moderate, low = threshold_manager.filter_issues_by_confidence(search_result.issues)
        avg_confidence = sum(
            getattr(issue, 'combined_confidence', 0.5)
            for issue in search_result.issues
        ) / len(search_result.issues) if search_result.issues else 0.0

        print(f"\n📊 보고서 요약:")
        print(f"   - 전체 평균 신뢰도: {avg_confidence:.1%}")
        confidence_level = threshold_manager.classify_confidence(avg_confidence)
        print(f"   - 신뢰도 등급: {confidence_level.value}")

    except Exception as e:
        print(f"❌ 마크다운 보고서 저장 중 오류 발생: {e}")
        # pdf_path는 이미 None으로 초기화되어 있으므로 별도 처리가 필요 없습니다.

    # 5. PDF 보고서 생성 (가능한 경우)
    if has_openai_key and pdf_path:
        print("\n[5/5] PDF 보고서 생성 완료!")
        try:
            print(f"✅ LLM으로 개선된 PDF 보고서가 생성되었습니다!")
            print(f"   - 파일 위치: {pdf_path}")
            print(f"   - 파일 크기: {os.path.getsize(pdf_path):,} bytes")
            print(f"\n💡 PDF 특징:")
            print(f"   - OpenAI GPT-4를 사용한 비즈니스 인사이트 추가")
            print(f"   - 전문적인 레이아웃과 디자인")
            print(f"   - 경영진 요약 및 권장 조치사항 포함")
        except Exception as e:
            print(f"⚠️  PDF 파일 정보 확인 중 오류: {e}")
    elif has_openai_key:
        print("\n[5/5] PDF 보고서 생성을 건너뜁니다 (생성 실패)")
    else:
        print("\n[5/5] PDF 보고서 생성을 건너뜁니다 (OpenAI API 키 없음)")

    print("\n" + "-" * 60)
    print("🎉 자동 테스트가 모두 종료되었습니다.")

    # 생성된 파일 목록 (이제 오류 없이 안전하게 출력됩니다)
    print("\n📁 생성된 파일:")
    print(f"   - 마크다운: {markdown_path}")
    if pdf_path:
        print(f"   - PDF: {pdf_path}")

    print("\n💡 팁: reports/ 폴더에서 생성된 보고서를 확인하세요.")
    print("-" * 60)


if __name__ == "__main__":
    # 스크립트를 직접 실행할 때 이 함수가 호출됩니다.
    asyncio.run(main())
