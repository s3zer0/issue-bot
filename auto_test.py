"""자동화된 E2E(End-to-End) 테스트 스크립트.

이 스크립트는 Discord 봇을 직접 실행하지 않고,
키워드 생성부터 이슈 검색, 환각 탐지, 보고서 생성까지의
핵심 기능 전체 흐름을 순차적으로 실행하여 검증합니다.
"""

import asyncio
import os
from datetime import datetime

# --- 코어 로직 임포트 ---
try:
    from src.keyword_generator import generate_keywords_for_topic
    from src.hallucination_detector import RePPLEnhancedIssueSearcher
    from src.reporting import create_detailed_report_from_search_result, save_report_to_file
    from src.config import config  # .env 파일 로드를 위해 import
    print("✅ 자동 테스트에 필요한 모듈을 성공적으로 불러왔습니다.")
except ImportError as e:
    print(f"❌ 모듈 불러오기 실패: {e}")
    print("스크립트를 프로젝트 최상위 폴더에서 실행하고 있는지 확인해주세요.")
    exit(1)


# --- 💡 테스트 파라미터 💡 ---
# 여기에서 테스트하고 싶은 주제와 기간을 자유롭게 수정하세요.
TEST_TOPIC = "클라우드 네이티브"
TEST_PERIOD = "최근 1주일"


async def main():
    """자동 테스트의 메인 실행 함수."""
    print("-" * 60)
    print(f"🚀 '{TEST_TOPIC}' 주제에 대한 자동 테스트를 시작합니다. (환각 탐지 활성화)")
    print(f"   (기간: {TEST_PERIOD})")
    print("-" * 60)

    # 1. 키워드 생성 단계
    print("\n[1/3] 키워드를 생성합니다...")
    try:
        keyword_result = await generate_keywords_for_topic(TEST_TOPIC)
        print(f"✅ 키워드 생성 완료! (핵심 키워드 {len(keyword_result.primary_keywords)}개)")
        print(f"   - 생성된 키워드 (일부): {', '.join(keyword_result.primary_keywords[:5])}")
    except Exception as e:
        print(f"❌ 키워드 생성 중 오류 발생: {e}")
        return

    # 2. 이슈 검색 및 환각 탐지 단계
    print("\n[2/3] 생성된 키워드로 이슈를 검색하고 환각 탐지를 수행합니다...")
    try:
        enhanced_searcher = RePPLEnhancedIssueSearcher()
        search_result = await enhanced_searcher.search_with_validation(
            keyword_result,
            TEST_PERIOD
        )
        print(f"✅ 이슈 검색 및 검증 완료! (신뢰도 높은 이슈 {search_result.total_found}개 발견)")
    except Exception as e:
        print(f"❌ 이슈 검색 중 오류 발생: {e}")
        return

    # 3. 보고서 생성 및 로컬 저장 단계
    print("\n[3/3] 최종 보고서를 생성하고 로컬에 저장합니다...")
    # 상세 분석된 이슈가 있을 경우에만 보고서 생성
    if search_result.total_found > 0 and search_result.detailed_issues_count > 0:
        try:
            report_content = create_detailed_report_from_search_result(search_result)
            file_path = save_report_to_file(report_content, TEST_TOPIC)

            print(f"✅ 보고서 저장을 완료했습니다!")
            print(f"   - 파일 위치: {file_path}")
        except Exception as e:
            print(f"❌ 보고서 저장 중 오류 발생: {e}")
    else:
        print("ℹ️ 신뢰도가 높은 상세 분석 이슈가 없어 보고서를 생성하지 않았습니다.")

    print("\n" + "-" * 60)
    print("🎉 자동 테스트가 모두 종료되었습니다.")
    print("-" * 60)


if __name__ == "__main__":
    # 스크립트를 직접 실행할 때 이 함수가 호출됩니다.
    asyncio.run(main())