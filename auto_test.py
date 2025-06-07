# auto_test.py

import asyncio
import os
from datetime import datetime

# --- 코어 로직 임포트 ---
# 봇의 핵심 기능을 가져옵니다.
try:
    from src.keyword_generator import generate_keywords_for_topic
    from src.issue_searcher import search_issues_for_keywords, create_detailed_report_from_search_result
    from src.config import config # .env 파일 로드를 위해 import
    print("✅ 자동 테스트에 필요한 모듈을 성공적으로 불러왔습니다.")
except ImportError as e:
    print(f"❌ 모듈 불러오기 실패: {e}")
    print("스크립트를 프로젝트 최상위 폴더에서 실행하고 있는지 확인해주세요.")
    exit(1)

# --- 💡 테스트 파라미터 💡 ---
# 여기에서 테스트하고 싶은 주제와 기간을 자유롭게 수정하세요.
TEST_TOPIC = "사이버 보안"
TEST_PERIOD = "최근 1주일"
COLLECT_DETAILS = True # 세부 분석 실행 여부 (True/False)

async def main():
    """자동 테스트의 메인 실행 함수"""
    print("-" * 60)
    print(f"🚀 '{TEST_TOPIC}' 주제에 대한 자동 테스트를 시작합니다.")
    print(f"   (기간: {TEST_PERIOD}, 세부분석: {'O' if COLLECT_DETAILS else 'X'})")
    print("-" * 60)

    # 1. 키워드 생성
    print("\n[1/3] 키워드를 생성합니다...")
    try:
        keyword_result = await generate_keywords_for_topic(TEST_TOPIC)
        print(f"✅ 키워드 생성 완료! (핵심 키워드 {len(keyword_result.primary_keywords)}개)")
        print(f"   - 생성된 키워드 (일부): {', '.join(keyword_result.primary_keywords[:5])}")
    except Exception as e:
        print(f"❌ 키워드 생성 중 오류 발생: {e}")
        return

    # 2. 이슈 검색
    print("\n[2/3] 생성된 키워드로 이슈를 검색합니다...")
    try:
        search_result = await search_issues_for_keywords(
            keyword_result,
            TEST_PERIOD,
            collect_details=COLLECT_DETAILS
        )
        print(f"✅ 이슈 검색 완료! (총 {search_result.total_found}개 발견)")
    except Exception as e:
        print(f"❌ 이슈 검색 중 오류 발생: {e}")
        return

    # 3. 보고서 생성 및 로컬에 저장
    print("\n[3/3] 최종 보고서를 생성하고 로컬에 저장합니다...")
    if search_result.total_found > 0 and COLLECT_DETAILS:
        try:
            report_content = create_detailed_report_from_search_result(search_result)

            # 'reports' 폴더가 없으면 생성
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)

            # 파일명 설정 및 저장
            filename = f"auto_report_{TEST_TOPIC.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            file_path = os.path.join(reports_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            print(f"✅ 보고서 저장을 완료했습니다!")
            print(f"   - 파일 위치: {file_path}")
        except Exception as e:
            print(f"❌ 보고서 저장 중 오류 발생: {e}")
    elif not COLLECT_DETAILS:
        print("ℹ️ 세부 분석(COLLECT_DETAILS)이 비활성화되어 보고서를 생성하지 않았습니다.")
    else:
        print("ℹ️ 발견된 이슈가 없어 보고서를 생성하지 않았습니다.")

    print("\n" + "-" * 60)
    print("🎉 자동 테스트가 모두 종료되었습니다.")
    print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())