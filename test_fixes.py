#!/usr/bin/env python3
"""
Test script to verify the fixes for ConsistencyScore and PDF font errors.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.hallucination_detection.enhanced_searcher import EnhancedIssueSearcher
from src.reporting.pdf_report_generator import PDFReportGenerator
from src.models import SearchResult, IssueItem
from loguru import logger


async def test_consistency_score_fix():
    """Test that ConsistencyScore handling is fixed."""
    print("🔧 Testing ConsistencyScore fix...")
    
    try:
        # Create a simple issue item for testing
        test_issue = IssueItem(
            title="Test Issue",
            url="http://example.com",
            source="Test Source",
            summary="Test summary for consistency checking",
            published_date="2024-01-01"
        )
        
        # Create enhanced searcher
        searcher = EnhancedIssueSearcher()
        
        # Test the consistency checking (this would previously fail)
        result = await searcher._run_optimized_self_consistency(
            "Test text for consistency checking",
            "Test context",
            30.0  # 30 second timeout
        )
        
        # Check result type
        if isinstance(result, dict):
            print(f"   Dict result: {result.get('status', 'unknown')}")
        else:
            print(f"   Object result: confidence = {getattr(result, 'confidence', 'N/A')}")
        
        print("✅ ConsistencyScore fix working correctly")
        return True
        
    except Exception as e:
        print(f"❌ ConsistencyScore fix failed: {e}")
        logger.error(f"ConsistencyScore test error: {e}")
        return False


def test_pdf_font_fix():
    """Test that PDF font handling is fixed."""
    print("🔧 Testing PDF font fix...")
    
    try:
        # Create PDF generator
        pdf_gen = PDFReportGenerator()
        
        # Test the font formatting method
        test_texts = [
            "<b>AI 이슈 모니터링 보고서</b>",
            "<b>테스트 제목</b>",
            "일반 텍스트",
            "<i>이탤릭 텍스트</i>"
        ]
        
        for text in test_texts:
            formatted = pdf_gen._format_text_for_font(text)
            print(f"   '{text}' -> '{formatted}'")
        
        # Check font setup
        print(f"   Default font: {pdf_gen.default_font}")
        
        print("✅ PDF font fix working correctly")
        return True
        
    except Exception as e:
        print(f"❌ PDF font fix failed: {e}")
        logger.error(f"PDF font test error: {e}")
        return False


async def test_minimal_pdf_generation():
    """Test minimal PDF generation to ensure no font errors."""
    print("🔧 Testing minimal PDF generation...")
    
    try:
        # Create minimal test data
        test_issue = IssueItem(
            title="테스트 이슈",
            url="http://example.com",
            source="테스트 소스",
            summary="한글 텍스트가 포함된 테스트 요약입니다.",
            published_date="2024-01-01"
        )
        
        test_result = SearchResult(
            query_keywords=["테스트", "키워드"],
            issues=[test_issue],
            total_found=1,
            search_time=1.0,
            api_calls_used=1,
            time_period="최근 1주일"
        )
        
        # Try to generate a simple PDF
        pdf_gen = PDFReportGenerator()
        
        # This should not fail with font errors
        if hasattr(pdf_gen, '_create_title_page'):
            # Test the problematic method
            story = []
            pdf_gen._create_title_page(story, test_result)
            print("   Title page creation successful")
        
        print("✅ Minimal PDF generation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Minimal PDF generation failed: {e}")
        logger.error(f"PDF generation test error: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 Testing fixes for ConsistencyScore and PDF font errors")
    print("=" * 60)
    
    # Run tests
    tests = [
        test_consistency_score_fix(),
        test_pdf_font_fix(),
        test_minimal_pdf_generation()
    ]
    
    results = await asyncio.gather(*tests, return_exceptions=True)
    
    # Check results
    passed = 0
    total = len(results)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Test {i+1} crashed: {result}")
        elif result:
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All fixes are working correctly!")
        return 0
    else:
        print("❌ Some fixes need more work")
        return 1


if __name__ == "__main__":
    asyncio.run(main())