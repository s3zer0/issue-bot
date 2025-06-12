"""
PDF 보고서 생성 모듈.

이슈 분석 결과를 LLM으로 개선하고 깔끔한 PDF 형식으로 변환합니다.
reportlab 라이브러리를 사용하여 전문적인 디자인의 PDF 보고서를 생성합니다.
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from io import BytesIO
import asyncio

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image, KeepTogether, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from loguru import logger
import openai

from src.models import SearchResult, IssueItem
from src.config import config
from src.hallucination_detection.threshold_manager import ThresholdManager
from src.reporting.markdown_parser import MarkdownToPDFConverter
from src.reporting.topic_classifier import TopicClassifier


class PDFReportGenerator:
    """LLM을 활용한 PDF 보고서 생성기."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        PDF 보고서 생성기를 초기화합니다.

        Args:
            openai_api_key: OpenAI API 키 (없으면 환경 변수에서 가져옴)
        """
        self.api_key = openai_api_key or config.get_openai_api_key()
        self.threshold_manager = ThresholdManager()

        # 한글 폰트 설정 (NanumGothic 또는 시스템 폰트 사용)
        self._setup_fonts()

        # Performance: Initialize markdown converter for better text formatting
        self.markdown_converter = MarkdownToPDFConverter(self.default_font)
        
        # Dynamic: Initialize topic classifier for adaptive sections
        self.topic_classifier = TopicClassifier()

        # 스타일 설정
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

        logger.info("PDF 보고서 생성기 초기화 완료")

    def _setup_fonts(self):
        """한글 지원을 위한 폰트 설정 - NotoSansKR 사용."""
        try:
            # Use absolute path to Fonts directory
            font_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "Fonts",
                "NotoSansKR-VariableFont_wght.ttf"
            )
            
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('NotoSansKR', font_path))
                self.default_font = 'NotoSansKR'
                logger.info(f"Successfully registered NotoSansKR font from: {font_path}")
            else:
                # Fallback for different OS
                if os.name == 'nt':
                    # Windows
                    malgun = "C:/Windows/Fonts/malgun.ttf"
                    if os.path.exists(malgun):
                        pdfmetrics.registerFont(TTFont('MalgunGothic', malgun))
                        self.default_font = 'MalgunGothic'
                        logger.info("Using MalgunGothic font on Windows")
                    else:
                        self.default_font = 'Helvetica'
                        logger.warning("Korean font not found on Windows, using Helvetica")
                else:
                    # macOS/Linux
                    nanum = "/Library/Fonts/NanumGothic.ttf"
                    if os.path.exists(nanum):
                        pdfmetrics.registerFont(TTFont('NanumGothic', nanum))
                        self.default_font = 'NanumGothic'
                        logger.info("Using NanumGothic font")
                    else:
                        # Last resort - use Helvetica (no Korean support)
                        self.default_font = 'Helvetica'
                        logger.warning("Korean font not found, using Helvetica (Korean text may not display properly)")
                        
        except Exception as e:
            logger.error(f"Font registration error: {e}")
            self.default_font = 'Helvetica'
    
    def _format_text_for_font(self, text: str) -> str:
        """폰트 호환성에 따라 텍스트 포맷팅 조정."""
        # NotoSansKR and other TTF fonts support bold/italic, so no need to strip tags
        # Only strip tags if using Helvetica as fallback
        if self.default_font == 'Helvetica':
            # Helvetica doesn't support Korean, so we might want to warn about this
            if any('\u3131' <= char <= '\u318e' or '\uac00' <= char <= '\ud7a3' for char in text):
                logger.warning("Korean text detected but Korean font not available")
        return text

    def _setup_custom_styles(self):
        """커스텀 스타일 정의."""
        # 제목 스타일
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontName=self.default_font,
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        # 부제목 스타일
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontName=self.default_font,
            fontSize=18,
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=20,
            spaceAfter=15,
            borderColor=colors.HexColor('#3498db'),
            borderWidth=2,
            borderPadding=5,
            borderRadius=3
        ))

        # 소제목 스타일
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontName=self.default_font,
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceBefore=15,
            spaceAfter=10
        ))

        # 본문 스타일
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontName=self.default_font,
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#2c3e50')
        ))

        # 강조 스타일
        self.styles.add(ParagraphStyle(
            name='CustomEmphasis',
            parent=self.styles['BodyText'],
            fontName=self.default_font,
            fontSize=11,
            leading=16,
            textColor=colors.HexColor('#e74c3c'),
            alignment=TA_LEFT
        ))

    async def enhance_content_with_llm(self, search_result: SearchResult) -> Dict[str, Any]:
        """
        LLM을 사용하여 보고서 내용을 개선하고 구조화합니다.

        Args:
            search_result: 검색 결과 객체

        Returns:
            Dict: LLM이 개선한 보고서 구조
        """
        if not self.api_key:
            logger.warning("OpenAI API 키가 없어 LLM 개선을 건너뜁니다.")
            return self._create_basic_structure(search_result)

        try:
            client = openai.AsyncOpenAI(api_key=self.api_key)

            # 프롬프트 구성
            prompt = f"""
당신은 전문적인 기술 보고서 작성자입니다. 다음 이슈 모니터링 결과를 바탕으로 
경영진을 위한 깔끔하고 인사이트 있는 보고서를 작성해주세요.

주제: {', '.join(search_result.query_keywords)}
기간: {search_result.time_period}
발견된 이슈 수: {search_result.total_found}

주요 이슈들:
"""
            # 상위 5개 이슈 요약
            for i, issue in enumerate(search_result.issues[:5], 1):
                prompt += f"\n{i}. {issue.title}\n   - {issue.summary}\n   - 신뢰도: {getattr(issue, 'combined_confidence', 0.5):.1%}\n"

            prompt += """
다음 형식으로 보고서를 작성해주세요:

1. 핵심 요약 (Executive Summary): 3-4문장으로 전체 상황을 요약
2. 주요 발견사항 (Key Findings): 가장 중요한 3-5개 인사이트를 bullet point로
3. 트렌드 분석 (Trend Analysis): 발견된 이슈들에서 나타나는 패턴이나 트렌드
4. 리스크 및 기회 (Risks & Opportunities): 비즈니스 관점에서의 리스크와 기회
5. 권장 조치사항 (Recommended Actions): 구체적인 액션 아이템 3-5개

각 섹션은 간결하면서도 인사이트가 있어야 합니다. 
전문 용어는 필요한 경우에만 사용하고, 비즈니스 임팩트를 중심으로 작성하세요.
"""

            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 기술 트렌드를 비즈니스 인사이트로 변환하는 전문가입니다."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )

            # LLM 응답 파싱
            enhanced_content = self._parse_llm_response(
                response.choices[0].message.content,
                search_result
            )

            logger.info("LLM을 통한 보고서 내용 개선 완료")
            return enhanced_content

        except Exception as e:
            logger.error(f"LLM 개선 중 오류 발생: {e}")
            return self._create_basic_structure(search_result)

    def _parse_llm_response(self, llm_response: str, search_result: SearchResult) -> Dict[str, Any]:
        """LLM 응답을 파싱하여 구조화된 데이터로 변환."""
        sections = {
            'executive_summary': '',
            'key_findings': [],
            'trend_analysis': '',
            'risks_opportunities': '',
            'recommended_actions': []
        }

        current_section = None
        current_content = []
        lines = llm_response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 섹션 헤더 감지 (더 유연한 매칭)
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['핵심 요약', 'executive summary']):
                if current_section and current_content:
                    # 이전 섹션 저장
                    self._save_section_content(sections, current_section, current_content)
                current_section = 'executive_summary'
                current_content = []
                continue
            elif any(keyword in line_lower for keyword in ['주요 발견', 'key findings']):
                if current_section and current_content:
                    self._save_section_content(sections, current_section, current_content)
                current_section = 'key_findings'
                current_content = []
                continue
            elif any(keyword in line_lower for keyword in ['트렌드 분석', 'trend analysis']):
                if current_section and current_content:
                    self._save_section_content(sections, current_section, current_content)
                current_section = 'trend_analysis'
                current_content = []
                continue
            elif any(keyword in line_lower for keyword in ['리스크', 'risks', '기회', 'opportunities']):
                if current_section and current_content:
                    self._save_section_content(sections, current_section, current_content)
                current_section = 'risks_opportunities'
                current_content = []
                continue
            elif any(keyword in line_lower for keyword in ['권장 조치', 'recommended actions']):
                if current_section and current_content:
                    self._save_section_content(sections, current_section, current_content)
                current_section = 'recommended_actions'
                current_content = []
                continue

            # 내용 추가
            if current_section:
                current_content.append(line)

        # 마지막 섹션 저장
        if current_section and current_content:
            self._save_section_content(sections, current_section, current_content)

        # 기본 구조와 병합
        return {
            'search_result': search_result,
            'enhanced_sections': sections,
            'metadata': {
                'generated_at': datetime.now(),
                'llm_enhanced': True
            }
        }

    def _save_section_content(self, sections: Dict, section_name: str, content_lines: List[str]):
        """섹션 내용을 저장하는 헬퍼 메서드."""
        if section_name in ['key_findings', 'recommended_actions']:
            # 리스트 형태로 저장
            for line in content_lines:
                clean_line = line.lstrip('-•* 0123456789.')
                if clean_line:
                    sections[section_name].append(clean_line)
        else:
            # 텍스트 형태로 저장
            sections[section_name] = ' '.join(content_lines)

    def _create_basic_structure(self, search_result: SearchResult) -> Dict[str, Any]:
        """LLM 없이 기본 보고서 구조 생성."""
        # 신뢰도별 이슈 분류
        high, moderate, low = self.threshold_manager.filter_issues_by_confidence(
            search_result.issues
        )

        # 주요 발견사항 생성
        key_findings = []
        for issue in high[:3]:  # 상위 3개 높은 신뢰도 이슈
            key_findings.append(f"{issue.title}: {issue.summary[:100]}...")

        # 발견사항이 없으면 기본 메시지 추가
        if not key_findings:
            key_findings = ["신뢰도 높은 이슈가 발견되지 않았습니다."]

        return {
            'search_result': search_result,
            'enhanced_sections': {
                'executive_summary': f"{len(search_result.issues)}개의 이슈가 발견되었으며, "
                                   f"이 중 {len(high)}개가 높은 신뢰도를 보였습니다.",
                'key_findings': key_findings,
                'trend_analysis': '수집된 이슈들을 분석한 결과입니다.',
                'risks_opportunities': '추가 분석이 필요합니다.',
                'recommended_actions': ['지속적인 모니터링 필요', '주요 이슈 심층 분석 권장']
            },
            'metadata': {
                'generated_at': datetime.now(),
                'llm_enhanced': False
            }
        }

    def generate_pdf(self, enhanced_data: Dict[str, Any], output_path: str) -> str:
        """
        PDF 보고서를 생성합니다.

        Args:
            enhanced_data: LLM으로 개선된 보고서 데이터
            output_path: 출력 파일 경로

        Returns:
            str: 생성된 PDF 파일 경로
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        # 문서 요소 리스트
        story = []

        # 표지
        story.extend(self._create_cover_page(enhanced_data))
        story.append(PageBreak())

        # 목차
        story.extend(self._create_table_of_contents())
        story.append(PageBreak())

        # 핵심 요약
        story.extend(self._create_executive_summary(enhanced_data))
        story.append(PageBreak())

        # Dynamic: Generate sections based on topic classification
        story.extend(self._create_dynamic_sections(enhanced_data))

        # 부록
        story.extend(self._create_appendix(enhanced_data))

        # PDF 생성
        doc.build(story, onFirstPage=self._add_page_number, onLaterPages=self._add_page_number)

        logger.info(f"PDF 보고서 생성 완료: {output_path}")
        return output_path

    def _create_cover_page(self, data: Dict[str, Any]) -> List:
        """표지 페이지 생성."""
        story = []
        search_result = data['search_result']

        # 여백
        story.append(Spacer(1, 3*inch))

        # 제목
        title = Paragraph(
            self._format_text_for_font(f"<b>AI 이슈 모니터링 보고서</b>"),
            self.styles['CustomTitle']
        )
        story.append(title)

        story.append(Spacer(1, 0.5*inch))

        # 부제목
        subtitle = Paragraph(
            self._format_text_for_font(f"<b>{', '.join(search_result.query_keywords[:3])}</b>"),
            ParagraphStyle(
                'Subtitle',
                parent=self.styles['CustomTitle'],
                fontSize=18,
                textColor=colors.HexColor('#7f8c8d')
            )
        )
        story.append(subtitle)

        story.append(Spacer(1, 2*inch))

        # 메타정보 테이블
        meta_data = [
            ['생성일시', datetime.now().strftime('%Y년 %m월 %d일 %H:%M')],
            ['검색 기간', search_result.time_period],
            ['발견 이슈', f'{search_result.total_found}개'],
            ['신뢰도 평가', 'AI 환각 탐지 시스템 적용']
        ]

        meta_table = Table(meta_data, colWidths=[3*inch, 3*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.default_font),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#7f8c8d')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor('#ecf0f1')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))

        story.append(meta_table)

        return story

    def _create_table_of_contents(self) -> List:
        """목차 생성."""
        story = []

        story.append(Paragraph("목차", self.styles['CustomHeading1']))
        story.append(Spacer(1, 0.3*inch))

        toc_items = [
            ("1. 핵심 요약", "3"),
            ("2. 주요 발견사항", "4"),
            ("3. 상세 이슈 분석", "5"),
            ("4. 트렌드 분석", "8"),
            ("5. 리스크 및 기회", "9"),
            ("6. 권장 조치사항", "10"),
            ("부록. 환각 탐지 상세 결과", "11")
        ]

        for item, page in toc_items:
            # 링크 없이 단순 텍스트로 생성
            toc_line = Paragraph(
                f'{item}{"." * (50 - len(item))}{page}',
                ParagraphStyle(
                    'TOCItem',
                    parent=self.styles['CustomBody'],
                    leftIndent=20,
                    rightIndent=20
                )
            )
            story.append(toc_line)
            story.append(Spacer(1, 0.1*inch))

        return story

    def _create_executive_summary(self, data: Dict[str, Any]) -> List:
        """핵심 요약 섹션 생성."""
        story = []
        sections = data['enhanced_sections']

        story.append(Paragraph("1. 핵심 요약", self.styles['CustomHeading1']))
        story.append(Spacer(1, 0.2*inch))

        summary_text = sections.get('executive_summary', '요약 정보가 없습니다.')
        story.append(Paragraph(summary_text, self.styles['CustomBody']))

        story.append(Spacer(1, 0.3*inch))

        # 핵심 지표 테이블
        search_result = data['search_result']
        high, moderate, low = self.threshold_manager.filter_issues_by_confidence(
            search_result.issues
        )

        metrics_data = [
            ['구분', '수치', '비율'],
            ['전체 이슈', f'{search_result.total_found}개', '100%'],
            ['높은 신뢰도', f'{len(high)}개', f'{len(high)/max(1, search_result.total_found)*100:.1f}%'],
            ['중간 신뢰도', f'{len(moderate)}개', f'{len(moderate)/max(1, search_result.total_found)*100:.1f}%'],
            ['낮은 신뢰도', f'{len(low)}개', f'{len(low)/max(1, search_result.total_found)*100:.1f}%']
        ]

        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), self.default_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ]))

        story.append(metrics_table)

        return story

    def _create_key_findings(self, data: Dict[str, Any]) -> List:
        """주요 발견사항 섹션 생성."""
        story = []
        sections = data['enhanced_sections']

        story.append(Paragraph("2. 주요 발견사항", self.styles['CustomHeading1']))
        story.append(Spacer(1, 0.2*inch))

        findings = sections.get('key_findings', [])
        if findings:
            for i, finding in enumerate(findings, 1):
                bullet = Paragraph(f"• {finding}", self.styles['CustomBody'])
                story.append(bullet)
                story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph("주요 발견사항이 없습니다.", self.styles['CustomBody']))

        story.append(Spacer(1, 0.3*inch))
        return story

    def _create_detailed_issues(self, data: Dict[str, Any]) -> List:
        """상세 이슈 분석 섹션 생성."""
        story = []
        search_result = data['search_result']

        story.append(Paragraph("3. 상세 이슈 분석", self.styles['CustomHeading1']))
        story.append(Spacer(1, 0.2*inch))

        # 신뢰도별로 이슈 분류
        high, moderate, low = self.threshold_manager.filter_issues_by_confidence(
            search_result.issues
        )

        # 높은 신뢰도 이슈
        if high:
            story.append(Paragraph("높은 신뢰도 이슈", self.styles['CustomHeading2']))
            for issue in high[:5]:  # 상위 5개만
                story.extend(self._create_issue_detail(issue))

        # 중간 신뢰도 이슈
        if moderate and len(story) < 100:  # 페이지 수 제한
            story.append(Paragraph("중간 신뢰도 이슈", self.styles['CustomHeading2']))
            for issue in moderate[:3]:  # 상위 3개만
                story.extend(self._create_issue_detail(issue))

        return story

    def _create_issue_detail(self, issue: IssueItem) -> List:
        """개별 이슈 상세 정보 생성 (Enhanced with markdown parsing)."""
        story = []

        # Enhanced: Use markdown converter for title formatting
        title_markdown = f"## {issue.title}"
        title_elements = self.markdown_converter.convert_to_pdf_elements(title_markdown)
        story.extend(title_elements)

        # Enhanced: Visual confidence indicator with color coding
        confidence = getattr(issue, 'combined_confidence', 0.5)
        confidence_indicator = self.markdown_converter.format_confidence_text(confidence)
        
        meta_info = f"**출처**: {issue.source} | **발행일**: {issue.published_date or 'N/A'} | **신뢰도**: {confidence_indicator}"
        meta_elements = self.markdown_converter.convert_to_pdf_elements(meta_info)
        story.extend(meta_elements)
        
        story.append(Spacer(1, 0.1*inch))

        # Enhanced: Parse summary as markdown for better formatting
        summary_markdown = f"### 요약\n{issue.summary}"
        summary_elements = self.markdown_converter.convert_to_pdf_elements(summary_markdown)
        story.extend(summary_elements)

        # Enhanced: Parse detailed content as markdown (if exists)
        if issue.detailed_content:
            story.append(Spacer(1, 0.1*inch))
            
            # Truncate if too long but preserve markdown structure
            detailed_content = issue.detailed_content
            if len(detailed_content) > 1000:
                # Find a good break point (end of paragraph or sentence)
                truncate_point = 1000
                for punct in ['\n\n', '\n', '. ', '? ', '! ']:
                    last_punct = detailed_content.rfind(punct, 0, 1000)
                    if last_punct > 800:  # Must be reasonably long
                        truncate_point = last_punct + len(punct)
                        break
                detailed_content = detailed_content[:truncate_point] + "\n\n*[내용이 너무 길어 일부만 표시됩니다...]*"
            
            detail_markdown = f"### 상세 내용\n{detailed_content}"
            detail_elements = self.markdown_converter.convert_to_pdf_elements(detail_markdown)
            story.extend(detail_elements)

        story.append(Spacer(1, 0.2*inch))
        return story

    def _create_trend_analysis(self, data: Dict[str, Any]) -> List:
        """트렌드 분석 섹션 생성 (Enhanced with markdown parsing)."""
        story = []
        sections = data['enhanced_sections']

        # Enhanced: Use markdown for section header
        header_markdown = "# 4. 트렌드 분석"
        header_elements = self.markdown_converter.convert_to_pdf_elements(header_markdown)
        story.extend(header_elements)

        # Enhanced: Parse trend analysis content as markdown
        trend_text = sections.get('trend_analysis', '트렌드 분석 정보가 없습니다.')
        trend_elements = self.markdown_converter.convert_to_pdf_elements(trend_text)
        story.extend(trend_elements)

        story.append(Spacer(1, 0.3*inch))
        return story

    def _create_risks_opportunities(self, data: Dict[str, Any]) -> List:
        """리스크 및 기회 섹션 생성 (Enhanced with markdown parsing)."""
        story = []
        sections = data['enhanced_sections']

        # Enhanced: Use markdown for section header
        header_markdown = "# 5. 리스크 및 기회"
        header_elements = self.markdown_converter.convert_to_pdf_elements(header_markdown)
        story.extend(header_elements)

        # Enhanced: Parse risks and opportunities content as markdown
        ro_text = sections.get('risks_opportunities', '리스크 및 기회 분석 정보가 없습니다.')
        ro_elements = self.markdown_converter.convert_to_pdf_elements(ro_text)
        story.extend(ro_elements)

        story.append(Spacer(1, 0.3*inch))
        return story

    def _create_dynamic_sections(self, data: Dict[str, Any]) -> List:
        """
        Dynamic: Generate report sections based on topic classification.
        Only creates relevant sections for the specific topic type.
        """
        story = []
        search_result = data['search_result']
        
        # Classify the topic
        topic = " ".join(search_result.query_keywords)
        keywords = search_result.query_keywords
        classification = self.topic_classifier.classify_topic(topic, keywords)
        
        logger.info(f"Topic classified as: {classification.primary_type.value} "
                   f"(confidence: {classification.confidence:.1%})")
        
        # Add classification info to the report
        classification_markdown = f"""
## 🎯 주제 분석 결과

**주제 유형**: {self._get_topic_type_korean(classification.primary_type)}  
**신뢰도**: {classification.confidence:.1%}  
**분석 근거**: {classification.reasoning}  
**매칭 키워드**: {', '.join(classification.keywords_matched)}

---
"""
        classification_elements = self.markdown_converter.convert_to_pdf_elements(classification_markdown)
        story.extend(classification_elements)
        
        # Generate sections based on classification
        section_number = 2  # Start after executive summary
        
        # Always include key findings
        story.extend(self._create_key_findings(data))
        section_number += 1
        
        # Always include detailed issues
        story.extend(self._create_detailed_issues(data))
        section_number += 1
        
        # Conditionally include other sections based on topic type
        if self._should_include_trend_analysis(classification):
            story.extend(self._create_trend_analysis(data))
            section_number += 1
        
        if self._should_include_risk_analysis(classification):
            story.extend(self._create_risks_opportunities(data))
            section_number += 1
        
        # Always include recommendations but adapt content
        story.extend(self._create_adaptive_recommendations(data, classification))
        
        return story

    def _get_topic_type_korean(self, topic_type) -> str:
        """Convert topic type to Korean description."""
        type_names = {
            'technical_announcement': '기술 발표/업데이트',
            'product_launch': '제품 출시',
            'business_strategic': '비즈니스/전략',
            'research_scientific': '연구/과학',
            'social_political': '사회/정치',
            'financial_market': '금융/시장',
            'general': '일반'
        }
        return type_names.get(topic_type.value, '일반')

    def _should_include_trend_analysis(self, classification) -> bool:
        """Determine if trend analysis should be included."""
        # Include for most types except pure technical announcements
        if classification.primary_type.value == 'technical_announcement' and classification.confidence > 0.8:
            return False
        return True

    def _should_include_risk_analysis(self, classification) -> bool:
        """Determine if risk analysis should be included."""
        return self.topic_classifier.should_include_risk_analysis(classification)

    def _create_adaptive_recommendations(self, data: Dict[str, Any], classification) -> List:
        """Create recommendations adapted to topic type."""
        story = []
        sections = data['enhanced_sections']

        # Adaptive header based on topic type
        if classification.primary_type.value == 'technical_announcement':
            header = "# 🔧 기술 구현 가이드라인"
        elif classification.primary_type.value == 'product_launch':
            header = "# 📱 제품 활용 방안"
        elif classification.primary_type.value == 'business_strategic':
            header = "# 📊 전략적 대응 방안"
        elif classification.primary_type.value == 'research_scientific':
            header = "# 🔬 연구 활용 및 후속 연구"
        else:
            header = "# 💡 권장 조치사항"

        header_elements = self.markdown_converter.convert_to_pdf_elements(header)
        story.extend(header_elements)

        # Adaptive content
        recommendations = sections.get('recommended_actions', [])
        if recommendations:
            rec_text = "\n".join([f"- {rec}" for rec in recommendations])
        else:
            # Generate default recommendations based on topic type
            rec_text = self._generate_default_recommendations(classification)

        rec_elements = self.markdown_converter.convert_to_pdf_elements(rec_text)
        story.extend(rec_elements)

        story.append(Spacer(1, 0.3*inch))
        return story

    def _generate_default_recommendations(self, classification) -> str:
        """Generate default recommendations based on topic type."""
        if classification.primary_type.value == 'technical_announcement':
            return """
- 새로운 기술 동향을 지속적으로 모니터링하세요
- 개발팀과 기술 변화 사항을 공유하세요
- 호환성 및 마이그레이션 계획을 수립하세요
- 관련 문서 및 가이드를 업데이트하세요
"""
        elif classification.primary_type.value == 'business_strategic':
            return """
- 시장 변화에 따른 전략 조정을 검토하세요
- 경쟁사 동향을 지속적으로 분석하세요
- 이해관계자들과 소통 계획을 수립하세요
- 리스크 관리 체계를 강화하세요
"""
        else:
            return """
- 관련 정보를 지속적으로 모니터링하세요
- 주요 이해관계자들과 정보를 공유하세요
- 필요시 전문가 의견을 구하세요
- 변화에 대한 대응 계획을 수립하세요
"""

    def _create_recommendations(self, data: Dict[str, Any]) -> List:
        """권장 조치사항 섹션 생성."""
        story = []
        sections = data['enhanced_sections']

        story.append(Paragraph("6. 권장 조치사항", self.styles['CustomHeading1']))
        story.append(Spacer(1, 0.2*inch))

        actions = sections.get('recommended_actions', [])
        if actions:
            for i, action in enumerate(actions, 1):
                bullet = Paragraph(f"{i}. {action}", self.styles['CustomBody'])
                story.append(bullet)
                story.append(Spacer(1, 0.1*inch))
        else:
            story.append(Paragraph("권장 조치사항이 없습니다.", self.styles['CustomBody']))

        story.append(PageBreak())
        return story

    def _create_appendix(self, data: Dict[str, Any]) -> List:
        """부록: 환각 탐지 상세 결과."""
        story = []
        search_result = data['search_result']

        story.append(Paragraph("부록. 환각 탐지 상세 결과", self.styles['CustomHeading1']))
        story.append(Spacer(1, 0.2*inch))

        # 환각 탐지 시스템 설명
        explanation = """
본 보고서의 모든 이슈는 3단계 AI 환각 탐지 시스템을 통해 검증되었습니다:
• RePPL (Relevant Paraphrased Prompt with Logit) 탐지기
• 자기 일관성 (Self-Consistency) 검사
• LLM-as-Judge 평가
        """
        story.append(Paragraph(explanation.strip(), self.styles['CustomBody']))

        story.append(Spacer(1, 0.2*inch))

        # 신뢰도 분포 차트 (간단한 테이블로 대체)
        if hasattr(search_result, 'confidence_distribution'):
            dist = search_result.confidence_distribution
            dist_data = [
                ['신뢰도 등급', '이슈 수', '비율'],
                ['높음 (70% 이상)', f"{dist.get('high', 0)}개",
                 f"{dist.get('high', 0)/max(1, search_result.total_found)*100:.1f}%"],
                ['중간 (40-70%)', f"{dist.get('moderate', 0)}개",
                 f"{dist.get('moderate', 0)/max(1, search_result.total_found)*100:.1f}%"],
                ['낮음 (40% 미만)', f"{dist.get('low', 0)}개",
                 f"{dist.get('low', 0)/max(1, search_result.total_found)*100:.1f}%"]
            ]

            dist_table = Table(dist_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            dist_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), self.default_font),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
            ]))

            story.append(dist_table)

        return story

    def _add_page_number(self, canvas_obj, doc):
        """페이지 번호 추가."""
        canvas_obj.saveState()
        canvas_obj.setFont(self.default_font, 9)
        canvas_obj.setFillColor(colors.HexColor('#7f8c8d'))

        page_num = canvas_obj.getPageNumber()
        if page_num > 1:  # 표지 제외
            text = f"페이지 {page_num - 1}"
            canvas_obj.drawRightString(
                doc.pagesize[0] - doc.rightMargin,
                doc.bottomMargin / 2,
                text
            )

        canvas_obj.restoreState()

    async def generate_report(self, search_result: SearchResult, topic: str) -> str:
        """
        전체 PDF 보고서 생성 프로세스.

        Args:
            search_result: 검색 결과 객체
            topic: 보고서 주제

        Returns:
            str: 생성된 PDF 파일 경로
        """
        try:
            # 1. LLM으로 내용 개선
            enhanced_data = await self.enhance_content_with_llm(search_result)

            # 2. 파일 경로 설정
            reports_dir = "reports"
            os.makedirs(reports_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"report_{safe_topic}_{timestamp}_enhanced.pdf"
            file_path = os.path.join(reports_dir, filename)

            # 3. PDF 생성
            self.generate_pdf(enhanced_data, file_path)

            return file_path

        except Exception as e:
            logger.error(f"PDF 보고서 생성 중 오류: {e}")
            raise


# 기존 시스템과의 통합을 위한 헬퍼 함수
async def create_pdf_report(search_result: SearchResult, topic: str) -> str:
    """
    검색 결과로부터 PDF 보고서를 생성하는 헬퍼 함수.

    Args:
        search_result: 검색 결과 객체
        topic: 보고서 주제

    Returns:
        str: 생성된 PDF 파일 경로
    """
    generator = PDFReportGenerator()
    return await generator.generate_report(search_result, topic)