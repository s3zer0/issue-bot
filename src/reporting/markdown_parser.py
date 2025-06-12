"""
Enhanced Markdown to PDF converter with proper formatting.

This module converts markdown content to ReportLab elements with proper styling,
including headers, bold/italic text, lists, code blocks, and links.
"""

import re
from typing import List, Any, Dict, Union
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from loguru import logger


class MarkdownToPDFConverter:
    """
    Converts markdown text to ReportLab PDF elements with proper formatting.
    
    Supports:
    - Headers (## ### ####)
    - Bold (**text**) and italic (*text*)
    - Lists (- and 1.)
    - Code blocks and inline code
    - Links [text](url)
    - Horizontal rules
    """
    
    def __init__(self, default_font: str = 'NotoSansKR'):
        """Initialize converter with font settings."""
        self.default_font = default_font
        self.styles = self._create_markdown_styles()
        
        # Regex patterns for markdown elements
        self.patterns = {
            'header1': re.compile(r'^# (.+)$', re.MULTILINE),
            'header2': re.compile(r'^## (.+)$', re.MULTILINE),
            'header3': re.compile(r'^### (.+)$', re.MULTILINE),
            'header4': re.compile(r'^#### (.+)$', re.MULTILINE),
            'bold': re.compile(r'\*\*(.+?)\*\*'),
            'italic': re.compile(r'\*(.+?)\*'),
            'code_block': re.compile(r'```[\w]*\n(.*?)\n```', re.DOTALL),
            'inline_code': re.compile(r'`([^`]+)`'),
            'link': re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),
            'list_item': re.compile(r'^[-*+] (.+)$', re.MULTILINE),
            'numbered_list': re.compile(r'^(\d+)\. (.+)$', re.MULTILINE),
            'horizontal_rule': re.compile(r'^---+$', re.MULTILINE),
            'emoji': re.compile(r'(🔍|📊|📋|🟢|🔴|🟡|⚪|✅|❌|⚠️|📈|📉|🎯|💡|🔧|📝|🔗|📄|📊)')
        }
        
        logger.info("Markdown to PDF converter initialized")
    
    def _create_markdown_styles(self) -> Dict[str, ParagraphStyle]:
        """Create paragraph styles for different markdown elements."""
        styles = {}
        
        # Base style
        styles['base'] = ParagraphStyle(
            'MarkdownBase',
            fontName=self.default_font,
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#333333'),
            alignment=TA_LEFT,
            spaceBefore=6,
            spaceAfter=6
        )
        
        # Headers
        styles['h1'] = ParagraphStyle(
            'MarkdownH1',
            parent=styles['base'],
            fontSize=20,
            fontName=f'{self.default_font}',
            textColor=colors.HexColor('#1a1a1a'),
            spaceBefore=20,
            spaceAfter=15,
            leftIndent=0,
            borderWidth=2,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=8
        )
        
        styles['h2'] = ParagraphStyle(
            'MarkdownH2',
            parent=styles['base'],
            fontSize=16,
            fontName=f'{self.default_font}',
            textColor=colors.HexColor('#2c3e50'),
            spaceBefore=16,
            spaceAfter=12,
            leftIndent=0,
            borderWidth=1,
            borderColor=colors.HexColor('#95a5a6'),
            borderPadding=6
        )
        
        styles['h3'] = ParagraphStyle(
            'MarkdownH3',
            parent=styles['base'],
            fontSize=14,
            fontName=f'{self.default_font}',
            textColor=colors.HexColor('#34495e'),
            spaceBefore=12,
            spaceAfter=8,
            leftIndent=0
        )
        
        styles['h4'] = ParagraphStyle(
            'MarkdownH4',
            parent=styles['base'],
            fontSize=12,
            fontName=f'{self.default_font}',
            textColor=colors.HexColor('#7f8c8d'),
            spaceBefore=10,
            spaceAfter=6,
            leftIndent=0
        )
        
        # Code styles
        styles['code_block'] = ParagraphStyle(
            'MarkdownCodeBlock',
            parent=styles['base'],
            fontName='Courier',
            fontSize=9,
            leading=11,
            backgroundColor=colors.HexColor('#f8f9fa'),
            borderWidth=1,
            borderColor=colors.HexColor('#e9ecef'),
            borderPadding=8,
            leftIndent=10,
            rightIndent=10,
            spaceBefore=10,
            spaceAfter=10
        )
        
        styles['inline_code'] = ParagraphStyle(
            'MarkdownInlineCode',
            parent=styles['base'],
            fontName='Courier',
            fontSize=10,
            backgroundColor=colors.HexColor('#f1f3f4'),
            borderWidth=1,
            borderColor=colors.HexColor('#dadce0'),
            borderPadding=2
        )
        
        # List styles
        styles['list_item'] = ParagraphStyle(
            'MarkdownListItem',
            parent=styles['base'],
            leftIndent=20,
            bulletIndent=10,
            spaceBefore=2,
            spaceAfter=2
        )
        
        # Quote style
        styles['quote'] = ParagraphStyle(
            'MarkdownQuote',
            parent=styles['base'],
            leftIndent=20,
            rightIndent=20,
            borderWidth=2,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=10,
            backgroundColor=colors.HexColor('#f8f9fa'),
            fontStyle='italic'
        )
        
        return styles
    
    def convert_to_pdf_elements(self, markdown_text: str) -> List[Any]:
        """
        Convert markdown text to ReportLab PDF elements.
        
        Args:
            markdown_text: Raw markdown content
            
        Returns:
            List of ReportLab flowable elements
        """
        if not markdown_text or not markdown_text.strip():
            return [Paragraph("내용이 없습니다.", self.styles['base'])]
        
        elements = []
        
        # Split text into blocks (paragraphs, code blocks, etc.)
        blocks = self._split_into_blocks(markdown_text)
        
        for block in blocks:
            block_elements = self._process_block(block)
            elements.extend(block_elements)
        
        return elements
    
    def _split_into_blocks(self, text: str) -> List[Dict[str, str]]:
        """Split markdown text into logical blocks."""
        blocks = []
        lines = text.split('\n')
        current_block = {'type': 'paragraph', 'content': ''}
        in_code_block = False
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Code block detection
            if line.startswith('```'):
                if in_code_block:
                    # End of code block
                    blocks.append({'type': 'code_block', 'content': current_block['content']})
                    current_block = {'type': 'paragraph', 'content': ''}
                    in_code_block = False
                else:
                    # Start of code block
                    if current_block['content'].strip():
                        blocks.append(current_block)
                    current_block = {'type': 'code_block', 'content': ''}
                    in_code_block = True
                i += 1
                continue
            
            if in_code_block:
                current_block['content'] += line + '\n'
                i += 1
                continue
            
            # Headers
            if line.startswith('#'):
                if current_block['content'].strip():
                    blocks.append(current_block)
                
                level = len(re.match(r'^#+', line).group())
                content = line[level:].strip()
                blocks.append({'type': f'h{min(level, 4)}', 'content': content})
                current_block = {'type': 'paragraph', 'content': ''}
                i += 1
                continue
            
            # List items
            if re.match(r'^[-*+] ', line) or re.match(r'^\d+\. ', line):
                if current_block['content'].strip():
                    blocks.append(current_block)
                
                # Collect consecutive list items
                list_items = []
                while i < len(lines) and (re.match(r'^[-*+] ', lines[i]) or re.match(r'^\d+\. ', lines[i])):
                    if re.match(r'^[-*+] ', lines[i]):
                        list_items.append(('bullet', lines[i][2:].strip()))
                    else:
                        list_items.append(('numbered', re.match(r'^\d+\. (.+)', lines[i]).group(1)))
                    i += 1
                
                blocks.append({'type': 'list', 'content': list_items})
                current_block = {'type': 'paragraph', 'content': ''}
                continue
            
            # Horizontal rule
            if re.match(r'^---+$', line):
                if current_block['content'].strip():
                    blocks.append(current_block)
                blocks.append({'type': 'hr', 'content': ''})
                current_block = {'type': 'paragraph', 'content': ''}
                i += 1
                continue
            
            # Empty line - paragraph break
            if not line.strip():
                if current_block['content'].strip():
                    blocks.append(current_block)
                    current_block = {'type': 'paragraph', 'content': ''}
                i += 1
                continue
            
            # Regular paragraph content
            if current_block['content']:
                current_block['content'] += ' '
            current_block['content'] += line
            i += 1
        
        # Add final block
        if current_block['content'].strip():
            blocks.append(current_block)
        
        return blocks
    
    def _process_block(self, block: Dict[str, str]) -> List[Any]:
        """Process a single markdown block into PDF elements."""
        elements = []
        
        if block['type'].startswith('h') and len(block['type']) >= 2 and block['type'][1].isdigit():
            # Header - ensure we have a valid header format like 'h1', 'h2', etc.
            level = int(block['type'][1])
            formatted_text = self._format_inline_markdown(block['content'])
            style_name = f'h{level}'
            elements.append(Paragraph(formatted_text, self.styles[style_name]))
            elements.append(Spacer(1, 0.1*inch))
        
        elif block['type'] == 'code_block':
            # Code block
            code_content = block['content'].strip()
            if code_content:
                # Split long lines to prevent overflow
                lines = code_content.split('\n')
                formatted_lines = []
                for line in lines:
                    if len(line) > 80:  # Wrap long lines
                        while line:
                            formatted_lines.append(line[:80])
                            line = line[80:]
                    else:
                        formatted_lines.append(line)
                
                formatted_code = '\n'.join(formatted_lines)
                elements.append(Paragraph(f'<font name=\"Courier\">{formatted_code}</font>', 
                                        self.styles['code_block']))
                elements.append(Spacer(1, 0.1*inch))
        
        elif block['type'] == 'list':
            # List
            list_items = block['content']
            formatted_items = []
            
            for item_type, item_content in list_items:
                formatted_content = self._format_inline_markdown(item_content)
                if item_type == 'bullet':
                    bullet_text = f"• {formatted_content}"
                else:
                    bullet_text = f"{len(formatted_items)+1}. {formatted_content}"
                
                formatted_items.append(Paragraph(bullet_text, self.styles['list_item']))
            
            elements.extend(formatted_items)
            elements.append(Spacer(1, 0.1*inch))
        
        elif block['type'] == 'hr':
            # Horizontal rule
            from reportlab.platypus import HRFlowable
            elements.append(Spacer(1, 0.1*inch))
            elements.append(HRFlowable(width="100%", thickness=1, 
                                     color=colors.HexColor('#bdc3c7')))
            elements.append(Spacer(1, 0.1*inch))
        
        elif block['type'] == 'paragraph':
            # Regular paragraph
            formatted_text = self._format_inline_markdown(block['content'])
            if formatted_text.strip():
                elements.append(Paragraph(formatted_text, self.styles['base']))
                elements.append(Spacer(1, 0.05*inch))
        
        return elements
    
    def _format_inline_markdown(self, text: str) -> str:
        """Format inline markdown elements (bold, italic, code, links)."""
        if not text:
            return ""
        
        # Escape XML/HTML entities first
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        # Bold formatting
        text = self.patterns['bold'].sub(r'<b>\1</b>', text)
        
        # Italic formatting  
        text = self.patterns['italic'].sub(r'<i>\1</i>', text)
        
        # Inline code
        text = self.patterns['inline_code'].sub(
            r'<font name="Courier" color="#e74c3c" backgroundColor="#f8f9fa">\1</font>', text)
        
        # Links - make them blue and underlined
        text = self.patterns['link'].sub(
            r'<link href="\2"><font color="#3498db"><u>\1</u></font></link>', text)
        
        # Preserve emojis by detecting and keeping them
        text = self.patterns['emoji'].sub(r'\1', text)
        
        return text
    
    def create_confidence_indicator(self, confidence: float) -> str:
        """Create a visual confidence indicator."""
        if confidence >= 0.8:
            return "🟢"  # Green
        elif confidence >= 0.6:
            return "🟡"  # Yellow
        elif confidence >= 0.4:
            return "🟠"  # Orange
        else:
            return "🔴"  # Red
    
    def format_confidence_text(self, confidence: float) -> str:
        """Format confidence score with visual indicator."""
        indicator = self.create_confidence_indicator(confidence)
        percentage = f"{confidence:.1%}"
        return f"{indicator} {percentage}"


# Global converter instance
markdown_converter = MarkdownToPDFConverter()