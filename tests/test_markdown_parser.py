"""
Comprehensive tests for markdown parser module.

Tests the MarkdownToPDFConverter class which handles converting markdown content
to ReportLab PDF elements with proper formatting and styling.
"""

import pytest
from unittest.mock import patch, MagicMock
from reportlab.platypus import Paragraph, Spacer, HRFlowable
from reportlab.lib import colors

from src.reporting.markdown_parser import MarkdownToPDFConverter


class TestMarkdownToPDFConverter:
    """Test MarkdownToPDFConverter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = MarkdownToPDFConverter()
    
    def test_converter_initialization(self):
        """Test converter initializes with correct settings."""
        assert self.converter.default_font == 'NotoSansKR'
        assert 'base' in self.converter.styles
        assert 'h1' in self.converter.styles
        assert 'h2' in self.converter.styles
        assert 'h3' in self.converter.styles
        assert 'h4' in self.converter.styles
        assert 'code_block' in self.converter.styles
        assert 'inline_code' in self.converter.styles
        assert 'list_item' in self.converter.styles
        assert 'quote' in self.converter.styles
    
    def test_converter_initialization_custom_font(self):
        """Test converter initialization with custom font."""
        converter = MarkdownToPDFConverter(default_font='Helvetica')
        assert converter.default_font == 'Helvetica'
    
    def test_convert_empty_text(self):
        """Test converting empty or None text."""
        # Empty string
        result = self.converter.convert_to_pdf_elements("")
        assert len(result) == 1
        assert isinstance(result[0], Paragraph)
        
        # None
        result = self.converter.convert_to_pdf_elements(None)
        assert len(result) == 1
        assert isinstance(result[0], Paragraph)
        
        # Whitespace only
        result = self.converter.convert_to_pdf_elements("   \n\t  ")
        assert len(result) == 1
        assert isinstance(result[0], Paragraph)
    
    def test_convert_simple_paragraph(self):
        """Test converting simple paragraph text."""
        text = "This is a simple paragraph."
        result = self.converter.convert_to_pdf_elements(text)
        
        assert len(result) >= 1
        assert isinstance(result[0], Paragraph)
    
    def test_convert_headers(self):
        """Test converting different header levels."""
        # Test each header level
        test_cases = [
            ("# Header 1", 'h1'),
            ("## Header 2", 'h2'),
            ("### Header 3", 'h3'),
            ("#### Header 4", 'h4'),
        ]
        
        for markdown, expected_style in test_cases:
            result = self.converter.convert_to_pdf_elements(markdown)
            assert len(result) >= 1
            assert isinstance(result[0], Paragraph)
    
    def test_convert_horizontal_rule_fixed(self):
        """Test converting horizontal rules - this was the bug that was fixed."""
        text = "Before rule\n\n---\n\nAfter rule"
        result = self.converter.convert_to_pdf_elements(text)
        
        # Should not raise an error and should contain HRFlowable
        assert len(result) > 0
        # Find HRFlowable in results
        hr_found = any(isinstance(elem, HRFlowable) for elem in result)
        assert hr_found
    
    def test_convert_code_block(self):
        """Test converting code blocks."""
        text = """
Some text

```python
def hello():
    print("Hello, world!")
```

More text
"""
        result = self.converter.convert_to_pdf_elements(text)
        assert len(result) > 0
        # Should contain paragraphs for text and code
        paragraphs = [elem for elem in result if isinstance(elem, Paragraph)]
        assert len(paragraphs) >= 2
    
    def test_convert_lists(self):
        """Test converting both bullet and numbered lists."""
        text = """
# List Test

Bullet list:
- Item 1
- Item 2
- Item 3

Numbered list:
1. First item
2. Second item
3. Third item
"""
        result = self.converter.convert_to_pdf_elements(text)
        assert len(result) > 0
        
        # Should contain multiple paragraphs for list items
        paragraphs = [elem for elem in result if isinstance(elem, Paragraph)]
        assert len(paragraphs) >= 6  # Header + list items
    
    def test_split_into_blocks_headers(self):
        """Test _split_into_blocks method with headers."""
        text = "# H1\n## H2\n### H3\n#### H4"
        blocks = self.converter._split_into_blocks(text)
        
        assert len(blocks) == 4
        assert blocks[0]['type'] == 'h1'
        assert blocks[0]['content'] == 'H1'
        assert blocks[1]['type'] == 'h2'
        assert blocks[1]['content'] == 'H2'
        assert blocks[2]['type'] == 'h3'
        assert blocks[2]['content'] == 'H3'
        assert blocks[3]['type'] == 'h4'
        assert blocks[3]['content'] == 'H4'
    
    def test_split_into_blocks_code_block(self):
        """Test _split_into_blocks method with code blocks."""
        text = """
Text before

```python
def test():
    pass
```

Text after
"""
        blocks = self.converter._split_into_blocks(text)
        
        # Should have 3 blocks: paragraph, code_block, paragraph
        assert len(blocks) >= 2
        code_blocks = [b for b in blocks if b['type'] == 'code_block']
        assert len(code_blocks) == 1
        assert 'def test():' in code_blocks[0]['content']
    
    def test_split_into_blocks_lists(self):
        """Test _split_into_blocks method with lists."""
        text = """
- Bullet 1
- Bullet 2

1. Number 1
2. Number 2
"""
        blocks = self.converter._split_into_blocks(text)
        
        list_blocks = [b for b in blocks if b['type'] == 'list']
        assert len(list_blocks) >= 2
        
        # Check bullet list
        bullet_block = None
        numbered_block = None
        for block in list_blocks:
            if any(item[0] == 'bullet' for item in block['content']):
                bullet_block = block
            elif any(item[0] == 'numbered' for item in block['content']):
                numbered_block = block
        
        assert bullet_block is not None
        assert numbered_block is not None
    
    def test_split_into_blocks_horizontal_rule(self):
        """Test _split_into_blocks method with horizontal rules."""
        text = "Before\n\n---\n\nAfter"
        blocks = self.converter._split_into_blocks(text)
        
        hr_blocks = [b for b in blocks if b['type'] == 'hr']
        assert len(hr_blocks) == 1
    
    def test_process_block_header_validation(self):
        """Test _process_block method properly validates headers."""
        # Valid header block
        valid_block = {'type': 'h1', 'content': 'Valid Header'}
        result = self.converter._process_block(valid_block)
        assert len(result) >= 1
        assert isinstance(result[0], Paragraph)
        
        # Invalid header-like block (this was the bug)
        invalid_block = {'type': 'hr', 'content': ''}
        result = self.converter._process_block(invalid_block)
        # Should not try to parse 'hr'[1] as integer
        assert len(result) >= 1
    
    def test_process_block_code_block(self):
        """Test _process_block method with code blocks."""
        block = {
            'type': 'code_block',
            'content': 'def hello():\n    print("Hello")'
        }
        result = self.converter._process_block(block)
        
        assert len(result) >= 1
        assert isinstance(result[0], Paragraph)
    
    def test_process_block_list(self):
        """Test _process_block method with lists."""
        block = {
            'type': 'list',
            'content': [
                ('bullet', 'First item'),
                ('bullet', 'Second item'),
                ('numbered', 'Third item')
            ]
        }
        result = self.converter._process_block(block)
        
        # Should create paragraph for each list item
        paragraphs = [elem for elem in result if isinstance(elem, Paragraph)]
        assert len(paragraphs) == 3
    
    def test_process_block_paragraph(self):
        """Test _process_block method with regular paragraphs."""
        block = {
            'type': 'paragraph',
            'content': 'This is a regular paragraph with **bold** and *italic* text.'
        }
        result = self.converter._process_block(block)
        
        assert len(result) >= 1
        assert isinstance(result[0], Paragraph)
    
    def test_format_inline_markdown_bold_italic(self):
        """Test _format_inline_markdown method with bold and italic."""
        text = "This has **bold** and *italic* text."
        result = self.converter._format_inline_markdown(text)
        
        assert '<b>bold</b>' in result
        assert '<i>italic</i>' in result
    
    def test_format_inline_markdown_code(self):
        """Test _format_inline_markdown method with inline code."""
        text = "This has `inline code` in it."
        result = self.converter._format_inline_markdown(text)
        
        assert '<font name="Courier"' in result
        assert 'inline code' in result
    
    def test_format_inline_markdown_links(self):
        """Test _format_inline_markdown method with links."""
        text = "Check out [this link](https://example.com) for more info."
        result = self.converter._format_inline_markdown(text)
        
        assert '<link href="https://example.com">' in result
        assert 'this link' in result
    
    def test_format_inline_markdown_escaping(self):
        """Test _format_inline_markdown method escapes HTML entities."""
        text = "This has < and > and & characters."
        result = self.converter._format_inline_markdown(text)
        
        assert '&lt;' in result
        assert '&gt;' in result
        assert '&amp;' in result
    
    def test_format_inline_markdown_emojis(self):
        """Test _format_inline_markdown method preserves emojis."""
        text = "Status: 🟢 Good 🔴 Bad 🟡 Warning"
        result = self.converter._format_inline_markdown(text)
        
        assert '🟢' in result
        assert '🔴' in result
        assert '🟡' in result
    
    def test_create_confidence_indicator(self):
        """Test create_confidence_indicator method."""
        # High confidence (>= 0.8)
        assert self.converter.create_confidence_indicator(0.9) == "🟢"
        assert self.converter.create_confidence_indicator(0.8) == "🟢"
        
        # Medium-high confidence (>= 0.6)
        assert self.converter.create_confidence_indicator(0.7) == "🟡"
        assert self.converter.create_confidence_indicator(0.6) == "🟡"
        
        # Medium-low confidence (>= 0.4)
        assert self.converter.create_confidence_indicator(0.5) == "🟠"
        assert self.converter.create_confidence_indicator(0.4) == "🟠"
        
        # Low confidence (< 0.4)
        assert self.converter.create_confidence_indicator(0.3) == "🔴"
        assert self.converter.create_confidence_indicator(0.0) == "🔴"
    
    def test_format_confidence_text(self):
        """Test format_confidence_text method."""
        result = self.converter.format_confidence_text(0.85)
        assert "🟢" in result
        assert "85.0%" in result
        
        result = self.converter.format_confidence_text(0.65)
        assert "🟡" in result
        assert "65.0%" in result
    
    def test_long_content_handling(self):
        """Test handling of very long content."""
        # Create long paragraph
        long_text = "This is a very long sentence. " * 100
        result = self.converter.convert_to_pdf_elements(long_text)
        
        assert len(result) >= 1
        assert isinstance(result[0], Paragraph)
        
        # Create long code block
        long_code = """
```python
# This is a very long code block
""" + "# Long comment line\n" * 50 + """
def long_function():
    pass
```
"""
        result = self.converter.convert_to_pdf_elements(long_code)
        assert len(result) >= 1
    
    def test_mixed_content_complex(self):
        """Test complex mixed content with all elements."""
        complex_text = """
# Main Title

This is an introduction paragraph with **bold** and *italic* text.

## Section 1

Here's a list:
- First item with `inline code`
- Second item with [a link](https://example.com)
- Third item

---

## Section 2

Here's a code block:

```python
def example_function():
    print("Hello, world!")
    return True
```

### Subsection

1. Numbered item one
2. Numbered item two
3. Numbered item three

Final paragraph with 🎯 emojis and **formatting**.
"""
        result = self.converter.convert_to_pdf_elements(complex_text)
        
        # Should contain various element types
        assert len(result) > 10  # Many elements
        
        # Check for different element types
        paragraphs = [elem for elem in result if isinstance(elem, Paragraph)]
        spacers = [elem for elem in result if isinstance(elem, Spacer)]
        hrs = [elem for elem in result if isinstance(elem, HRFlowable)]
        
        assert len(paragraphs) > 5
        assert len(spacers) > 0
        assert len(hrs) == 1


class TestMarkdownParserEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = MarkdownToPDFConverter()
    
    def test_malformed_markdown(self):
        """Test handling of malformed markdown."""
        malformed_cases = [
            "# Header without content\n",
            "```\nUnclosed code block",
            "- List item\nNot a list item\n- Another list item",
            "**Bold without closing",
            "*Italic without closing",
            "[Link without closing",
            "`Code without closing",
        ]
        
        for case in malformed_cases:
            # Should not raise exceptions
            result = self.converter.convert_to_pdf_elements(case)
            assert len(result) >= 1
    
    def test_nested_formatting(self):
        """Test nested formatting that might cause issues."""
        text = "This has **bold with *italic inside* it** and more."
        result = self.converter._format_inline_markdown(text)
        
        # Should handle nested formatting gracefully
        assert '<b>' in result
        assert '<i>' in result
    
    def test_special_characters(self):
        """Test handling of special characters."""
        text = "Special chars: < > & \" ' \n\t \r Unicode: café naïve résumé"
        result = self.converter.convert_to_pdf_elements(text)
        
        assert len(result) >= 1
        assert isinstance(result[0], Paragraph)
    
    def test_empty_code_block(self):
        """Test empty code block handling."""
        text = "```\n\n```"
        result = self.converter.convert_to_pdf_elements(text)
        
        # Should handle gracefully without errors
        assert len(result) >= 0
    
    def test_multiple_consecutive_headers(self):
        """Test multiple consecutive headers."""
        text = "# Header 1\n## Header 2\n### Header 3\n#### Header 4"
        result = self.converter.convert_to_pdf_elements(text)
        
        paragraphs = [elem for elem in result if isinstance(elem, Paragraph)]
        assert len(paragraphs) == 4
    
    def test_mixed_list_types(self):
        """Test mixed bullet and numbered lists."""
        text = """
- Bullet 1
- Bullet 2
1. Number 1
2. Number 2
- Bullet 3
"""
        result = self.converter.convert_to_pdf_elements(text)
        
        # Should create separate list blocks
        paragraphs = [elem for elem in result if isinstance(elem, Paragraph)]
        assert len(paragraphs) >= 5
    
    def test_performance_large_document(self):
        """Test performance with large document."""
        # Create a large document
        sections = []
        for i in range(50):
            sections.append(f"## Section {i}")
            sections.append(f"This is content for section {i}.")
            sections.append(f"- Item 1 for section {i}")
            sections.append(f"- Item 2 for section {i}")
            sections.append("---")
        
        large_text = "\n\n".join(sections)
        
        # Should complete without timeout or memory issues
        result = self.converter.convert_to_pdf_elements(large_text)
        assert len(result) > 100  # Many elements


class TestMarkdownParserStyles:
    """Test the style creation and application."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = MarkdownToPDFConverter()
    
    def test_style_creation(self):
        """Test that all required styles are created."""
        styles = self.converter.styles
        
        required_styles = [
            'base', 'h1', 'h2', 'h3', 'h4',
            'code_block', 'inline_code', 'list_item', 'quote'
        ]
        
        for style_name in required_styles:
            assert style_name in styles
            style = styles[style_name]
            assert hasattr(style, 'fontName')
            assert hasattr(style, 'fontSize')
    
    def test_header_style_hierarchy(self):
        """Test that header styles have correct size hierarchy."""
        h1_size = self.converter.styles['h1'].fontSize
        h2_size = self.converter.styles['h2'].fontSize
        h3_size = self.converter.styles['h3'].fontSize
        h4_size = self.converter.styles['h4'].fontSize
        
        assert h1_size > h2_size > h3_size > h4_size
    
    def test_code_style_font(self):
        """Test that code styles use monospace font."""
        code_block_style = self.converter.styles['code_block']
        assert 'Courier' in code_block_style.fontName
    
    def test_custom_font_propagation(self):
        """Test that custom font is used in styles when it's a known safe font."""
        # Test with a known safe font
        converter = MarkdownToPDFConverter(default_font='Times')
        
        # Check that custom font is used in styles
        assert 'Times' in converter.styles['base'].fontName
        assert 'Times' in converter.styles['h1'].fontName
        assert 'Times' in converter.styles['h2'].fontName
        
        # Test with an unknown font - should fallback to Helvetica
        converter_unknown = MarkdownToPDFConverter(default_font='UnknownFont')
        assert 'Helvetica' in converter_unknown.styles['base'].fontName