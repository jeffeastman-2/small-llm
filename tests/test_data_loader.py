"""
Unit tests for data_loader module.
"""

import unittest
import tempfile
import os
import fitz  # PyMuPDF

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data_loader import extract_text_from_pdfs, clean_text


class TestDataLoader(unittest.TestCase):
    """Test cases for data loader functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test data."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    def create_test_pdf(self, filename, content):
        """Create a test PDF with given content."""
        pdf_path = os.path.join(self.test_dir, filename)
        
        # Create a simple PDF with text
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), content)
        doc.save(pdf_path)
        doc.close()
        
        return pdf_path
    
    def test_extract_single_pdf(self):
        """Test extracting text from a single PDF."""
        content = "This is test content for the PDF."
        self.create_test_pdf("test1.pdf", content)
        
        extracted = extract_text_from_pdfs(self.test_dir)
        
        # Should contain the test content
        self.assertIn("test content", extracted.lower())
        self.assertGreater(len(extracted), 0)
    
    def test_extract_multiple_pdfs(self):
        """Test extracting text from multiple PDFs."""
        content1 = "Content from first PDF."
        content2 = "Content from second PDF."
        
        self.create_test_pdf("test1.pdf", content1)
        self.create_test_pdf("test2.pdf", content2)
        
        extracted = extract_text_from_pdfs(self.test_dir)
        
        # Should contain content from both PDFs
        self.assertIn("first pdf", extracted.lower())
        self.assertIn("second pdf", extracted.lower())
    
    def test_empty_directory(self):
        """Test extraction from directory with no PDFs."""
        with self.assertRaises(ValueError):
            extract_text_from_pdfs(self.test_dir)
    
    def test_directory_with_non_pdf_files(self):
        """Test extraction from directory with non-PDF files."""
        # Create a non-PDF file
        txt_path = os.path.join(self.test_dir, "test.txt")
        with open(txt_path, 'w') as f:
            f.write("This is a text file")
        
        # Should raise error since no PDFs found
        with self.assertRaises(ValueError):
            extract_text_from_pdfs(self.test_dir)
    
    def test_mixed_files(self):
        """Test extraction from directory with both PDF and non-PDF files."""
        # Create PDF and non-PDF files
        self.create_test_pdf("document.pdf", "PDF content here")
        
        txt_path = os.path.join(self.test_dir, "readme.txt")
        with open(txt_path, 'w') as f:
            f.write("Text file content")
        
        extracted = extract_text_from_pdfs(self.test_dir)
        
        # Should only contain PDF content
        self.assertIn("pdf content", extracted.lower())
        self.assertNotIn("text file content", extracted.lower())
    
    def test_clean_text(self):
        """Test text cleaning function."""
        # Test with messy text
        messy_text = "  This   is\n\na\t\ttest   with   \n  whitespace  "
        cleaned = clean_text(messy_text)
        
        expected = "This is a test with whitespace"
        self.assertEqual(cleaned, expected)
    
    def test_clean_text_empty(self):
        """Test cleaning empty text."""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("   \n\t  "), "")
    
    def test_clean_text_newlines_tabs(self):
        """Test cleaning text with newlines and tabs."""
        text_with_formatting = "Line 1\nLine 2\tTabbed\nLine 3"
        cleaned = clean_text(text_with_formatting)
        
        # Should replace newlines and tabs with spaces
        self.assertNotIn('\n', cleaned)
        self.assertNotIn('\t', cleaned)
        self.assertEqual(cleaned, "Line 1 Line 2 Tabbed Line 3")
    
    def test_nonexistent_directory(self):
        """Test extraction from non-existent directory."""
        fake_dir = "/path/that/does/not/exist"
        
        with self.assertRaises(FileNotFoundError):
            extract_text_from_pdfs(fake_dir)
    
    def test_large_pdf_content(self):
        """Test extraction from PDF with larger content."""
        # Create longer content (but more realistic for PDF rendering)
        long_content = " ".join([f"Line {i} content." for i in range(20)])
        self.create_test_pdf("large.pdf", long_content)
        
        extracted = extract_text_from_pdfs(self.test_dir)
        
        # Should contain substantial content
        self.assertGreater(len(extracted), 50)
        self.assertIn("Line 0", extracted)
        self.assertIn("Line 1", extracted)
        # Check that multiple lines are present
        self.assertTrue(any(f"Line {i}" in extracted for i in range(5, 15)))


class MockPDFTests(unittest.TestCase):
    """Tests that don't require actual PDF creation."""
    
    def test_clean_text_edge_cases(self):
        """Test edge cases for text cleaning."""
        test_cases = [
            ("", ""),
            ("   ", ""),
            ("a", "a"),
            ("a b", "a b"),
            ("a\nb", "a b"),
            ("a\tb", "a b"),
            ("a\n\nb", "a b"),
            ("  a  \n  b  ", "a b"),
            ("word1\n\n\nword2\t\t\tword3", "word1 word2 word3"),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input_text=repr(input_text)):
                result = clean_text(input_text)
                self.assertEqual(result, expected)


if __name__ == '__main__':
    # Check if PyMuPDF is available
    try:
        import fitz
        unittest.main()
    except ImportError:
        print("Skipping data_loader tests: PyMuPDF not available")
        print("Install with: pip install PyMuPDF")
