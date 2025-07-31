"""
PDF text extraction utilities for training data preparation.
"""

import fitz  # PyMuPDF
import os


def extract_text_from_pdfs(directory):
    """
    Extract text from all PDF files in a directory.
    
    Args:
        directory (str): Path to directory containing PDF files
        
    Returns:
        str: Concatenated text from all PDF files
    """
    all_text = ""
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in directory: {directory}")
    
    print(f"Found {len(pdf_files)} PDF files to process:")
    for filename in pdf_files:
        print(f"  - {filename}")
    
    for filename in pdf_files:
        print(f"\nReading from {filename}")
        path = os.path.join(directory, filename)
        
        try:
            doc = fitz.open(path)
            file_text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                file_text += page_text
            
            all_text += file_text
            print(f"  Extracted {len(file_text)} characters from {len(doc)} pages")
            doc.close()
            
        except Exception as e:
            print(f"  Error reading {filename}: {e}")
            continue
    
    print(f"\nTotal text extracted: {len(all_text)} characters")
    return all_text


def clean_text(text):
    """
    Basic text cleaning for better tokenization.
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common PDF artifacts
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    
    return text
