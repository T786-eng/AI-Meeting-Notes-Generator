import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import PyPDF2
from docx import Document

# If you are on Windows, uncomment the line below and point to your tesseract.exe
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    # 1. Handle standard Text
    if file_type == 'txt':
        return uploaded_file.getvalue().decode("utf-8")
    
    # 2. Handle Images (New!)
    elif file_type in ['jpg', 'jpeg', 'png']:
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    
    # 3. Handle PDF (with OCR fallback)
    elif file_type == 'pdf':
        # First try standard extraction
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
            
        # If extraction failed (likely a scanned PDF), use OCR
        if len(text.strip()) < 10:
            # Note: This requires 'poppler' installed on your system
            images = convert_from_path(uploaded_file)
            for img in images:
                text += pytesseract.image_to_string(img)
        return text

    # 4. Handle Word Docs
    elif file_type == 'docx':
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    
    return None