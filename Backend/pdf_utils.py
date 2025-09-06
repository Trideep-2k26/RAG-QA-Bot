import pdfplumber
from typing import List, Dict, Any, Tuple
import os
import re
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import io
try:
    import fitz
except Exception:
    fitz = None

_TESS_CMD = os.getenv("TESSERACT_CMD")
if _TESS_CMD:
    try:
        pytesseract.pytesseract.tesseract_cmd = _TESS_CMD
    except Exception:
        pass

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        pdf_io = BytesIO(pdf_bytes)
        with pdfplumber.open(pdf_io) as pdf:
            text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        return text
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def extract_pages_from_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    try:
        pdf_io = BytesIO(pdf_bytes)
        pages: List[Dict[str, Any]] = []
        with pdfplumber.open(pdf_io) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ''
                pages.append({'page': idx, 'text': page_text})
        return pages
    except Exception as e:
        raise Exception(f"Failed to extract pages from PDF: {str(e)}")


def _ocr_images(images: List[Image.Image], lang: str) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    for idx, img in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(img, lang=lang) or ""
            text = text.strip()
            if text:
                pages.append({'page': idx, 'text': text})
        except Exception:
            continue
    return pages


def _render_with_pymupdf(pdf_bytes: bytes, dpi: int = 150) -> List[Image.Image]:
    if fitz is None:
        raise Exception("PyMuPDF not installed")
    images: List[Image.Image] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            png_bytes = pix.tobytes("png")
            img = Image.open(io.BytesIO(png_bytes))
            images.append(img)
    finally:
        doc.close()
    return images


def _extract_pages_via_ocr(pdf_bytes: bytes, lang: str = "eng") -> List[Dict[str, Any]]:
    try:
        poppler_path = os.getenv("POPPLER_PATH")
        if poppler_path:
            images = convert_from_bytes(pdf_bytes, poppler_path=poppler_path)
        else:
            images = convert_from_bytes(pdf_bytes)
        return _ocr_images(images, lang)
    except Exception:
        pass

    try:
        images = _render_with_pymupdf(pdf_bytes, dpi=150)
        return _ocr_images(images, lang)
    except Exception as e:
        raise Exception(f"OCR extraction failed after fallbacks: {e}")


def extract_pages_with_ocr_fallback(pdf_bytes: bytes, lang: str = "eng") -> Tuple[List[Dict[str, Any]], str]:
    try:
        pages = extract_pages_from_pdf(pdf_bytes)
    except Exception:
        pages = []
    has_text = any((p.get('text') or '').strip() for p in pages)
    if has_text:
        return pages, 'pdfplumber'

    ocr_pages = _extract_pages_via_ocr(pdf_bytes, lang=lang)
    return ocr_pages, 'ocr'

def split_into_chunks(text: str, max_tokens: int = 500) -> List[str]:
    max_chars = max_tokens * 4
    sentences = re.split('([.!?]+)', text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0
    for i in range(0, len(sentences), 2):
        sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
        sentence = sentence.strip()
        if not sentence:
            continue
        if current_length + len(sentence) > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def split_into_chunks_by_page(pages: List[Dict[str, Any]], max_tokens: int = 500) -> List[Dict[str, Any]]:
    max_chars = max_tokens * 4
    results: List[Dict[str, Any]] = []
    for entry in pages:
        page_no = entry.get('page')
        text = entry.get('text', '')
        sentences = re.split('([.!?]+)', text)
        current_chunk: List[str] = []
        current_length = 0
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_length + len(sentence) > max_chars and current_chunk:
                results.append({'page': page_no, 'text': ' '.join(current_chunk)})
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += len(sentence)
        if current_chunk:
            results.append({'page': page_no, 'text': ' '.join(current_chunk)})
    return results


def extract_page_images(pdf_bytes: bytes, media_root: str, base_name: str, dpi: int = 150) -> Dict[int, str]:
    try:
        os.makedirs(media_root, exist_ok=True)
        pdf_io = BytesIO(pdf_bytes)
        page_to_image: Dict[int, str] = {}
        safe_base = os.path.splitext(os.path.basename(base_name))[0]
        with pdfplumber.open(pdf_io) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                try:
                    im = page.to_image(resolution=dpi)
                    filename = f"{safe_base}-p{idx}.png"
                    out_path = os.path.join(media_root, filename)
                    im.save(out_path, format="PNG")
                    page_to_image[idx] = filename
                except Exception:
                    continue
        return page_to_image
    except Exception as e:
        return {}
