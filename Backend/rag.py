import os
from typing import List, Dict, Any, Optional
import logging
import re
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    genai = None
    logging.warning(f"google.generativeai not available: {e}")

_CHROMA_DIR = os.getenv('CHROMA_DB_DIR', os.path.join(os.path.dirname(__file__), '.chroma'))
_VECTOR_INDEX_NAME = os.getenv('CHROMA_DATABASE') or os.getenv('PINECONE_INDEX') or os.getenv('VECTOR_INDEX') or 'ragv1'
_MEDIA_ROOT = os.getenv('MEDIA_ROOT', os.path.join(os.path.dirname(__file__), 'media'))

_chroma_client = None
_chroma_collection = None


def _init_chroma():
    global _chroma_client, _chroma_collection
    if _chroma_client is not None and _chroma_collection is not None:
        return
    try:
        import chromadb
    except Exception as e:
        raise Exception(f"chromadb not installed: {e}")

    try:
        _chroma_client = chromadb.PersistentClient(path=_CHROMA_DIR)
        _chroma_collection = _chroma_client.get_or_create_collection(name=_VECTOR_INDEX_NAME)
        logging.info(f"Initialized Chroma collection '{_VECTOR_INDEX_NAME}' with {_chroma_collection.count()} documents")
    except Exception as e:
        logging.error(f"Failed to init Chroma: {e}")
        raise Exception(f"Failed to init Chroma collection '{_VECTOR_INDEX_NAME}': {e}")


_local_embedding_model = None


def _load_local_embedding_model():
    global _local_embedding_model
    if _local_embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise Exception("Local embedding model not installed. Run 'pip install sentence-transformers'.")
        _local_embedding_model = SentenceTransformer(os.getenv('LOCAL_EMBED_MODEL', 'all-MiniLM-L6-v2'))
    return _local_embedding_model


def generate_embedding(text: str) -> List[float]:
    try:
        if os.getenv('USE_LOCAL_EMBEDDINGS', 'false').strip().lower() in ('1', 'true', 'yes'):
            model = _load_local_embedding_model()
            emb = model.encode(text)
            return emb.tolist() if hasattr(emb, 'tolist') else list(map(float, emb))

        if genai is not None:
            model_id = os.getenv('GENAI_EMBED_MODEL', 'text-embedding-004')
            candidates = [model_id, f"models/{model_id}"] if not model_id.startswith('models/') else [model_id, model_id.replace('models/', '', 1)]
            last_err: Optional[Exception] = None
            for m in candidates:
                try:
                    result = genai.embed_content(model=m, content=text)
                    if isinstance(result, dict) and 'embedding' in result:
                        return list(result['embedding'])
                    if hasattr(result, 'embedding'):
                        return list(result.embedding)
                    return list(result)
                except Exception as e:
                    last_err = e
                    emsg = str(e).lower()
                    if '404' in emsg or 'not found' in emsg:
                        logging.warning(f"Embedding model '{m}' not found, trying alternate")
                        continue
                    else:
                        break
            if last_err:
                logging.warning(f"Embedding API failed, falling back: {last_err}")

        model = _load_local_embedding_model()
        emb = model.encode(text)
        return emb.tolist() if hasattr(emb, 'tolist') else list(map(float, emb))
    except Exception as e:
        raise Exception(f"Embedding generation failed: {e}")


def store_vectors(vectors: List[Dict[str, Any]]) -> None:
    try:
        _init_chroma()
        if _chroma_collection is None:
            raise Exception("Chroma collection not initialized")
            
        ids = [v['id'] for v in vectors]
        embeddings = [v['values'] for v in vectors]
        metadatas = [v.get('metadata', {}) for v in vectors]
        documents = [m.get('text', '') for m in metadatas]
        
        logging.info(f"Storing {len(vectors)} vectors in ChromaDB")
        
        try:
            _chroma_collection.delete(ids=ids)
        except Exception:
            logging.info("Chroma delete() not supported; continuing")
            
        _chroma_collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
        logging.info(f"Successfully stored {len(vectors)} vectors")
    except Exception as e:
        logging.error(f"Failed to store vectors: {e}")
        raise Exception(f"Failed to store vectors: {e}")


def query_vectors(query_vector: List[float], top_k: int = 2, source: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        _init_chroma()
        if _chroma_collection is None:
            raise Exception("Chroma collection not initialized")
            
        logging.info(f"Querying vector DB with {len(query_vector)}-dim vector for top {top_k} matches" + (f" filtered by source='{source}'" if source else ""))
    
        query_kwargs = {
            'query_embeddings': [query_vector],
            'n_results': top_k,
            
            'include': ['metadatas', 'distances']
        }
        if source:
            try:
                query_kwargs['where'] = { 'source': source }
            except Exception:
                pass
        try:
            res = _chroma_collection.query(**query_kwargs)
        except TypeError:
            
            res = _chroma_collection.query(query_embeddings=[query_vector], n_results=top_k, include=['metadatas', 'distances'])
        
        matches = []
        if res['ids'] and res['ids'][0]:
            for i, id_ in enumerate(res['ids'][0]):
                meta = res['metadatas'][0][i]
                
                if source and meta.get('source') != source:
                    continue
                matches.append(
                    type('Match', (), {
                        'id': id_,
                        'score': (1.0 - res['distances'][0][i]) if res.get('distances') else None,
                        'metadata': meta
                    })
                )
            logging.info(f"Found {len(matches)} matches")
        else:
            logging.info("No matches found in vector DB")
            
        return matches
    except Exception as e:
        logging.error(f"Failed to query vectors: {e}")
        raise Exception(f"Failed to query vectors: {e}")


def _format_history(history: Optional[List[Dict[str, str]]]) -> str:
    if not history:
        return ""
    
    trimmed = history[-10:]
    lines: List[str] = []
    for item in reversed(trimmed):
        role = str(item.get('role', '')).lower()
        if role not in ('user', 'assistant'):
            role = 'user'  
        content = str(item.get('content', '')).strip()
        
        if len(content) > 1200:
            content = content[:1200] + '…'
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def generate_answer(question: str, context: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    if not genai:
        raise Exception("google.generativeai not available. Check GOOGLE_API_KEY.")

    history_text = _format_history(history)

    if context.strip():
        prompt = f"""You are a helpful AI assistant.
Use the provided document context to answer the user's question accurately. Prefer factual content from the context over general knowledge. If the context is insufficient, say so briefly. If the user is simply greeting (e.g., "hi", "hello") or making small talk, respond politely and briefly even without requiring document context.

Formatting requirements:
- Use Markdown for structure (headings, lists, tables).
- When user asks for "tabular form" or "table", ALWAYS create complete Markdown tables with proper formatting.
- Table format: | Header 1 | Header 2 | Header 3 | followed by | --- | --- | --- | then complete data rows.
- ENSURE tables are complete - do not truncate or cut off table content mid-generation.
- For comparison tables, include ALL comparison points in separate rows.
- Render ALL math using LaTeX delimiters: inline math with $...$ and display math with $$...$$.
- Do NOT use HTML tags like <sub>, <sup>, <i>, <b> for math; use LaTeX (e.g., F_{{x}}, x^{{2}}).

Conversation History (most recent first):
{history_text}

Document Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"""You are a helpful AI assistant.
Use the conversation history to keep continuity and answer the user's question. If the question is a follow-up, resolve references using the history. If the user is greeting or making small talk, respond naturally and briefly. If you still don't have enough information to answer a content question, ask a concise clarifying question.

Formatting requirements:
- Use Markdown for structure (headings, lists, tables).
- When user asks for "tabular form" or "table", ALWAYS create complete Markdown tables with proper formatting.
- Table format: | Header 1 | Header 2 | Header 3 | followed by | --- | --- | --- | then complete data rows.
- ENSURE tables are complete - do not truncate or cut off table content mid-generation.
- For comparison tables, include ALL comparison points in separate rows.
- Render ALL math using LaTeX delimiters: inline math with $...$ and display math with $$...$$.
- Do NOT use HTML tags like <sub>, <sup>, <i>, <b> for math; use LaTeX (e.g., F_{{x}}, x^{{2}}).

Conversation History (most recent first):
{history_text}

Question: {question}

Answer:"""

    base_name = os.getenv('GENAI_MODEL', 'gemini-2.5-flash-lite')
    variants = [base_name, f"{base_name}-latest" if not base_name.endswith("-latest") else base_name]
   
    preferred = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-latest",
        "gemini-2.5-flash",
        "gemini-2.5-flash-latest",
        "gemini-2.0-flash",
        "gemini-2.0-flash-latest",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
    ]
    seen, candidates = set(), []
    for b in variants + preferred:
        if b and (b not in seen):
            seen.add(b)
            if b.startswith('models/'):
                candidates.extend([b, b.replace('models/', '', 1)])
            else:
                candidates.extend([b, f'models/{b}'])

    last_error: Optional[Exception] = None
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 32,
                    "max_output_tokens": 4096,
                }
            )
            if hasattr(response, 'text') and response.text:
                return response.text
            if hasattr(response, 'candidates') and response.candidates:
                first = response.candidates[0]
                if hasattr(first, 'content') and getattr(first.content, 'parts', None):
                    texts = [getattr(p, 'text', None) for p in first.content.parts]
                    texts = [t for t in texts if t]
                    if texts:
                        return "\n".join(texts)
                return str(first)
            return str(response)
        except Exception as e:
            last_error = e
            emsg = str(e).lower()
            if '404' in emsg or 'not found' in emsg:
                logging.warning(f"Gemini model '{model_name}' not found, trying next")
                continue
            else:
                logging.warning(f"Gemini call failed: {e}")
                break
    if last_error:
        raise Exception(f"Gemini generate_content failed: {last_error}")

    raise Exception("No valid Gemini response")


def get_status() -> Dict[str, Any]:
    """Return basic health/status info for the RAG backend.
    Includes Chroma status and collection name if available.
    """
    status: Dict[str, Any] = {
        'chroma_ok': False,
        'chroma_dir': _CHROMA_DIR,
        'vector_index': _VECTOR_INDEX_NAME,
        'collection': None,
    }
    try:
        _init_chroma()
        if _chroma_collection is not None:
            status['chroma_ok'] = True
            
            status['collection'] = getattr(_chroma_collection, 'name', None)
    except Exception as e:
        status['error'] = str(e)
    return status


def _get_all_text_for_source(source: str) -> List[str]:
    """Retrieve all text chunks for a given source (filename) from Chroma."""
    _init_chroma()
    if _chroma_collection is None:
        raise Exception("Chroma collection not initialized")

    base = os.path.basename(source)
    texts: List[str] = []
    try:
       
        res = _chroma_collection.get(where={'source': base}, include=['documents', 'metadatas'])
        docs = res.get('documents') or []
        if docs:
            texts.extend([d for d in docs if d])
        else:
            metas = res.get('metadatas') or []
            for m in metas:
                if isinstance(m, dict) and m.get('text'):
                    texts.append(m['text'])
    except Exception:
        
        try:
            res = _chroma_collection.get(include=['documents', 'metadatas'])
            docs = res.get('documents') or []
            metas = res.get('metadatas') or []
            for i in range(max(len(docs), len(metas))):
                m = metas[i] if i < len(metas) else {}
                if isinstance(m, dict) and m.get('source') == base:
                    if i < len(docs) and docs[i]:
                        texts.append(docs[i])
                    elif m.get('text'):
                        texts.append(m['text'])
        except Exception as e:
            raise Exception(f"Failed to retrieve document texts: {e}")

   
    seen = set()
    uniq = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _summarize_text_block(context: str, length: str = 'short') -> str:
    if not genai:
        raise Exception("google.generativeai not available. Check GOOGLE_API_KEY.")

    guidance = {
        'short': 'Write a concise summary in 5-7 bullet points with a brief intro.',
        'medium': 'Write a structured summary with sections (Overview, Key Points, Details).',
        'detailed': 'Write a comprehensive, well-structured summary with headings, bullet lists, and key equations if present.'
    }.get(length, 'Write a concise summary in 5-7 bullet points with a brief intro.')

    prompt = f"""You are a helpful AI assistant.
Summarize the following document in a clear and structured way.

Requirements:
- {guidance}
- Use Markdown for structure (headings, lists, tables) when appropriate.
- Render ALL math using LaTeX delimiters: inline math with $...$ and display math with $$...$$.
- Do NOT use HTML tags like <sub>, <sup>, <i>, <b> for math; use LaTeX.

Document:
{context}

Summary:"""

    base_name = os.getenv('GENAI_MODEL', 'gemini-2.5-flash-lite')
    variants = [base_name, f"{base_name}-latest" if not base_name.endswith("-latest") else base_name]
    preferred = [
        "gemini-2.5-flash-lite", "gemini-2.5-flash-lite-latest",
        "gemini-2.5-flash", "gemini-2.5-flash-latest",
        "gemini-2.0-flash", "gemini-2.0-flash-latest",
        "gemini-1.5-flash", "gemini-1.5-flash-latest",
    ]
    seen, candidates = set(), []
    for b in variants + preferred:
        if b and (b not in seen):
            seen.add(b)
            if b.startswith('models/'):
                candidates.extend([b, b.replace('models/', '', 1)])
            else:
                candidates.extend([b, f'models/{b}'])

    last_error: Optional[Exception] = None
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 32,
                    "max_output_tokens": 512,
                }
            )
            if hasattr(response, 'text') and response.text:
                return response.text
            if hasattr(response, 'candidates') and response.candidates:
                first = response.candidates[0]
                if hasattr(first, 'content') and getattr(first.content, 'parts', None):
                    texts = [getattr(p, 'text', None) for p in first.content.parts]
                    texts = [t for t in texts if t]
                    if texts:
                        return "\n".join(texts)
                return str(first)
            return str(response)
        except Exception as e:
            last_error = e
            emsg = str(e).lower()
            if '404' in emsg or 'not found' in emsg:
                logging.warning(f"Gemini model '{model_name}' not found, trying next")
                continue
            else:
                logging.warning(f"Gemini call failed: {e}")
                break
    if last_error:
        raise Exception(f"Gemini generate_content failed: {last_error}")
    raise Exception("No valid Gemini response")


async def summarize_document(filename: str, length: str = 'short') -> Dict[str, Any]:
    """Summarize all chunks belonging to a processed PDF file."""
    try:
        texts = _get_all_text_for_source(filename)
        if not texts:
            raise Exception("No chunks found for this PDF. Upload/process it first.")

        combined = "\n\n".join(texts)
        max_chunk_chars = 6000  
        if len(combined) <= max_chunk_chars:
            summary = _summarize_text_block(combined, length)
        else:
            
            partials = []
            start = 0
            while start < len(combined):
                piece = combined[start:start+max_chunk_chars]
                partials.append(_summarize_text_block(piece, length))
                start += max_chunk_chars
            merged_context = "\n\n".join(partials)
            summary = _summarize_text_block(merged_context, length)

        return {"summary": summary, "length": length, "source": os.path.basename(filename)}
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        raise

async def process_pdf(pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
    from pdf_utils import extract_pages_with_ocr_fallback, split_into_chunks_by_page, extract_page_images

    try:
       
        ocr_lang = os.getenv('OCR_LANG', 'eng')
        pages, method = extract_pages_with_ocr_fallback(pdf_bytes, lang=ocr_lang)
        logging.info(f"PDF extraction method for '{filename}': {method}")
        if not any((p.get('text') or '').strip() for p in pages):
            raise Exception("PDF empty or unreadable after extraction")

        chunks_by_page = split_into_chunks_by_page(pages)
        if not chunks_by_page:
            raise Exception("No valid text chunks from PDF")

  
        page_images = {}
        try:
            if os.getenv('RENDER_PAGE_IMAGES', 'true').strip().lower() in ('1','true','yes'):
                page_images = extract_page_images(pdf_bytes, _MEDIA_ROOT, filename)
        except Exception as _:
            page_images = {}

        vectors = []
        for idx, chunk in enumerate(chunks_by_page):
            chunk_text = chunk['text']
            page_no = int(chunk.get('page') or 1)
            embedding = generate_embedding(chunk_text)
            vectors.append({
                'id': f"{filename}-p{page_no}-c{idx}",
                'values': embedding,
                'metadata': {
                    'text': chunk_text,
                    'source': os.path.basename(filename),
                    'page': page_no,
                    'image': page_images.get(page_no)
                }
            })
        store_vectors(vectors)
        return {"status": "ok", "filename": filename, "chunks": len(chunks_by_page)}
    except Exception as e:
        logging.error(f"PDF processing failed: {e}")
        raise Exception(f"PDF processing failed: {e}")



def _chunk_and_store_text(text: str, filename: str) -> Dict[str, Any]:
    try:
        from pdf_utils import split_into_chunks
        chunks = split_into_chunks(text, max_tokens=500)
        if not chunks:
            raise Exception("No textual content found to index")
        vectors = []
        for idx, chunk_text in enumerate(chunks):
            embedding = generate_embedding(chunk_text)
            vectors.append({
                'id': f"{filename}-c{idx}",
                'values': embedding,
                'metadata': {
                    'text': chunk_text,
                    'source': os.path.basename(filename),
                    'page': 1,
                }
            })
        store_vectors(vectors)
        return {"status": "ok", "filename": filename, "chunks": len(chunks)}
    except Exception as e:
        raise Exception(f"Text ingestion failed: {e}")


async def process_docx(docx_bytes: bytes, filename: str) -> Dict[str, Any]:
    try:
        try:
            import docx
        except Exception as e:
            raise Exception("python-docx not installed. Add 'python-docx' to requirements.")
        document = docx.Document(BytesIO(docx_bytes))
        paras = []
        for p in document.paragraphs:
            t = (p.text or '').strip()
            if t:
                paras.append(t)
        # Include tables
        for table in getattr(document, 'tables', []) or []:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                line = ' | '.join([c for c in cells if c])
                if line:
                    paras.append(line)
        full_text = '\n'.join(paras)
        return _chunk_and_store_text(full_text, filename)
    except Exception as e:
        raise Exception(f"DOCX processing failed: {e}")


async def process_csv(csv_bytes: bytes, filename: str) -> Dict[str, Any]:
    try:
        import pandas as pd
        bio = BytesIO(csv_bytes)
        try:
            df = pd.read_csv(bio, dtype=str, encoding='utf-8', on_bad_lines='skip')
        except Exception:
            bio.seek(0)
            df = pd.read_csv(bio, dtype=str, encoding='latin1', on_bad_lines='skip')
        # Convert to text
        lines = []
        cols = list(df.columns)
        if cols:
            lines.append(' | '.join(map(str, cols)))
        for _, row in df.iterrows():
            vals = [str(row.get(c, '')) for c in cols]
            lines.append(' | '.join(vals))
        text = '\n'.join(lines)
        return _chunk_and_store_text(text, filename)
    except Exception as e:
        raise Exception(f"CSV processing failed: {e}")


async def process_xlsx(xlsx_bytes: bytes, filename: str) -> Dict[str, Any]:
    try:
        import pandas as pd
        bio = BytesIO(xlsx_bytes)
        # Read first sheet by default
        df = pd.read_excel(bio, dtype=str, engine='openpyxl')
        lines = []
        cols = list(df.columns)
        if cols:
            lines.append(' | '.join(map(str, cols)))
        for _, row in df.iterrows():
            vals = [str(row.get(c, '')) for c in cols]
            lines.append(' | '.join(vals))
        text = '\n'.join(lines)
        return _chunk_and_store_text(text, filename)
    except Exception as e:
        raise Exception(f"XLSX processing failed: {e}")


async def process_png(image_bytes: bytes, filename: str, ocr_lang: str = None) -> Dict[str, Any]:
    try:
        from PIL import Image
        import pytesseract
        ocr_lang = ocr_lang or os.getenv('OCR_LANG', 'eng')
        img = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(img, lang=ocr_lang) or ''
        text = text.strip()
        if not text:
            raise Exception("No text recognized in image")
        return _chunk_and_store_text(text, filename)
    except Exception as e:
        raise Exception(f"PNG processing failed: {e}")


async def ask_question(question: str, pdf_name: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    logging.info(f"Processing question: {question[:100]}...")
    try:
        logging.info(f"Q: {question}")
    except Exception:
        pass

    # Handle general greetings without invoking RAG
    try:
        qnorm = (question or "").strip().lower()
        if qnorm:
            if (
                re.search(r"^(hi+|hii+|hiii+|hello+|hey+|yo+|hola|namaste)\b", qnorm)
                or re.search(r"^good\s+(morning|afternoon|evening|night)\b", qnorm)
            ):
                friendly = "Hello! I'm your PDF Q&A assistant. Ask about the current document."
                try:
                    logging.info(f"A (greeting): {friendly}")
                except Exception:
                    pass
                return {
                    'answer': friendly,
                    'sources': [],
                    'note': 'Greeting received; responded without RAG.'
                }
    except Exception:
        # Non-fatal; proceed with normal flow
        pass

    # Heuristic: if this is a follow-up like "do it again", "explain again", etc.,
    # use the last significant USER message as the retrieval query so we stay on the same problem.
    def _is_followup(q: str) -> bool:
        qn = (q or '').strip().lower()
        if len(qn) <= 4:
            return True
        patterns = [
            r"\bagain\b",
            r"\brepeat\b",
            r"\bre\-?do\b",
            r"\bdo (it|this) again\b",
            r"\bexplain (it|this) again\b",
            r"\bsame (one|problem|question)\b",
            r"\bcontinue\b",
            r"\bmore detail\b",
            r"\bstep\s*by\s*step\b",
            r"\bgive it again\b",
            r"\bshow (it|this|that) again\b",
            r"\btabular form\b",
            r"\bin table\b",
            r"\bas (a )?table\b",
        ]
        return any(re.search(p, qn) for p in patterns)

    def _last_significant_user(hist: Optional[List[Dict[str, str]]]) -> Optional[str]:
        try:
            if not hist:
                return None
            for item in reversed(hist):  # most recent first
                if str(item.get('role','')).lower() == 'user':
                    content = (item.get('content') or '').strip()
                    if not content:
                        continue
                    # Significant if long enough or has digits/equations or a question mark
                    if len(content) >= 25 or re.search(r"[0-9]", content) or ('?' in content):
                        return content
            # fallback: first user we find
            for item in reversed(hist):
                if str(item.get('role','')).lower() == 'user':
                    return (item.get('content') or '').strip()
        except Exception:
            return None
        return None

    retrieval_text = question
    if _is_followup(question):
        prev_q = _last_significant_user(history)
        if prev_q:
            retrieval_text = prev_q
            logging.info("Follow-up detected; using last significant user question for retrieval")

    # Generate embedding for retrieval text (question or prior question when follow-up)
    question_embedding = generate_embedding(retrieval_text)
    logging.info(f"Generated question embedding ({len(question_embedding)} dimensions)")

    # Query vector DB for relevant chunks
    # Normalize source name for matching
    norm_source = os.path.basename(pdf_name) if pdf_name else None
    matches = query_vectors(question_embedding, source=norm_source)
    logging.info(f"Found {len(matches)} matching chunks" + (f" for source='{norm_source}'" if norm_source else ""))

    if not matches:
        try:
            answer_text = generate_answer(question, "", history)
            try:
                preview = (answer_text or "")
                preview = preview if len(preview) <= 400 else preview[:400] + "…"
                logging.info(f"A (no-context): {preview}")
            except Exception:
                pass
            return {
                'answer': answer_text,
                'sources': [],
                'note': 'No documents matched; answered directly with Gemini.'
            }
        except Exception as e:
            return {
                'answer': 'No relevant info found and no LLM response available.',
                'sources': [],
                'error': str(e),
                'hint': 'Upload documents via /upload or check GOOGLE_API_KEY.'
            }

    context = '\n\n'.join(match.metadata['text'] for match in matches)
    answer = generate_answer(question, context, history)
    try:
        preview = (answer or "")
        preview = preview if len(preview) <= 400 else preview[:400] + "…"
        logging.info(f"A: {preview}")
    except Exception:
        pass

    # Decide whether to include citations: include for explicit requests or complex/important answers
    qnorm = (question or '').lower()
    keyword_trigger = any(k in qnorm for k in ['cite', 'citation', 'reference', 'source', 'page', 'where', 'quote', 'figure', 'diagram'])
    top_score = None
    try:
        top_score = max((m.score for m in matches if m.score is not None), default=None)
    except Exception:
        top_score = None
    score_trigger = (top_score is not None and top_score >= 0.62)

    # Heuristics based on the generated answer
    ans = (answer or '')
    long_answer = len(ans) >= 250
    has_math = ('$' in ans) or ('∑' in ans) or bool(re.search(r"\\b=[^=]", ans))
    important_words = ['example', 'derive', 'derivation', 'explain', 'explanation', 'solve', 'worked', 'equation', 'tension', 'proof']
    answer_trigger = any(w in ans.lower() for w in important_words)
    multi_match = len(matches) >= 2

    include_citations = keyword_trigger or score_trigger or (
        long_answer and (has_math or answer_trigger or multi_match)
    )

    sources_payload = []
    if include_citations:
        try:
            ordered = sorted(
                matches,
                key=lambda m: ((m.score is not None), (m.score or 0.0)),
                reverse=True,
            )
        except Exception:
            ordered = list(matches)
        top2 = ordered[:2]
        sources_payload = [{
            'chunk_id': match.id,
            'page': match.metadata.get('page'),
            'text': match.metadata['text'][:200] + '...',
            'image': match.metadata.get('image'),
            'pdf': match.metadata.get('source')
        } for match in top2]

    return {
        'answer': answer,
        'sources': sources_payload
    }


def _build_prompt(question: str, context: str, history: Optional[List[Dict[str, str]]]) -> str:
    history_text = _format_history(history)
    if context.strip():
        prompt = f"""You are a helpful AI assistant.
Use the provided document context to answer the user's question accurately. Prefer factual content from the context over general knowledge. If the context is insufficient, say so briefly. If the user is simply greeting (e.g., "hi", "hello") or making small talk, respond politely and briefly even without requiring document context.

Formatting requirements:
- Use Markdown for structure (headings, lists, tables).
- When user asks for "tabular form" or "table", ALWAYS create complete Markdown tables with proper formatting.
- Table format: | Header 1 | Header 2 | Header 3 | followed by | --- | --- | --- | then complete data rows.
- ENSURE tables are complete - do not truncate or cut off table content mid-generation.
- For comparison tables, include ALL comparison points in separate rows.
- Render ALL math using LaTeX delimiters: inline math with $...$ and display math with $$...$$.
- Do NOT use HTML tags like <sub>, <sup>, <i>, <b> for math; use LaTeX (e.g., F_{{x}}, x^{{2}}).

Conversation History (most recent first):
{history_text}

Document Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"""You are a helpful AI assistant.
Use the conversation history to keep continuity and answer the user's question. If the question is a follow-up, resolve references using the history. If the user is greeting or making small talk, respond naturally and briefly. If you still don't have enough information to answer a content question, ask a concise clarifying question.

Formatting requirements:
- Use Markdown for structure (headings, lists, tables).
- When user asks for "tabular form" or "table", ALWAYS create complete Markdown tables with proper formatting.
- Table format: | Header 1 | Header 2 | Header 3 | followed by | --- | --- | --- | then complete data rows.
- ENSURE tables are complete - do not truncate or cut off table content mid-generation.
- For comparison tables, include ALL comparison points in separate rows.
- Render ALL math using LaTeX delimiters: inline math with $...$ and display math with $$...$$.
- Do NOT use HTML tags like <sub>, <sup>, <i>, <b> for math; use LaTeX (e.g., F_{{x}}, x^{{2}}).

Conversation History (most recent first):
{history_text}

Question: {question}

Answer:"""
    return prompt


def ask_question_stream(question: str, pdf_name: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None):
    """Return a generator that yields answer text chunks as they are produced by the model."""
    if not genai:
        raise Exception("google.generativeai not available. Check GOOGLE_API_KEY.")

    # Retrieval (same as non-stream)
    question_embedding = generate_embedding(question)
    norm_source = os.path.basename(pdf_name) if pdf_name else None
    matches = query_vectors(question_embedding, source=norm_source)
    context = '\n\n'.join(m.metadata['text'] for m in matches) if matches else ''
    prompt = _build_prompt(question, context, history)

    # Model fallbacks
    base_name = os.getenv('GENAI_MODEL', 'gemini-2.5-flash-lite')
    variants = [base_name, f"{base_name}-latest" if not base_name.endswith("-latest") else base_name]
    preferred = [
        "gemini-2.5-flash-lite", "gemini-2.5-flash-lite-latest",
        "gemini-2.5-flash", "gemini-2.5-flash-latest",
        "gemini-2.0-flash", "gemini-2.0-flash-latest",
        "gemini-1.5-flash", "gemini-1.5-flash-latest",
    ]
    seen, candidates = set(), []
    for b in variants + preferred:
        if b and (b not in seen):
            seen.add(b)
            if b.startswith('models/'):
                candidates.extend([b, b.replace('models/', '', 1)])
            else:
                candidates.extend([b, f'models/{b}'])

    last_error: Optional[Exception] = None
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name)
            stream = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "top_k": 32,
                    "max_output_tokens": 4096,
                },
                stream=True,
            )
            for chunk in stream:
                text = getattr(chunk, 'text', None)
                if text:
                    yield text
            return
        except Exception as e:
            last_error = e
            emsg = str(e).lower()
            if '404' in emsg or 'not found' in emsg:
                logging.warning(f"Gemini model '{model_name}' not found, trying next")
                continue
            else:
                logging.warning(f"Gemini streaming call failed: {e}")
                break
    if last_error:
        raise Exception(f"Gemini streaming failed: {last_error}")


async def get_sources(question: str, pdf_name: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Return sources only (no LLM call). Uses keyword/score triggers to decide inclusion, capped to top 2."""
    question_embedding = generate_embedding(question)
    norm_source = os.path.basename(pdf_name) if pdf_name else None
    matches = query_vectors(question_embedding, source=norm_source)
    qnorm = (question or '').lower()
    keyword_trigger = any(k in qnorm for k in ['cite', 'citation', 'reference', 'source', 'page', 'where', 'quote', 'figure', 'diagram'])
    top_score = None
    try:
        top_score = max((m.score for m in matches if m.score is not None), default=None)
    except Exception:
        top_score = None
    score_trigger = (top_score is not None and top_score >= 0.62)
    include = keyword_trigger or score_trigger or len(matches) >= 2
    sources_payload = []
    if include and matches:
        ordered = sorted(matches, key=lambda m: ((m.score is not None), (m.score or 0.0)), reverse=True)
        top2 = ordered[:2]
        sources_payload = [{
            'chunk_id': m.id,
            'page': m.metadata.get('page'),
            'text': m.metadata['text'][:200] + '...',
            'image': m.metadata.get('image'),
            'pdf': m.metadata.get('source')
        } for m in top2]
    return { 'sources': sources_payload }
