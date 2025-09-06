from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, List, Optional
import os
from dotenv import load_dotenv
import logging
from fastapi.staticfiles import StaticFiles

from rag import (
    process_pdf,
    process_docx,
    process_csv,
    process_xlsx,
    process_png,
    ask_question,
    get_status,
    summarize_document,
    ask_question_stream,
    get_sources,
)
import json

load_dotenv()

# Ensure INFO-level logs are printed to the terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Document Q&A API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_ROOT = os.getenv("MEDIA_ROOT", os.path.join(os.path.dirname(__file__), "media"))
os.makedirs(MEDIA_ROOT, exist_ok=True)
app.mount("/media", StaticFiles(directory=MEDIA_ROOT), name="media")

@app.post("/test-upload")
async def test_upload(file: UploadFile):
    return {"filename": file.filename, "content_type": file.content_type}

class HistoryMessage(BaseModel):
    role: str
    content: str

class QuestionRequest(BaseModel):
    question: str
    pdfName: str | None = None
    pastMessages: Optional[List[HistoryMessage]] = None

class SummarizeRequest(BaseModel):
    pdfName: str
    summary_length: Optional[str] = 'short'

@app.post("/upload")
async def upload_document(file: UploadFile) -> Dict[str, Any]:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = {".pdf", ".docx", ".csv", ".xlsx", ".xls", ".png"}
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Allowed: PDF, DOCX, CSV, XLSX/XLS, PNG")
    
    try:
        contents = await file.read()
        
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        try:
            dest_path = os.path.join(MEDIA_ROOT, file.filename)
            with open(dest_path, 'wb') as f:
                f.write(contents)
        except Exception:
            pass
            
        if ext == ".pdf":
            result = await process_pdf(contents, file.filename)
        elif ext == ".docx":
            result = await process_docx(contents, file.filename)
        elif ext == ".csv":
            result = await process_csv(contents, file.filename)
        elif ext in (".xlsx", ".xls"):
            result = await process_xlsx(contents, file.filename)
        elif ext == ".png":
            result = await process_png(contents, file.filename)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'")
            
        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "data": result
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"Upload processing error: {error_msg}")
        
        if "Failed to extract text" in error_msg:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        elif "Vector storage" in error_msg:
            raise HTTPException(status_code=500, detail="Vector storage error")
        elif "AI processing" in error_msg:
            raise HTTPException(status_code=500, detail="AI processing error")
        else:
            raise HTTPException(status_code=500, detail=f"Upload failed: {error_msg}")

@app.post("/ask_stream")
async def ask_stream(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        history = [{"role": m.role, "content": m.content} for m in (request.pastMessages or [])]
        pre_sources = await get_sources(request.question, request.pdfName, history)

        def _sse():
            try:
                for chunk in ask_question_stream(request.question, request.pdfName, history):
                    yield f"data: {json.dumps({'type':'text','delta': chunk})}\n\n"
                yield f"data: {json.dumps({'type':'done','sources': pre_sources.get('sources', [])})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','message': str(e)})}\n\n"

        return StreamingResponse(
            _sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(request: QuestionRequest) -> Dict[str, Any]:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
        
    try:
        history = [{"role": m.role, "content": m.content} for m in (request.pastMessages or [])]
        result = await ask_question(request.question, request.pdfName, history)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        status = get_status()
        return {"status": "ok", **status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/summarize")
async def summarize(req: SummarizeRequest) -> Dict[str, Any]:
    if not req.pdfName:
        raise HTTPException(status_code=400, detail="pdfName is required")
    try:
        result = await summarize_document(req.pdfName, req.summary_length or 'short')
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )
