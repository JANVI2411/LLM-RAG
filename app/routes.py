from fastapi import FastAPI, UploadFile, File
from src.document_loader import extract_text_from_pdf
from src.rag_pipeline import answer_query
from LLMOps.src.rag.chunker import chunk_text

app = FastAPI()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)
    
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    
    return {"num_chunks": len(chunks), "first_chunk": chunks[0]}


@app.get("/ask/")
def ask_query(q: str):
    answer = answer_query(q)
    return {"query": q, "answer": answer}

