from fastapi import FastAPI, UploadFile, HTTPException
from pathlib import Path
import shutil
from langchain_community.document_loaders import PyMuPDFLoader

app = FastAPI()

@app.post("/summarize/")
async def create_upload_file(file: UploadFile):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="file type not supported")

    temp_path = Path(f"tmp/{file.filename}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    loader = PyMuPDFLoader(str(temp_path))
    docs = loader.load()

    

    return {"total_docs": len(docs)}
