from fastapi import FastAPI, UploadFile, HTTPException
from pathlib import Path
import shutil
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize.chain import load_summarize_chain
from langchain_core.documents import Document

app = FastAPI()

@app.post("/summarize/")
async def summarize_pdf(file: UploadFile):
    
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File type not supported")

    temp_path = Path(f"tmp/{file.filename}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = PyMuPDFLoader(str(temp_path))
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        all_splits = text_splitter.split_documents(docs)

        llm = OllamaLLM(
            model="llama3.1",
            temperature=0.1,
            base_url="http://localhost:11434"  # Add base URL
        )

        map_prompt_template = """Write a concise summary of the following:
        {text}
        CONCISE SUMMARY:"""

        map_prompt = PromptTemplate(
            template=map_prompt_template,
            input_variables=["text"]
        )

        combine_prompt_template = """Write a comprehensive summary of the following text by synthesizing the key points from individual summaries.
        
        Individual summaries:
        {text}
        
        Comprehensive summary:"""

        combine_prompt = PromptTemplate(
            template=combine_prompt_template,
            input_variables=["text"]
        )

        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False,
            return_intermediate_steps=False
        )

        result = chain.invoke({"input_documents": all_splits})
        
        if temp_path.exists():
            temp_path.unlink()
        
        summary_text = result.get("output_text", "") if isinstance(result, dict) else str(result)
        
        return {
            "total_chunks": len(all_splits),
            "summary": summary_text.strip()
        }
        
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")