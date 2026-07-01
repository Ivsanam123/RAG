from __future__ import annotations
import os, glob, uuid, asyncio, traceback
from typing import Iterable, List, Dict, Any
from pathlib import Path

from langchain_classic.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader,TextLoader

from .utils import get_vector_store
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy


DATA_DIR = os.getenv("DATA_DIR", "data")

def _load_docs(base: str = DATA_DIR) -> List[Document]:
    docs: List[Document] = []#empty list #the syntax: docs is the name of a variable and the : means "is of type" List implie a python list of Document implies every single instance of the class is of type Document, we do this because python is dynamically types

    # recurse through all files under base
    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):#this follows through the database recursively and checks through the documents 
        if os.path.isdir(path) or os.path.basename(path).startswith("."):#if it is a directory we skip that, we only continue through the loop if its a direct file or pdf
            continue
        ext = os.path.splitext(path)[1].lower()#we get the extention of the file name so that we can process each type separately

        relative_path = os.path.relpath(path,base)#it takes the relative path... here it would be announcements\merger-new.pdf so we are splitting it at the separation which is \ in windows, and we put in category in the next line
        category = relative_path.split(os.sep)[0] if os.sep in relative_path else "general"

        try:
            loaded_docs = []
            if ext == ".md":#for mark down files
                for d in UnstructuredMarkdownLoader(path).load(): #imported form langchain documetn loaders, path is the file path, load does the magic taking one page from the file at a time and converting it into a langCHain structure as required
                    loaded_docs.append(d)
            elif ext == ".pdf":#for pdf files
                for d in PyMuPDFLoader(path).load():
                    loaded_docs.append(d)
            elif ext == ".docx":
                for d in UnstructuredWordDocumentLoader(path).load():
                    loaded_docs.append(d)
            elif ext == ".txt":
                for d in TextLoader(path).load():
                    loaded_docs.append(d)
            
            for d in loaded_docs:
                d.metadata["category"] = category #the document object from langChain consists of page_content which is the actual text, and a .metadata which is a dictionary holding the extra information
                docs.append(d)
        except Exception:
            print(f"INGEST ERROR: failed to load {path}")
            traceback.print_exc()

    return docs
        

def _chunk(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(#the text splitter is for 
        chunk_size = 900,
        chunk_overlap = 120
    )
    try:
        return splitter.split_documents(docs)
    except Exception:
        print(f"INGEST ERROR: chunking failed")
        traceback.print_exc()
        raise

async def run_ingest_async() -> dict: #completing the vectorization pipeline
    docs = _load_docs()#get the docs
    chunks = _chunk(docs)#split into chunks
    store = await get_vector_store()#we need the vecotr store
    await store.aadd_documents(chunks)#asynchronous add documents into the vecotr database
    print(f"INGEST: {len(docs)} docs, {len(chunks)} chunks")#so the ui expects the length of the docs and the chunks

    return {"documents": len(docs),"chunks":len(chunks)}

