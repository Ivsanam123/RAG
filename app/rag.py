from typing import List, Tuple
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.docstore.document import Document

from langchain_core .globals import set_llm_cache
from langchain_redis import RedisSemanticCache

from .utils import get_vector_store

from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever




SYSTEM = """You are a grounded company knowledge assistant.
Always base answers strictly on the provided context.
If the answer isn't present, reply with "I don't know."
Respond concisely and clearly.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("user",
     "Question:\n{input}\n\n" #from where are we getting the input?
     "Context:\n{context}\n\n"
     "Rule: Prefer the most recent policy by effective date.")
])

async def _build_chain(): #this is only called once so we dont havet ot build the chain every time??
    store = await get_vector_store()
    retriever = store.as_retriever(search_kwargs={"k":os.getenv("RETRIEVAL_K")})
    llm = ChatOpenAI(model = "gpt-4o-mini") #here llm is a callable object which wraps the OpenAI API. 
    doc_chain = create_stuff_documents_chain(llm, PROMPT) #where is he getting all of these functions is it langchani, do i need to learn langchain for placements, 
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    return rag_chain

async def answer_with_docs_async(question: str) -> Tuple[str, List[str]]:
    chain = await _build_chain()
    result = await chain.ainvoke({"input":question})
    answer = result["answer"]

    sources = []
    docs:List[Document] = result["context"]
    unique_sources = {d.metadata.get("source") for d in docs}
    sources = sorted(unique_sources)

    return answer,sources