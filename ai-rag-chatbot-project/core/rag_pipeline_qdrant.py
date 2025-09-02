import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing.loader import Load
from processing.spliter import Spliter
from processing.embedding import Embedding_VectorStore
from core.llm_manager import LLMManager
from core.retriever_reranker_qdrant import Retriever_Reranker_Qdrant
import logging
from dotenv import load_dotenv
from typing import Any, Dict
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from qdrant.qdrant_upload import QdrantUploader


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

class RAGPipelineInteractive_Qdrant:
    def __init__(self, qdrant_vectorstore, llm_manager: LLMManager, retriever_reranker_qdrant):
        self.qdrant_vectorstore = qdrant_vectorstore
        self.llm_manager = llm_manager
        self.retriever_reranker_qdrant = Retriever_Reranker_Qdrant(qdrant_vectorstore)
        # self.retriever = self.setup_retriever()
        # self.chain = self.setup_chain()
        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.llm_manager.get_rag_prompt()
            | self.llm_manager.llm
            | StrOutputParser()
        )
        
        
    
    def get_response(self, question, use_reranker=False):
        """
        주어진 질문에 대해 RAG 응답 생성.
        use_reranker=True이면 cross encoder reranker 적용.
        
        """
        try:
            if use_reranker:
                logger.info("Applying Cross Encoder Reranker...")
                retriever_docs = self.retriever_reranker_qdrant.cross_encoder_reranker_qdrant(question)
          
            else:
                logger.info("Using Qdrant Retriever...")
                retriever_docs = self.retriever_reranker_qdrant.get_qdrant_retriever().get_relevant_documents(question)
              
            if not retriever_docs:
                logger.warning("No documents retrieved for the question.")
                return "죄송합니다. 관련 문서를 찾을 수 없습니다."
            
            context_text = "\n\n".join([doc.page_content for doc in retriever_docs])

            
            return self.chain.invoke({"context": context_text, "question": question})
        
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            raise
        
      

            
