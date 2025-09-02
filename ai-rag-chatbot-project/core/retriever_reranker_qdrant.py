import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing.loader import Load
from processing.spliter import Spliter
from processing.embedding import Embedding_VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from qdrant.qdrant_upload import QdrantUploader

class Retriever_Reranker_Qdrant : 
    def __init__(self, qdrant_vectorstore):
        self.qdrant_vectorstore = qdrant_vectorstore
        
    def use_qdrant_vectorstore_retriever(self, query):
      
        retriever = self.qdrant_vectorstore.as_retriever(search_type="similarity", 
                                            search_kwargs={"k": 5})
        
        search_result =retriever.get_relevant_documents(query)
        
        if not search_result:
            print(f"관련된 질문 정보를 찾을 수 없습니다 : {query}")
        else:
            print(f"[검색 쿼리] : {query}")
            print(f"[검색 결과] : {search_result}")
            print(f"[검색 결과 개수] : {len(search_result)}")
            for idx, doc in enumerate(search_result):
                print(f"[Result {idx+1}] {doc.page_content[:200]}...")
        
        return f"[qdrant_vectorstore검색 결과] : {search_result}"
    

    def get_qdrant_retriever(self, k=5):
            return self.qdrant_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


    """cross encoder Reranker: RAG시스템 성능향상 시키기 위한 기술
    - 검색된 문서들의 순위를 재조정하여 질문에 가장 관련성 높은 문서를 상위로 올림
    """
    def cross_encoder_reranker_qdrant(self, query):
        # 모델 초기화
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        # 상위 n개의 문서 선택
        compressor = CrossEncoderReranker(model=model, top_n=3)
        
        # faiss_retriever = self.get_faiss_retriever()
        retriever = self.qdrant_vectorstore.as_retriever(search_type="similarity", 
                                            search_kwargs={"k": 5})
        # 문서 압축 검색기 초기화
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever )

        # 압축된 문서 검색
        print(f"[Query]: {query}")
        compressed_docs = compression_retriever.get_relevant_documents(query)
        
        for doc in compressed_docs:
            print(f" =>[Reranker적용] : {doc.page_content[:80]}")
        return compressed_docs
        
    
