import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing.loader import Load
from processing.spliter import Spliter
from processing.embedding import Embedding_VectorStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

class Retriever_Reranker: 
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        
    def use_vectorstore_retriever(self, query):
        # mmr : maximum marginal search result 를 사용하여 검색
        # - k: 반환할 문서 수 (기본값: 4)
        # - fetch_k: MMR 알고리즘에 전달할 문서 수 (기본값: 20)
        # - lambda_mult: MMR 결과의 다양성 조절 (0~1, 기본값: 0.5, 0: 유사도 점수만 고려, 1: 다양성만 고려)
        retriever = self.vectorstore.as_retriever(search_type="mmr", 
                                            search_kwargs={"k": 5, "fetch_k": 10})
        
        # 유사도 기반 검색 : score_threshold 이상인 결과만 반환
        # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", 
        #                                      search_kwargs={"score_threshold": 0.8})
        
        search_result =retriever.get_relevant_documents(query)
        
        if not search_result:
            print(f"관련된 질문 정보를 찾을 수 없습니다 : {query}")
        else:
            print(f"[검색 쿼리] : {query}")
            print(f"[검색 결과] : {search_result}")
            print(f"[검색 결과 개수] : {len(search_result)}")
            for idx, doc in enumerate(search_result):
                print(f"[Result {idx+1}] {doc.page_content[:200]}...")
        
        return f"[vectorstore검색 결과] : {search_result}"

    def get_faiss_retriever(self, k=5):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


    def use_ensemble_retriever(self, query):    
        # vectorstore 안의 원본 문서 가져오기
        docs = self.vectorstore.docstore._dict.values()
        
        # BM25 retriever(키워드 검색)
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 2
        
        # Faiss retriever(백터 검색)
        faiss_retriever = self.get_faiss_retriever()
        
        # Ensemble retriever
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], 
                                            weights=[0.5, 0.5])
        
        print(f"[Query]: {query}")
        bm25_relevant_docs = bm25_retriever.get_relevant_documents(query)
        faiss_relevant_docs = faiss_retriever.get_relevant_documents(query)
        ensemble_relevant_docs = ensemble_retriever.get_relevant_documents(query)
        
        print(f"=======[BM25_Retriever]:======= ")
        for doc in bm25_relevant_docs:
            print(f"=> {doc.page_content[:80]}")
            
        print(f"=======[FAISS_Retriever]:======= ")
        for doc in faiss_relevant_docs:
            print(f"=> {doc.page_content[:80]}")
            
        print(f"=======[Ensemble_Retriever]:======= ")
        for doc in ensemble_relevant_docs:
            print(f"=> {doc.page_content[:80]}")
        
        return {
            "bm25": bm25_relevant_docs,
            "faiss": faiss_relevant_docs,
            "ensemble": ensemble_relevant_docs
                }


    """cross encoder Reranker: RAG시스템 성능향상 시키기 위한 기술
    - 검색된 문서들의 순위를 재조정하여 질문에 가장 관련성 높은 문서를 상위로 올림
    """
    def cross_encoder_reranker(self, query):
        # 모델 초기화
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        # 상위 n개의 문서 선택
        compressor = CrossEncoderReranker(model=model, top_n=3)
        
        faiss_retriever = self.get_faiss_retriever()
        # 문서 압축 검색기 초기화
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=faiss_retriever )

        # 압축된 문서 검색
        print(f"[Query]: {query}")
        compressed_docs = compression_retriever.get_relevant_documents(query)
        for doc in compressed_docs:
            print(f" =>[Reranker적용] : {doc.page_content[:80]}")
        return compressed_docs
        
    
