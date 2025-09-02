import os
from dotenv import load_dotenv
from processing.loader import Load
from processing.spliter import Spliter
from processing.embedding import Embedding_VectorStore
from core.llm_manager import LLMManager
from core.retriever_reranker import Retriever_Reranker
from core.rag_pipeline import RAGPipelineInteractive
from qdrant.qdrant_upload import Qdrant

load_dotenv()

def main():
    # API 키
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "data")
    vs_dir = os.path.join(BASE_DIR, "vectorstores")
    os.makedirs(vs_dir, exist_ok=True)
    
    # 1. PDF 로드
    loader = Load()
    docs = loader.load_directory_data(data_dir)

    # 2. 문서 분할
    spliter = Spliter(docs)
    # [*==선택가능==*] recursive split / Semantic Similarity(의미적 유사성 기준 텍스트 분할) 선택가능
    split_data = spliter.split_data_recursive()  # recursive split 사용
    # split_data = spliter.split_data_semantic()  #  Semantic Similarity 

    # 3. 벡터스토어 생성
    emb_vectorstore = Embedding_VectorStore(loader)
    # [*==선택가능==*]유료 임베딩/무료 임베딩 선택가능
    # (유료: OpenAIEmbeddings / 무료: HuggingFaceBgeEmbeddings, FastEmbedEmbeddings )
    vectorstore = emb_vectorstore.create_vectorstore(split_data, use_paid=True)
    # vectorstore = emb_vectorstore.create_vectorstore(split_data, use_paid=True)

    # 4. Retriever 초기화
    retriever_reranker = Retriever_Reranker(vectorstore)
    # retriever.get_faiss_retriever(k=5)
    
    # # Vectorstore Retriever 실행
    # print("\n=== Vectorstore Retriever ===")
    # for q in query_list:
    #     retriever.use_vectorstore_retriever(q)
    
    # # Ensemble Retriever 실행
    # print("\n=== Ensemble Retriever ===")
    # for q in query_list:
    #     retriever.use_ensemble_retriever(q)
        
    # # Cross Encoder Reranker 실행
    # print("\n=== Cross Encoder Reranker ===")
    # for q in query_list:  
    #     retriever.cross_encoder_reranker(q)

    # 5. LLM Manager 초기화
    llm_manager = LLMManager(api_key)

    # 6. RAG Pipeline 초기화
    rag_pipeline = RAGPipelineInteractive(vectorstore, llm_manager, retriever_reranker)

    print("=======AI RAG Chatbot. 종료하려면 'exit' 입력=======")

    while True:
        user_input = input("\n[User]: ")
        if user_input.lower() in ["exit", "quit"]:
            print("대화를 종료합니다.")
            break
        
        # [*==선택가능==*] 7.Retriever적용선택 가능
        # => vectoresotre retriever, FAISS retriever, ensemble retriever[bm25,FAISS, ensemble ], cross encoder reranker 적용 선택 가능
        # reranked_docs = retriever_reranker.cross_encoder_reranker(user_input)
        reranked_docs = retriever_reranker.get_faiss_retriever(user_input)
        # reranked_docs = retriever_reranker.use_vectorstore_retriever(user_input)
        # reranked_docs = retriever_reranker.use_ensemble_retriever(user_input)

        # [*==선택가능==*] 8. 질문 + RAG Pipeline 응답
        # => reranker 사용(True)/사용안함(False) 선택 가능
        # answer = rag_pipeline.get_response(user_input, use_reranker=True)
        answer = rag_pipeline.get_response(user_input, use_reranker=False)
        
        print(f"[AI]: {answer}")

if __name__ == "__main__":
    main()