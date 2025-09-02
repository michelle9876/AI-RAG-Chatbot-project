import os
from dotenv import load_dotenv
from processing.loader import Load
from processing.spliter import Spliter
from core.llm_manager import LLMManager
from core.retriever_reranker_qdrant import Retriever_Reranker_Qdrant
from core.rag_pipeline_qdrant import RAGPipelineInteractive_Qdrant
from qdrant.qdrant_upload import QdrantUploader
from processing.loader_sensitive_data import Load_Sensitive_Data

load_dotenv()

def main_qrant():
    # API 키
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

    
    # 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # 패턴 기반 찾기
    PATTERNS = {
        # 1) Phone Numbers (한국 + 국제 번호 포함)
        "phone": r"""
            (?:\+?\d{1,3}[-.\s]?)?   # 국제번호 (+1, +82, +44 등) 선택
            (?:\(?\d{1,4}\)?[-.\s]?) # 지역번호/앞자리 (02, 202, 010 등)
            \d{3,4}[-.\s]?\d{4}      # 나머지 번호
        """,
        # 2) Email Addresses
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  
        # 3) 주민등록번호
        "ssn": r"\d{6}-\d{7}" ,
        # 4) Links / URLs
        "url": r"https?://[^\s]+",  # http:// or https:// 로 시작하는 문자열

        # 5) IBANs (국제 계좌번호)
        "iban": r"[A-Z]{2}[0-9]{2}[ ]?[A-Z0-9]{4}(?:[ ]?[A-Z0-9]{4}){1,5}",

        # 6) BICs (은행 식별 코드)
        "bic": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b",

        # 7) Timestamps (YYYY-MM-DD HH:MM:SS or variants)
        "timestamp": r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b|\b\d{2}/\d{2}/\d{4} \d{2}:\d{2}\b",

        # 8) Dates (YYYY-MM-DD, DD.MM.YYYY, Month DD, YYYY 등)
        "date": r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}\.\d{2}\.\d{4}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b",

        # 9) QR Codes / Barcodes (숫자, 알파벳 혼합 8~20자리 정도)
        "qrcode_barcode": r"\b[A-Z0-9]{8,20}\b"
    }
    
    # 1. PDF 로드 - Sensitive Data 마킹 성공한 PDF
    # loader = Load()
    # docs = loader.load_directory_data(DATA_DIR)
    load_sensitive_data = Load_Sensitive_Data()
    documents = load_sensitive_data.redact_all_pdfs_in_directory(DATA_DIR, PATTERNS)
    
    # 2. 문서 분할
    spliter = Spliter(documents)
    # [*==선택가능==*] recursive split / Semantic Similarity(의미적 유사성 기준 텍스트 분할) 선택가능
    split_data = spliter.split_data_recursive()  # recursive split 사용
    # split_data = spliter.split_data_semantic()  #  Semantic Similarity 

    # 3. Qdrant 벡터스토어 생성
    uploader = QdrantUploader(data_dir=DATA_DIR)
    qdrant_vectorstore = uploader.upload_qdrant()
    
    # emb_vectorstore = Embedding_VectorStore(loader)
    
    # [*==선택가능==*]유료 임베딩/무료 임베딩 선택가능
    # (유료: OpenAIEmbeddings / 무료: HuggingFaceBgeEmbeddings, FastEmbedEmbeddings )
    # vectorstore = emb_vectorstore.create_vectorstore(split_data, use_paid=True)
    # vectorstore = emb_vectorstore.create_vectorstore(split_data, use_paid=True)

    # 4. Retriever 초기화
    retriever_reranker_qdrant = Retriever_Reranker_Qdrant(qdrant_vectorstore)


    # 5. LLM Manager 초기화
    llm_manager = LLMManager(OPENAI_API_KEY)

    # 6. RAG Pipeline 초기화
    rag_pipeline_qdrant = RAGPipelineInteractive_Qdrant(qdrant_vectorstore, llm_manager, retriever_reranker_qdrant)

    print("=======AI RAG Chatbot. 종료하려면 'exit' 입력=======")

    while True:
        user_input = input("\n[User]: ")
        if user_input.lower() in ["exit", "quit"]:
            print("대화를 종료합니다.")
            break
        
        # 7.Retriever적용선택 가능
        reranked_docs_qdrant = retriever_reranker_qdrant.get_qdrant_retriever(user_input)
       
        # [*==선택가능==*] 8. 질문 + RAG Pipeline 응답
        # => reranker 사용(True)/사용안함(False) 선택 가능
        # answer = rag_pipeline_qdrant.get_response(user_input, use_reranker=True)
        answer = rag_pipeline_qdrant.get_response(user_input, use_reranker=False)
        
        print(f"[AI]: {answer}")

if __name__ == "__main__":
    main_qrant()