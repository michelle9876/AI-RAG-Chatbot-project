# AI-RAG-Chatbot-project
작성자 : 김보경
## 프로젝트 개요
이 프로젝트는 **RAG (Retrieval-Augmented Generation)** 시스템을 구현한 AI 챗봇입니다. PDF 문서를 로드하고, 텍스트를 분할하여 벡터 데이터베이스를 구축한 후, 사용자의 질문에 대해 관련 문서를 검색하여 AI가 답변을 생성하는 시스템입니다.
## 주요 기능

### 1. **문서 처리 (Document Processing)**

- PDF 파일 로드 및 파싱

- 단일 파일 및 디렉토리 전체 처리

- 한국어/영어 문서 지원

- **민감 데이터 자동 마스킹**: 개인정보, 연락처, 계좌번호 등 자동 탐지 및 마스킹

### 2. **텍스트 분할 (Text Splitting)**

- **Recursive Character Splitter**: 문자 단위 재귀적 분할

- **Semantic Similarity Splitter**: 의미적 유사성 기반 분할, OpenAI 임베딩 기반 고급 의미적 분할

### 3. **벡터 임베딩 (Vector Embedding)**

- **유료 임베딩**: OpenAI Embeddings (text-embedding-3-small)

- **무료 임베딩**: HuggingFace BGE Embeddings, FastEmbed Embeddings

- **FAISS 벡터 데이터베이스**: 로컬 벡터 저장소

- **Qdrant 벡터 데이터베이스**: 클라우드 기반 벡터 저장소 (새로운 기능)


### 4. **검색 및 재순위화 (Retrieval & Reranking)**

- **Vectorstore Retriever**: 벡터 유사도 기반 검색

- **FAISS Retriever**: FAISS 기반 검색

- **BM25 Retriever**: 키워드 기반 검색

- **Ensemble Retriever**: BM25 + FAISS 조합 검색

- **Qdrant Retriever**: Qdrant 기반 검색 (새로운 기능)

- **Cross Encoder Reranker**: 문서 순위 재조정

### 5. **AI 응답 생성 (AI Response Generation)**

- OpenAI GPT 모델을 사용한 답변 생성

- RAG 파이프라인을 통한 컨텍스트 기반 응답

### 6. **보안 및 개인정보 보호 (Security & Privacy)**

- **민감 데이터 자동 탐지**: 전화번호, 이메일, 주민등록번호, URL, IBAN, BIC, 타임스탬프, 날짜

- **PDF 마스킹**: 민감 정보를 "REDACTED"로 자동 교체

- **배치 처리**: 디렉토리 내 모든 PDF 파일 일괄 처리

## 프로젝트 구조
```
rag-sample-code4/

├── 📄 run_chat.py # 기존 RAG 챗봇 실행 파일 (FAISS 기반)

├── 📄 run_chat_qdrant.py # 새로운 RAG 챗봇 실행 파일 (Qdrant 기반)

├── 📄 requirements.txt # 필요한 패키지 목록

├── 📄 docker-compose.yaml # Qdrant Docker 설정

├── 📄 README.md # 프로젝트 설명서

├── 📄 .python-version # Python 버전 설정

│

├── 📁 data/ # PDF 문서 저장소

│ ├── sample-pdf-English-test-with-answers.pdf

│ └── unstructured_pdf_sample.pdf

│ └── TEST-DOCUMENT.pdf

│

├── 📁 processing/ # 문서 처리 모듈

│ ├── loader.py # 기본 PDF 로더 (단일/디렉토리)

│ ├── loader_sensitive_data.py # 민감 데이터 처리 로더 (새로운 기능)

│ ├── spliter.py # 텍스트 분할기

│ └── embedding.py # 벡터 임베딩 및 저장

│

├── 📁 core/ # 핵심 RAG 모듈

│ ├── retriever_reranker.py # 기존 검색 및 재순위화 (FAISS 기반)

│ ├── retriever_reranker_qdrant.py # 새로운 검색 및 재순위화 (Qdrant 기반)

│ ├── rag_pipeline.py # 기존 RAG 파이프라인 (FAISS 기반)

│ ├── rag_pipeline_qdrant.py # 새로운 RAG 파이프라인 (Qdrant 기반)

│ └── llm_manager.py # LLM 관리자

│

├── 📁 qdrant/ # Qdrant 관련 모듈 (새로운 기능)

│ └── qdrant_upload.py # Qdrant 클라우드 업로드

│

├── 📁 vectorstores/ # FAISS 벡터 데이터베이스 저장소

└── 📁 storage/ # 기타 저장소

```

## 파일별 상세 기능

### **run_chat.py** (기존 FAISS 기반)

- **RAG 챗봇 메인 실행 파일 (FAISS 기반)**

- 전체 RAG 파이프라인 통합 실행

- 대화형 인터페이스 제공

- 다양한 검색 방법 선택 가능

### **run_chat_qdrant.py** (새로운 Qdrant 기반)

- **RAG 챗봇 메인 실행 파일 (Qdrant 기반)**

- 민감 데이터 자동 마스킹 기능 포함

- Qdrant 클라우드 기반 벡터 검색

- 향상된 보안 및 개인정보 보호


### **processing/loader.py**

```python
# 주요 함수

load_single_data(file_path) # 단일 PDF 파일 로드

load_directory_data(directory) # 디렉토리 내 모든 PDF 로드

```

- PDF 파일 파싱 및 텍스트 추출

- 한국어/영어 문서 자동 감지

- 메타데이터 포함 문서 객체 생성

### **processing/loader_sensitive_data.py** (새로운 기능)

```python
# 주요 함수
extract_texts_from_pdf(pdf_path) # PDF에서 텍스트 추출

find_patterns(texts, patterns) # 민감 데이터 패턴 탐지

redact_pdf(pdf_path, matches, output_path) # PDF 마스킹

redact_all_pdfs_in_directory(data_dir, patterns) # 디렉토리 전체 처리

```

- **민감 데이터 자동 탐지**: 전화번호, 이메일, 주민등록번호, URL, IBAN, BIC, 타임스탬프, 날짜

- **PDF 마스킹**: 민감 정보를 "REDACTED"로 자동 교체

- **배치 처리**: 디렉토리 내 모든 PDF 파일 일괄 처리


### **processing/spliter.py**

```python
# 주요 함수
split_data_recursive(docs) # 재귀적 문자 분할

split_data_semantic(docs) # 의미적 유사성 분할 (OpenAI 임베딩 기반)

```

- 다양한 텍스트 분할 방법 제공

- 청크 크기 및 오버랩 설정 가능

- 의미적 경계를 고려한 분할

### **processing/embedding.py**

```python
# 주요 함수

create_vectorstore(split_data, use_paid=True) # 벡터스토어 생성

save_vectorstore(data_dir, vs_dir, use_paid=True) # 벡터스토어 저장

```

- OpenAI 및 HuggingFace 임베딩 모델 지원

- FAISS 벡터 데이터베이스 구축

- 벡터스토어 로컬 저장 및 로드


### **qdrant/qdrant_upload.py** (새로운 기능)

```python
# 주요 함수

QdrantUploader(data_dir, collection_name) # Qdrant 업로더 초기화

upload_qdrant() # Qdrant 클라우드 업로드

```

- **Qdrant 클라우드 연동**: 클라우드 기반 벡터 데이터베이스

- **배치 업로드**: 대용량 문서 처리 지원

- **컬렉션 관리**: 동적 컬렉션 생성 및 관리

  
### **core/retriever_reranker.py** (기존 FAISS 기반)

```python
# 주요 함수

use_vectorstore_retriever(vectorstore, query) # 벡터 검색

get_faiss_retriever(k=5) # FAISS 검색

use_ensemble_retriever(vectorstore, query) # 앙상블 검색

cross_encoder_reranker(vectorstore, query) # 재순위화

```

- 다양한 검색 알고리즘 구현

- 벡터 검색, FAISS 검색, BM25 + FAISS 앙상블 검색

- Cross Encoder를 통한 문서 재순위화

### **core/retriever_reranker_qdrant.py** (새로운 Qdrant 기반)

```python
# 주요 함수

use_qdrant_vectorstore_retriever(query) # Qdrant 벡터 검색

get_qdrant_retriever(k=5) # Qdrant 검색

cross_encoder_reranker_qdrant(query) # Qdrant 재순위화

```

- **Qdrant 기반 검색**: 클라우드 벡터 데이터베이스 검색

- **향상된 성능**: 대용량 데이터 처리 최적화

- **실시간 검색**: 클라우드 기반 실시간 검색

### **core/rag_pipeline.py** (기존 FAISS 기반)

```python
# 주요 기능

RAGPipelineInteractive(vectorstore, llm_manager, retriever)

get_response(query, use_reranker=False) # RAG 응답 생성

```

- 전체 RAG 파이프라인 통합

- 검색 결과와 AI 응답 결합

- 대화형 인터페이스 제공

### **core/rag_pipeline_qdrant.py** (새로운 Qdrant 기반)

```python
# 주요 기능

RAGPipelineInteractive_Qdrant(qdrant_vectorstore, llm_manager, retriever_reranker_qdrant)

get_response(query, use_reranker=False) # Qdrant RAG 응답 생성

```

- **Qdrant 기반 RAG 파이프라인**: 클라우드 벡터 데이터베이스 활용

- **향상된 확장성**: 대용량 데이터 처리 지원

- **실시간 응답**: 클라우드 기반 실시간 응답

### **core/llm_manager.py**

```python
# 주요 기능

LLMManager(api_key) # LLM 관리자 초기화

get_response(prompt) # AI 응답 생성

```

- OpenAI GPT 모델 관리

- API 키 설정 및 오류 처리

- 응답 생성 및 스트리밍

## 설치 및 설정

### 1. **필요한 패키지 설치**

```bash

pip install -r requirements.txt

```


### 2. **환경변수 설정**

```bash

# .env 파일 생성

echo "OPENAI_API_KEY=api-key-here" > .env

echo "QDRANT_URL=qdrant-url-here" >> .env

echo "QDRANT_API_KEY=qdrant-api-key-here" >> .env

```

### 3. **Docker 설정 (Qdrant 로컬 실행)**

```bash
# Qdrant 로컬 실행

docker-compose up -d

# 또는 Qdrant 클라우드 사용

https://cloud.qdrant.io/ 에서 계정 생성 후 API 키 발급

```

### 4. **PDF 파일 준비**

- `data/` 폴더에 PDF 파일들을 저장

## 사용 방법

### **기존 FAISS 기반 RAG 챗봇 실행**

```bash

python run_chat.py

```

### **새로운 Qdrant 기반 RAG 챗봇 실행 **

```bash

python run_chat_qdrant.py

```

## 설정 옵션
### **텍스트 분할 방법 선택**

```python

# run_chat.py 또는 run_chat_qdrant.py에서 선택

split_data = spliter.split_data_recursive() # 재귀적 분할

split_data = spliter.split_data_semantic() # 의미적 분할

```

### **임베딩 모델 선택**

```python
# 유료 임베딩 (OpenAI)
vectorstore = emb_vectorstore.create_vectorstore(split_data, use_paid=True)


# 무료 임베딩 (HuggingFace)
vectorstore = emb_vectorstore.create_vectorstore(split_data, use_paid=False)

```

### **검색 방법 선택**

#### **FAISS 기반 (run_chat.py)**

```python
reranked_docs = retriever_reranker.get_faiss_retriever(user_input) # FAISS 검색

reranked_docs = retriever_reranker.use_vectorstore_retriever(user_input) # 벡터스토어 검색

reranked_docs = retriever_reranker.use_ensemble_retriever(user_input) # 앙상블 검색

reranked_docs = retriever_reranker.cross_encoder_reranker(user_input) # 재순위화

```

#### **Qdrant 기반 (run_chat_qdrant.py)**

```python
reranked_docs_qdrant = retriever_reranker_qdrant.get_qdrant_retriever(user_input) # Qdrant 검색

reranked_docs_qdrant = retriever_reranker_qdrant.cross_encoder_reranker_qdrant(user_input) # Qdrant 재순위화

```

  
### **민감 데이터 처리 설정**

```python
# 민감 데이터 패턴 정의

PATTERNS = {

"phone": r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)\d{3,4}[-.\s]?\d{4}",

"email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",

"ssn": r"\d{6}-\d{7}",

"url": r"https?://[^\s]+",

"iban": r"[A-Z]{2}[0-9]{2}[ ]?[A-Z0-9]{4}(?:[ ]?[A-Z0-9]{4}){1,5}",

"bic": r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b",

"timestamp": r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}\b",

"date": r"\b\d{4}-\d{2}-\d{2}\b",

"qrcode_barcode": r"\b[A-Z0-9]{8,20}\b"

}

# 민감 데이터 처리 적용

load_sensitive_data = Load_Sensitive_Data()

documents = load_sensitive_data.redact_all_pdfs_in_directory(DATA_DIR, PATTERNS)

```

## 성능 특징
### **장점**

- **모듈화된 구조**: 각 기능이 독립적으로 구현되어 유지보수 용이

- **다양한 옵션**: 여러 분할 방법, 임베딩 모델, 검색 알고리즘 제공

- **확장성**: 새로운 문서 형식이나 모델 추가 가능

- **비용 효율성**: 무료 임베딩 모델 사용 가능

- **보안 강화**: 민감 데이터 자동 탐지 및 마스킹

- **클라우드 지원**: Qdrant 클라우드 기반 대용량 데이터 처리

- **Docker 지원**: 컨테이너화된 배포 환경

### **새로운 기능**

- **Qdrant 벡터 데이터베이스**: 클라우드 기반 벡터 저장소

- **민감 데이터 처리**: 개인정보 자동 탐지 및 마스킹

- **향상된 보안**: PDF 레벨에서 민감 정보 보호

- **배치 처리**: 대량 문서 일괄 처리


### **제한사항**

- **API 의존성**: OpenAI API 사용 시 비용 발생

- **처리 시간**: 대용량 문서 처리 시 시간 소요

- **클라우드 의존성**: Qdrant 클라우드 사용 시 인터넷 연결 필요

## 기술 스택

### **핵심 라이브러리**

- **LangChain**: RAG 파이프라인 구축

- **OpenAI**: GPT 모델 및 임베딩

- **Qdrant**: 벡터 데이터베이스

- **FAISS**: 로컬 벡터 검색

- **PyMuPDF**: PDF 처리

- **Transformers**: HuggingFace 모델


### **보안 및 개인정보 보호**

- **정규표현식**: 민감 데이터 패턴 매칭

- **PyMuPDF**: PDF 마스킹

- **Phonenumbers**: 전화번호 검증

### **인프라**

- **Docker**: 컨테이너화

- **Qdrant Cloud**: 클라우드 벡터 데이터베이스

- **Python-dotenv**: 환경변수 관리
