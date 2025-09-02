import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing.loader import Load
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

from dotenv import load_dotenv
from langchain.schema import Document

# """.env 로드"""
# load_dotenv()

class Spliter:
    def __init__(self, docs):
        self.docs = docs
        
    def split_data_recursive(self):
        """ 재귀적 문자 텍스트 분할 : Recursive Character """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        
        # Split the documents
        texts = text_splitter.split_documents(self.docs)
        print(f"[총 청크 수]: {len(texts)}")
    
        for i, text in enumerate(texts):
            print(f"[Recursive] {i}번째 청크 : {text.page_content[:50]}...")
    
        return texts

    def split_data_semantic(self):
        """ 
            의미적 유사성 기준 텍스트 분할 : Semantic Similarity 
            높은 수준(high level)에서 문장으로 분할한 다음 3개 문장으로 그룹화한 다음 
            임베딩 공간에서 유사한 문장을 병합하는 방식
        """
        semantic_text_splitter = SemanticChunker(OpenAIEmbeddings(), add_start_index=True)
        all_chunks = []
        for doc in self.docs:
            chunks = semantic_text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                print(f"[Semantic Similarity] {i}번째 청크 : {chunk[:50]}...")
                
                """Document객체로 변환"""
                all_chunks.append(Document(
                    page_content = chunk,
                    metadata = {**doc.metadata, "chunk_idx": i}
                ))
                
        print(f"[총 청크 수]: {len(all_chunks)}")
        
        return all_chunks

