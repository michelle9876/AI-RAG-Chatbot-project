import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from processing.loader import Load
from processing.spliter import Spliter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


class Embedding_VectorStore: 
    """임베딩 & 벡터스토어 생성"""
    def __init__(self, loader: Load):
        self.loader = loader

    def create_vectorstore(self, split_data, use_paid=True):
        """유료 임베딩/무료 임베딩 """
        # 유료 임베딩
        if use_paid:
            embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            # 무료 임베딩 중 하나 선택
            embedding = HuggingFaceBgeEmbeddings()
            # embedding = FastEmbedEmbeddings()
            
        vectorstore = FAISS.from_documents(documents = split_data, embedding=embedding)
        # print(f"[vectorstore]: {vectorstore}")
        return vectorstore

    """data 폴더 안 모든 PDF 파일 개별 처리 후 vectorstores 생성 및 저장"""
    def save_vectorstore(self, data_dir, vs_dir, use_paid=True, use_semantic=False):
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(data_dir, file_name)
                print(f"[Processing]: {file_name}")
                # 1. PDF 파일 단위 로드
                docs = self.loader.load_single_data(file_path)
                print(f" => {file_name} 로드 완료 (페이지 수: {len(docs)})")
                
                # 2. 분할 방식 선택
                # 파일 단위 Spliter 객체 생성
                spliter = Spliter(docs)
                if use_semantic:
                    split_data = spliter.split_data_semantic()
                else:
                    split_data = spliter.split_data_recursive()
                print(f" => {file_name} 분할 완료 (청크 수 : {len(split_data)})")
                
                # 3. 벡터스토어 생성
                vectorstore = self.create_vectorstore(split_data, use_paid=use_paid)
                
                # 4. vectorstore 저장
                file_base = os.path.splitext(file_name)[0]
                save_path = os.path.join(vs_dir, file_base)
                vectorstore.save_local(save_path)
                print(f"{file_base} 벡터스토어 저장 완료 : {save_path}")

    # def delete_collection(): 도 추가

