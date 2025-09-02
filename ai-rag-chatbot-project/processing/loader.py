from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os
import sys

class Load:
    def load_single_data(self, pdf_path: str) -> List[dict]:
        """PDF파일 로드. 파일 경로 입력 """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found : {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        """페이지별 문서 로드"""
        docs = loader.load()
        print(f"문서의수 : {len(docs)}")
        print(f"[페이지 내용]:\n {docs[10].page_content[:500]}")
        print(f"[메타데이터]:\n {docs[10].metadata}")
        return docs

    def load_directory_data(self, data_dir: str) -> List[dict]:
        # directory_path = "data"
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"'{data_dir}'는 디렉토리가 아닙니다.")
        
        all_docs = []
        
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                print(f"[Loading]: {pdf_path}...")
                
                try:
                    # Load the PDF
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    all_docs.extend(docs)
                    print(f"-> {file_name} 로드 완료 (페이지 수: {len(docs)})")
                except Exception as e:
                    print(f"->{file_name} 처리 실패: {e}")

        print(f"[All Docs 수]:{len(all_docs)}")
        
        return all_docs
        

