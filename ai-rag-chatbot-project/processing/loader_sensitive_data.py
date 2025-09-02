import os
import sys
import re
from typing import List, Pattern, Dict
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document
import phonenumbers
import argparse
from tqdm import tqdm
import fitz
from PIL import Image
import cv2
import numpy as np
#from pyzbar.pyzbar import decode  # QR코드/바코드 인식 라이브러리

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# pdf_path = os.path.join(BASE_DIR, "data", "Test-Document.pdf")
# DATA_DIR = os.path.join(BASE_DIR, "data")

class Load_Sensitive_Data:
    
    # 1. PDF에서 텍스트 추출
    def extract_texts_from_pdf(self, pdf_path: str) -> List[str]:
        """PDF파일 로드. 파일 경로 입력 """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found : {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        """페이지별 문서 로드"""
        docs = loader.load()
        print(f"문서의수 : {len(docs)}")
        return [doc.page_content for doc in docs]

    # texts = extract_texts_from_pdf(pdf_path)
    
    
    # 2. 패턴 기반 찾기 (페이지 단위)
    def find_patterns(self, texts: List[str], patterns: dict) -> List[List[str]]:
        """페이지별로 모든 패턴을 탐색해서 리턴"""
        all_matches = []
        for text in texts:  # 페이지 단위
            page_matches = []
            for key, pattern in patterns.items():
                # 뒤에 공백/콤마/마침표 허용 ([\s,\.]?)
                compiled = re.compile(rf"({pattern})([\s,\.]?)", re.VERBOSE)
                found = compiled.findall(text)
                page_matches.extend([f"{m[0]}{m[1]}" for m in found])
            all_matches.append(page_matches)
        return all_matches

    # matches = find_patterns(texts, PATTERNS)
    # print(matches)


    # 3. PDF 마스킹
    def redact_pdf(self, pdf_path: str, matches: List[List[str]], output_path: str, replace_text="REDACTED", fill=(0, 0, 0)):
        """fitz로 PDF 열어서 특정 패턴 마스킹"""
        pdf_doc = fitz.open(pdf_path)
        
        for page_num, items in enumerate(matches):
            page = pdf_doc[page_num]
            items = list(dict.fromkeys(items))  # 중복 제거
            # 마스킹할 영역 찾기
            for item in items:
                clean_item = item.strip()
                if not clean_item:
                    continue
                # 공백, 콤마, . 기준으로 부분 문자열 나누기
                parts = re.split(r'[\s,.]+', clean_item)
                for part in parts:
                    if not part:
                        continue
                    text_instances = page.search_for(part)
                    for inst in text_instances:
                        page.add_redact_annot(inst, text=replace_text, fill=fill)
            page.apply_redactions()
        
        pdf_doc.save(output_path)
        print(f"[완료]저장 위치: {output_path}")
        
        return output_path

    def redact_all_pdfs_in_directory(self, data_dir: str, patterns: dict, replace_text="REDACTED", fill=(0, 0, 0) ) -> List[dict]:
        # directory_path = "data"
        if not os.path.isdir(data_dir):
            raise NotADirectoryError(f"'{data_dir}'는 디렉토리가 아닙니다.")

        redacted_files = []  
        
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith(".pdf"):
                pdf_path = os.path.join(data_dir, file_name)
                print(f"[Loading]: {pdf_path}...")
                output_path = os.path.join(data_dir, f"{os.path.splitext(file_name)[0]}_redacted.pdf")
            
                print(f"\n=== 처리 중: {file_name} ===")
            
                try:
                    # 1. 텍스트 추출
                    texts = self.extract_texts_from_pdf(pdf_path)
                    # 2. 패턴 찾기
                    matches = self.find_patterns(texts, patterns)
                    # 3. PDF 마스킹
                    self.redact_pdf(pdf_path, matches, output_path, replace_text=replace_text, fill=fill)
                    redacted_files.append(output_path)
                    
                except Exception as e:
                    print(f"->{file_name} 처리 실패: {e}")
                    
        documents = []
        for pdf_path in redacted_files:
            texts = self.extract_texts_from_pdf(pdf_path)
            for page_text in texts:
                documents.append(Document(page_content=page_text))
   
        return documents   

   

