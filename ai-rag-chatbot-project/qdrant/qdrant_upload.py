import os
from dotenv import load_dotenv
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from processing.loader import Load
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


class QdrantUploader:
    """Qdrant Cloud에 문서 업로드"""
    
    def __init__(self, data_dir: str, collection_name: str = "ai-rag-chatbot", batch_size: int = 16):
        self.data_dir = data_dir
        self.collection_name = collection_name
        self.batch_size = batch_size
        # np.bool 패치
        np.bool = np.bool_

        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
    def upload_qdrant(self):
        # 1. PDF문서 로드
        loader = Load()
        docs = loader.load_directory_data(self.data_dir)
        # 임베딩 생성
        embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)


        # 컬렉션 생성
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        print(f"컬렉션 생성 완료: {self.collection_name}")



        # Qdrant Cloud 업로드
        qdrant_vectorstore = Qdrant.from_documents(
            documents=docs,
            embedding=embedding,
            url=QDRANT_URL,
            prefer_grpc=True,
            api_key=QDRANT_API_KEY,
            collection_name=self.collection_name,
            batch_size=self.batch_size
        )


        print("Qdrant 업로드 완료!")
        return qdrant_vectorstore



