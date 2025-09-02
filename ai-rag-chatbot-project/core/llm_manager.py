import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import Annotated, List, TypedDict

# load_dotenv()

class LLMManager:
    """LLM configuration & promps """
    def __init__(self, api_key, model_name="gpt-3.5-turbo"):
        # self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model=model_name, api_key=api_key)

    # 프롬프트 생성
    # prompt = hub.pull("rlm/rag-prompt")

    def get_query_prompt(self) -> PromptTemplate:
        return PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate 2
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}"""
        )
        

    def get_rag_prompt(self) -> ChatPromptTemplate:
        """Get RAG prompt template."""
        template = """Answer the question based ONLY on the following context and explain it in details :
        {context}
        Question: {question}
        """
        return ChatPromptTemplate.from_template(template)
    


