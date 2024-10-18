import os
import sys

from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from embedding.embedding_server import Rerank
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
class RAGChain:
    def __init__(self, llm, database):
        self.llm = llm
        self.database = database
        self.prompt_template = ChatPromptTemplate.from_template("""
        你现在是一名秘书，请你根据用户提供的信息回答用户的问题。信息内容如下:\n
        <内容开始>{context}<内容结束>\n
        用户问题: {question}
        """)

    def run(self, prompt):
        # 将数据库转化为检索器
        retriever = self.database.as_retriever()
        # 定义 RAG 链
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )

        return rag_chain.stream(prompt)

# 以下为封装前的代码
# prompt_template = ChatPromptTemplate.from_template("""
# 你现在是一名秘书，请你根据用户提供的信息回答用户的问题。信息内容如下:\n
# <内容开始>{context}<内容结束>\n
# 用户问题: {question}
# """)
#
#
# # @tool('rag-tool that can be used if the user offer documents or ask something about the contexts in the doc', return_direct=True)
# def rag(llm, database, prompt):
#     """
#     :param llm: langchain[Basemodel] llm
#     :param db: database
#     :return: chain object
#     """
#     # 将数据库转化为检索器
#     retriever = database.as_retriever()
#     # 定义 RAG 链
#     rag_chain = (
#             {"context": retriever, "question": RunnablePassthrough()}
#             | prompt_template
#             | llm
#             | StrOutputParser()
#     )
#
#     return rag_chain.stream(prompt)
