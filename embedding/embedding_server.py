import os
import sys

p = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(p)
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_community.vectorstores import FAISS
import warnings
from document_loaders.docloader import Docloader
import faiss  # 导入 faiss 以进行索引操作
from FlagEmbedding import FlagReranker

class EmbeddingServer:
    def __init__(self, hf_endpoint='https://hf-mirror.com', model_name="BAAI/bge-small-zh-v1.5",
                 model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}):
        os.environ['HF_ENDPOINT'] = hf_endpoint
        self.model = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs,
                                              encode_kwargs=encode_kwargs)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    def embed_documents(self, file_path):
        docloader = Docloader(file_path)
        documents = docloader.load()
        split_documents = self.text_splitter.split_documents(documents)
        vectordb = FAISS.from_documents(split_documents, self.model)
        return vectordb
def Rerank(quest,db):
    reranker=FlagReranker('search', use_fp16=True)
    pre= db.similarity_search(quest,k=77)
    scores=[]
    pre_=[]
    
    for i in pre:
        pre_.append([quest,i.page_content])
    print(pre_)
    score=reranker.compute_score(pre_)
    print(score)
    max_index, max_value = max(enumerate(score), key=lambda x: x[1]) 
    return pre[max_index]

# 以下为封装前的代码
# warnings.filterwarnings("ignore")  # 忽略警告
# 设置环境变量
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# model_name = "BAAI/bge-small-zh-v1.5"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True}  # 设置为 True 以计算余弦相似度
#
# # 初始化嵌入模型
# model = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )
#
#
# def embedding_func(file):
#     # 初始化文本分割器
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
#
#     # 加载并分割文档
#     docloader = Docloader(file)  # Docloader 是一个自定义的文档加载类
#     documents = docloader.load()
#     docs = text_splitter.split_documents(documents)
#
#     db = FAISS.from_documents(docs, model)
#
#     return db


