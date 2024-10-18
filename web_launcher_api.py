import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema.output_parser import StrOutputParser
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import ChatOpenAI

from agents.agent import get_agent_answer
from agents.excel_agent.create_agent import create_csv_agent
from chains.RAG_chain import RAGChain
from embedding.embedding_server import EmbeddingServer


class FilePart:
    def __init__(self):
        pass

    @classmethod
    def shorten_filename(cls, filename, max_length=10):
        name, ext = os.path.splitext(filename)
        available_length = max_length - len(ext)
        if len(name) > available_length:
            return name[:available_length] + '***' + ext
        else:
            return filename

    @classmethod
    def classify_file(cls, file):
        if file.name.endswith('.txt'):
            return '文本文件'
        elif file.name.endswith('.csv'):
            return 'CSV文件'
        elif file.name.endswith('.jpg') or file.name.endswith('.png'):
            return '图像文件'
        elif file.name.endswith('.pdf'):
            return 'PDF文件'
        else:
            return '其他文件'


    @classmethod
    def loader_part(cls, uploaded_files):
        classified_files = {'文本文件': [], 'CSV文件': [], '图像文件': [], '其他文件': [], 'PDF文件': []}
        for uploaded_file in uploaded_files:
            # display_name = cls.shorten_filename(uploaded_file.name)
            file_type = cls.classify_file(uploaded_file)
            classified_files[file_type].append(uploaded_file)

    @classmethod
    def csv_reload(cls):
        # 使用Path对象来处理路径
        saved_file_path = Path(__file__).parent / 'filestore' / 'excel'
        #print(saved_file_path)

        # 使用Path对象的glob方法来查找所有csv文件
        excel_files_x86=list(saved_file_path.glob('*.xlsx'))
        for file_path in excel_files_x86:
            df=pd.read_excel(file_path)
            file_name,uselessness=os.path.splitext(file_path)
            df.to_csv(file_name+'.csv')
        excel_files = list(saved_file_path.glob('*.csv'))
        #print(excel_files)

        if 'csv' not in st.session_state:
            st.session_state.csv = []

        for file_path in excel_files:
            try:
                # 直接使用Path对象读取CSV文件
                #df = pd.read_csv(file_path, encoding='utf-8')
                st.session_state.csv.append(str(file_path))
                #print(st.session_state['csv'])
            except Exception as e:
                st.error(f"读取文件 {file_path.name} 时出错: {str(e)}")

        st.success("完成")

    @classmethod
    def csv_opt_page(cls):
        st.title('xlsx/csv操作器')
        uploaded_files_csv = st.file_uploader("上传你的文件", accept_multiple_files=True, key='file_uploader',
                                              type=['csv', 'xlsx'])
        if uploaded_files_csv:
            if 'csv_pre' not in st.session_state:
                st.session_state.csv_pre = []
            st.session_state.csv_pre.append(uploaded_files_csv)
        if st.button('全部重新读取'):
            cls.csv_reload()


class WebUI:
    def __init__(self, model_list=['FastAPI', 'gpt-4o-mini-2024-07-18']):
        self.model_list = model_list
        self.llm = None

    def stream_text(self, text):
        # 使用正则表达式分割文本
        # 这将分割单词（对于带空格的语言）和单个字符（对于不带空格的语言）
        words = re.findall(r'\w+|[^\w\s]', text)

        for word in words:
            time.sleep(0.02)
            # 对于英文单词，在后面添加空格
            if word.isascii() and word.isalnum():
                yield word + " "
            else:
                yield word

    def chat_mode_part(self):
        option = st.sidebar.selectbox("选择对话模式", ("普通对话", "知识库问答", "chat with Agent", "csv问答"), key='chat_mode')
        return option

    def main(self):
        st.sidebar.title("导航")
        # page = st.sidebar.radio("选择页面", ["聊天", "数据操作",  "csv读取器"])
        page = st.sidebar.radio("选择页面", ["聊天", "数据操作"])

        model_name = st.sidebar.selectbox("选择模型", self.model_list)
        if st.sidebar.button("清除记录(警告,模型对话记忆一并清除)", key='clean'):
            st.session_state['memory'].clear()
            st.session_state.messages = []

        if 'api_key' not in st.session_state:
            st.session_state['api_key'] = None
        if 'api_url' not in st.session_state:
            st.session_state['api_url'] = None

        if model_name == 'gpt-4o-mini-2024-07-18':
            st.session_state['api_key'] = os.getenv("OPENAI_API_KEY")
            st.session_state['api_url'] = os.getenv("OPENAI_API_BASE")
        elif model_name == 'FastAPI':
            st.session_state['api_key'] = os.getenv("FASTAPI_API_KEY")
            st.session_state['api_url'] = os.getenv("FASTAPI_API_URL")
        api_key = st.session_state['api_key']
        api_url = st.session_state['api_url']

        
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key, openai_api_base=api_url, streaming=True)
        #self.llm =  QianfanChatEndpoint(streaming=True,model="ERNIE-4.0-Turbo-8K")
        st.session_state['llm'] = self.llm
        if 'memory' not in st.session_state:
            st.session_state['memory'] = ConversationBufferMemory()

        if page == "聊天":
            self.chat_page()
        elif page == "数据操作":
            self.data_operation_page()
        elif page == 'csv读取器':
            FilePart.csv_opt_page()

    def chat_page(self):
        st.title("Chat with Ascend")
        self.chat_mode_part()
        st_callback = StreamlitCallbackHandler(st.container())
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state['full_response'] = ''

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What is up?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            chat_mode = st.session_state.get('chat_mode')
            response = None
            conversation_chain = ConversationChain(
                llm=st.session_state['llm'],
                verbose=True,
                memory=st.session_state['memory'],
                output_parser=StrOutputParser(),
            )
            if chat_mode == '普通对话':
                response = conversation_chain.invoke(prompt)
                #print(response['response'])
                stream_res = self.stream_text(response['response'])

                with st.chat_message("assistant"):
                    st.write_stream(stream_res)
                    st.session_state['full_response'] = response['response']

            elif chat_mode == '知识库问答':
                rag_chain = RAGChain(st.session_state['llm'], database=st.session_state.get('vectordb'))
                response = rag_chain.run(prompt)
                with st.chat_message("assistant"):
                    full_res = st.write_stream(response)
                    st.session_state['full_response'] = full_res

            elif chat_mode == 'chat with Agent':
                st_callback = StreamlitCallbackHandler(st.container())
                response = get_agent_answer(llm=st.session_state['llm'], prompt=prompt, container=st_callback)
                with st.chat_message("assistant"):
                    full_res = response['output'].rstrip(f'\n```')
                    #print(full_res)
                    st.write(full_res)
                    st.session_state['full_response'] = full_res

            elif chat_mode == 'csv问答':
                #st.warning('警告:不安全的代码，可能具有潜在危险性。这只是一个警告:<')
                if 'csv' not in st.session_state:
                    st.warning('你尚未加载csv，将自动为你重载')
                    FilePart.csv_reload()
                #print(st.session_state.csv)
                agent_executor = create_csv_agent(
                    st.session_state['llm'],
                    st.session_state.csv,
                    verbose=True,
                    agent_type='zero-shot-react-description',
                    #就用zero-shot-react-description
                    allow_dangerous_code=True,
                    handle_parsing_errors=True
                    
                )
                
 
                anti_error_code='''mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']'''
                history = []
                with st.chat_message("assistant"):
                    response=agent_executor.invoke({'input':prompt,'chat_history':None},  {'callbacks': [st_callback]})                  
                    full_res = response['output']
                    st.write(full_res)
                    st.session_state['full_response'] = full_res

            st.session_state.messages.append({
                "role": "assistant",
                "content": st.session_state['full_response']
            })

            st.session_state['full_response'] = ''


    def data_operation_page(self):
        st.title("向量化您的数据")

        uploaded_files = st.file_uploader("上传你的文件", accept_multiple_files=True, key='file_uploader')
        file_path = None

        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                if uploaded_file.name.endswith('csv') or uploaded_file.name.endswith('xlsx'):
                    file_path = Path(__file__).parent / 'filestore' / 'excel'

                elif uploaded_file.name.endswith('pdf') or uploaded_file.name.endswith('docx') or uploaded_file.name.endswith('txt'):
                    file_path = Path(__file__).parent / 'filestore' / 'pdf'
                try:
                    #print(file_path)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                except Exception as e:
                    print(e)

        if uploaded_files is not None:
            FilePart.loader_part(uploaded_files)

        vectordb = None

        # 设置点击按钮后的操作
        if st.button("向量化存入知识库", key='vectorize'):
            _vectordb = None
            saved_file_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), 'filestore', 'pdf'))
            #print(saved_file_path)
            pdf_files = [f for f in os.listdir(saved_file_path) if f.endswith('.pdf') or f.endswith('.docx') or f.endswith('.txt')]
            excel_files = [f for f in os.listdir(saved_file_path) if f.endswith('.xlsx')]
            embedding_server = EmbeddingServer()
            embedding_func = embedding_server.embed_documents
            for pdf_file in pdf_files:
                pdf_file_path = os.path.join(saved_file_path, pdf_file)
                if vectordb is not None:
                    db1 = embedding_func(pdf_file_path)
                    _vectordb.merge_from(db1)
                else:
                    _vectordb = embedding_func(pdf_file_path)
            for excel_file in excel_files:
                excel_file_path = os.path.join(saved_file_path, excel_file)
                if vectordb is not None:
                    db2 = embedding_func(excel_file_path)
                    _vectordb.merge_from(db2)
                else:
                    _vectordb = embedding_func(excel_file_path)
            vectordb = _vectordb
            st.session_state['vectordb'] = vectordb

            st.success("完成")

class FunctionImplement:
    def __init__(self):
        pass

if __name__ == '__main__':
    load_dotenv(find_dotenv(), override=True)
    webui = WebUI()
    webui.main()