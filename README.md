# 智驭数澜项目
本项目旨在通过Ascend芯片为基础搭建一个大语言模型（LLM）平台，扩展大模型的功能，如代码执行、知识库问答、联网搜索等。该平台基于FastAPI、LangChain等技术，并使用Streamlit提供Web端交互。
## 目录结构
```
llm_ascend
│
├── .env
├── requirements.txt
├── READMEv2.1.1_1.md
├── web_server
│   ├── .env
│   ├── file_operator.py
│   ├── web_launcher_api.py
├── chains
│   ├── RAG_chain.py
│   ├── __init__.py
├── agents
│   ├── react_agent.py
│   ├── rag_agent
│   ├── excel_agent
│   ├── data_analyse_agent
├── embedding
│   ├── embedding_server.py
│   ├── __init__.py
├── filestore
│   ├── excel
│   ├── pdf
├── file_operator.py
├── pic
│   ├── {pics}
├── document_loaders
│   ├── docloader.py
│   ├── __init__.py
└── startup.bat
```

## 功能
### 1.大模型功能扩展：
- 支持执行Python代码
- 用户可以上传CSV、PDF、WORD等文件并进行查询
- 使用Ascend芯片进行本地模型推理及Embedding
### 2.文件上传与存储：
- 支持上传多个文件（如Excel、PDF），并保存到本地存储
- 内置文件处理模块
### 3.绘图与数据分析：
- 使用Matplotlib库与python_repl执行python内代码生成图表
### 4.知识库问答：
- 集成RAG Chain进行知识库检索和问答
### 5.ReAct Agent 系统：
- 通过`思考-行动-输入-观察-循环`进行问题分解和分析，使用多步循环推理和AI自主选择工具调用

## 快速开始
### 1.安装依赖
```bash
cd ./llm_ascend
pip install -r requirements.txt
```

### 2.配置环境变量 
webserver文件夹下修改`.env`文件中的API相关配置：
```python
OPENAI_API_KEY = "<Your api key>"
OPENAI_API_BASE = "<Your api proxy url>"
FASTAPI_API_KEY = "ascend_password"   #该选项具体参考FastAPI组件的介绍
FASTAPI_API_URL = "http://localhost:8000/v1" #该选项具体参考FastAPI组件的介绍
```
### 3. 运行Web服务
在终端中输入以下命令：
```bash
python -m streamlit run ./web_launcher_api.py
```
windows环境下执行以下指令:

-（可选）利用`llm_ascend/startup.bat`在windows环境下启动Web服务器。

## 系统组件

RAG WebServer系统功能实现主要依赖于以下关键组件：

- **ReAct Agent**：通过 Prompt 指导模型逐步拆解复杂问题，结合工具实现多功能扩展。

- **Tools**：系统中的工具模块（如代码执行、联网搜索、知识库查询等）扩展了大语言模型的功能，提供了更加丰富的交互能力。

- **LangChain**：用于构建复杂链条和多功能交互的 LLM 框架。

- **RAG Chain**：通过知识库检索增强生成模型回答能力。

- **EmbeddingServer**：使用 Ascend 芯片本地生成向量，提供高效的 Embedding 服务。

- **ReRanker**：ReRanker 使用预训练模型对初步检索结果进行二次排序，从而提升了最终答案的精确度。

- **FastAPI**：提供与 LLM 交互的 API 接口。

  参考链接： [api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)
  
- **Streamlit**：Streamlit 用于构建轻量化的 Web 界面，并简化了前后端的交互。

## 主要模块说明
### 1. FastAPI
用于提供与大模型交互的标准API接口，兼容多种LLM，具体参考：[api-for-open-llm](https://github.com/xusenlinzy/api-for-open-llm)

#### Ascend 设备部署

在芯片算力允许的情况下,我们有以下两个方案:

-   直接利用`transformers`库以及`torch-npu`完成模型的加载
-   修改FastAPI项目的源代码的,适配`mindformers`或者`mindnlp`的接口
-   利用`torch->onnx->om`的转化，将模型转化成om模型
目前我们主要使用基于`torch-npu`的`transformers`完成模型的加载
使用的前提必须是该大模型必须支持`Function Calling`, 并且思考能力不小于`7b`参数量的平均水平,因为Agent的执行能力由大模型的推理效果决定.
### 2. Agent 
LLM 的原始生成能力在垂直领域有局限性,我们需要引入**Agent**和**Tools**来扩展大模型的功能,包括但不限于"知识库问答", "代码生成与执行", "联网搜索"等扩展性功能。只要大模型的思考与推理能力足够强大, 那么我们就能实现大量的以Prompt为驱动,以多形式结果为输出的`Agent-Tools`和`Agent`.


下面是本项目初期使用的**Agent**和**Tools**:

#### ReAct Agent

通过 ReAct（推理与行动）范式，模型在每次执行行动后会进行观察再继续推理。该 `Agent` 基于 `LangChain` 提供的 `PromptTemplate` 结合 `Python` 代码执行
- 优势：通过思维链逐步拆解问题，形成更清晰的推理步骤。ReAct还引入了观察（Observation）环节。在每次执行行动（Action）之后，模型会先观察当前状况（Observation。
- 实现：基于LangChain的PromptTemplate，结合Python代码执行。

PromptTemplate设计：
```python
    prompt_template = PromptTemplate.from_template(
        """
Assistant has access to the following tools:
{tools}
To use a tool, please use the following format:


Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action which are/is chinese

Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:


Thought: Do I need to use a tool? No

Final Answer: [your response here]

If you meet a syntax error, please check your input and try again.
If you are using Python REPL, you need also use print() to print the output.Your python input should follow the format:
"your code here",a great example is "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\nresult = fibonacci(10)\nprint(result)
If you think there you have got the answer, just stop and reply the final answer.
Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}
""")
```
参考论文地址: https://arxiv.org/abs/2210.03629

#### CSV Agent
CSV Agent 是一个智能代理，支持用户上传 CSV 文件进行代码生成、数据分析和结果反馈。它结合了 `Python REPL` 环境和 `Pandas` 数据处理工具，帮助用户便捷地分析 CSV 数据，并自动生成代码。

`create_csv_agent.py` 实现了 CSV Agent 的核心功能，能够将用户上传的 CSV 文件加载为 `Pandas DataFrame`，并通过 `Pandas` 工具进行数据查询和处理。

**功能说明**：

- **CSV 文件加载**:可以处理单个或多个 CSV 文件，将它们加载为 `Pandas DataFrame`。
- **自定义参数**：通过 `pandas_kwarg`s 传递自定义参数来加载 CSV 文件，比如指定分隔符、编码等。
- **Pandas 数据处理工具集**：一旦 CSV 文件被加载为 `DataFrame`，用户可以通过` Pandas` 工具进行查询、筛选、聚合等操作。

**实现方法**：

- **基于 Prompt 生成代码**：CSV Agent 根据用户输入的提示（Prompt）和上传的 CSV 文件，自动生成相应的 Python 代码。代码会在 `Python REPL` 环境中执行，执行结果会经过 Agent 的思考与整合后返回给用户。
- **思考-行动-观察循环**：CSV Agent 借鉴了 `ReAct Agent` 的思路，采用“**思考-行动-输入-观察-循环**”的执行方式。每一步代码执行后，Agent 会根据结果分析下一步的操作，逐步实现数据分析和问题解决。
csv_agent研习`agent-react`的思路，使用`思考-行动-输入-观察-循环`的方式来结合实现多个程序。

- **错误分析与自我修正**：如果代码执行过程中出现错误，CSV Agent 能够自动分析错误原因，并重新组织代码进行修正，直到生成正确的结果。`循环`中的错误分析解决了以往`chain`中因某个链出错而导致整个链条错误中断的问题。也显著增高了代码的正确性。在使用`python-repl`执行代码失败时agent会自行分析错误原因并且重新组织代码直至正确。

**增强稳定性和鲁棒性**：
`csv_agent`是项目初期开发的主要功能之一。该agent通过对原langchain csv_agent的修改，使用三个而非原来的一个tool，通过分解需求增强了该agent的稳定性和鲁棒性。

**CSV Agent 模块结构**：
- **prompt**：用于生成 CSV Agent 的提示词模板，帮助 Agent 理解用户需求并生成合适的代码。

- **create_agent**：创建 CSV Agent 实例，并优化错误处理逻辑，减少意外中断的发生。

- **agent**：Agent 的主体逻辑，负责处理上传的文件、执行代码和返回结果。

- **tools**：CSV Agent 所调用的工具模块，支持数据处理、代码执行等功能，增强了 Agent 的灵活性和可扩展性。
**增强稳定性和鲁棒性**：

### 3. EmbeddingServer
使用`bge-large-zh`中文`Embedding`模型，借助Ascend芯片本地生成向量。向量数据库采用`FAISS`，使用Ascend芯片进行存储和搜索。
```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-large-zh"
model = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

### 4. Chains

**RAG Chain**用于知识库问答，通过**RAG**（检索增强生成）链实现，将数据库转换为检索器，并结合LLM进行生成。
`RAG_chain.py `实现了一个 `RAG（Retrieval-Augmented Generation）`链，它使用预训练的语言模型结合知识库中的信息来回答用户问题。
#### 功能说明：
- **检索器**：使用 `self.database.as_retriever() `从知识库中检索与用户问题相关的信息。
- **自定义提示模板**：使用 `ChatPromptTemplate `定义了一个秘书角色的模板，通过上下文和用户问题生成对话。
- **RAG 链结构**：
    - 从数据库检索的上下文 (`context`)
    - 用户输入的问题 (`question`)
    - LLM 处理模板中的数据并生成回答
    - 使用 `StrOutputParser` 解析生成的文本输出
-**运行方式**：通过调用 `run() `方法，根据用户的输入生成最终的回答。

####  实现方法：
目前，项目仅使用 **RAG Chain** 实现知识库问答，下一步将会整合 **RAG Chain** 为 Agent Tools，提升系统灵活性.**
```python
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
```
### 5.Reranker
由于知识库检索器基于向量相似度，可能会存在一定的不准确性和不稳定性，因此引入了 **Reranker** 对相似度检索结果进行进一步打分，增强 **RAG** 的检索准确性。
本项目使用`BAAI/bge-reranker-base`模型对检索结果进行重新排序（Rerank），提升检索的精确性。新增代码如下：

```python
from FlagEmbedding import FlagReranker
def Rerank(quest,db,k=77):
    reranker=FlagReranker(path, use_fp16=True)#model_loading
    pre= db.vector.similarity_search(quest,k=k)
    scores=[]
    pre_=[]
    
    for i in pre:
        pre_.append([quest,i.page_content])
    print(pre_)
    score=reranker.compute_score(pre_)
    print(score)
    max_index, max_value = max(enumerate(score), key=lambda x: x[1]) 
    return pre[max_index]
```
该代码首先进行向量相似度检索，检索前` k `个最相关的向量，随后使用 **Reranker** 重新计算得分，确保检索的准确性更高。
### 6. 文件存储
系统支持上传多种文件格式，文件将保存到 `filestore` 目录下，供后续分析与处理。
```python
uploaded_files = st.file_uploader("上传你的文件", accept_multiple_files=True, key='file_uploader')
```
### 7. WebServer & Streamlit
本项目不再单独使用 HTML、CSS、JavaScript 等语言构建 Web 页面，而是通过 **Streamlit** 简化整个过程。Streamlit 提供了强大的容器（Container）和回调处理器（Callback Handler）功能，可以动态展示 Agent 的思考链，提升用户的交互体验。
为了打造轻量化且简洁美观的 Web UI 交互界面，系统采用 Streamlit 作为前后端框架。**Streamlit** 不仅能够解决前端渲染、状态维护和数据交互等问题，还通过其内置的 **Callback Handler** 组件实现函数回调，实时同步展示 Agent 的思考过程，让用户可以清晰地看到每一步的推理过程，带来更流畅、智能的使用体验。

### 8.TO_DO
#### agent构造知识查询
`!NOT FINISHED!`
使用 Agent 对用户问题进行重新构造，以便更精准地查询知识库。
#### agent react_chain
`!NOT FINISHED!`
正在构建 **Agent React Chain**，结合 Chain 和 React 的优势。虽然这种方式可能会带来更高的 Token 消耗，并对模型的能力提出更高要求，但可以实现更加灵活、智能的多功能处理。
#### 提示词沙盒
`!NOT FINISHED!`
为了防止在 **REPL** 执行危险代码，正在构建提示词沙盒，确保系统安全性。
#### 自动计量经济学agent
`!正在测试中`
利用 `react`的强大优势，可以制作自动计量经济学agent,相对于普通的agent问答，自动计量学agent通过更加复杂的agent构造和react的特点，使用户并不必完全掌握计量经济学的知识来进行数据疑问。如用户不必问“2020-2030年人口的线性回归图和其F检验对应的F/p值，只需要问”给我2020-2030年人口的趋势，这个有多少可靠性“便可让AI进行自动计量经济学处理。



