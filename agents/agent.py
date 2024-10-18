import os
import sys

from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import PromptTemplate

# 添加当前文件路径到sys.path，以便能够导入本地模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# 定义一个Python REPL工具，用于执行Python代码
python_repl = PythonREPL()
coding_tool = Tool(
    name="CodingTool",
    func=python_repl.run,
    description="Useful when you need to use python codes to calculate, search for data or do other possible things that python can help with."
)

# 定义工具列表，并添加coding_tool到列表中
tools = [coding_tool]


# 定义一个函数，用于获取Agent的回答
def get_agent_answer(llm, prompt, container):
    # 获取所有工具的名字
    tool_names = [tool.name for tool in tools]

    # 定义Prompt模板
    prompt_template = PromptTemplate.from_template(
        """
Assistant has access to the following tools:
{tools}
To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action
```


When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No

Final Answer: [your response here]
```

If you think there you have got the answer, just stop and reply the final answer.
Begin!

Previous conversation history:

{chat_history}

New input: {input}

{agent_scratchpad}
""")

    # 创建React Agent，并传入LLM、工具和Prompt模板
    conversational_chat_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt_template)

    # 创建Agent执行器，并传入Agent、工具和verbose参数
    agent_executor = AgentExecutor(agent=conversational_chat_agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # 调用Agent执行器，获取回答，并传入输入和聊天历史记录
    response = agent_executor.invoke({'input': prompt, 'chat_history': None}, {'callbacks': [container]})
    # response = agent_executor.invoke({'input': prompt, 'chat_history': None})

    # 返回回答
    return response

def sp_mapper(llm,prompt,container):
    prompt_template = PromptTemplate.from_template(
        '''
        代理可以获得以下csv表格:
        {csv_line}
        Thought: 代理的目标是按照用户的需求绘制折线图
        Action : 代理需要使用[{csv_line}]来完成绘制
        Action : 代理需要获得表格中的行来制作一个列表来填充x轴
        Action : 代理需要获得表格中的列来制作一个列表来填充y轴
        Observation: the result of the action
        Final Answer: [[x轴列表],[y轴列表]]
        '''
    )
    return llm(prompt=prompt_template)
# 如果该文件被直接运行，则执行以下代码
if __name__ == '__main__':
    try:
        # 执行一段Python代码，并打印结果
        result = python_repl.run("""print("Hello, world!")
        """)
        print("Python code execution result:", result)
    except Exception as e:
        # 处理执行Python代码时可能出现的异常
        print("An error occurred while executing the Python code:", e)
