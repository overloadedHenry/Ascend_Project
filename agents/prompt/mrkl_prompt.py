# flake8: noqa
# encoding: gbk
PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Thought: If you don't need to use tool,you should take "base_require_tool"
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action,which must be chinese

Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin!

Question: {input}
Thought:{agent_scratchpad}"""
