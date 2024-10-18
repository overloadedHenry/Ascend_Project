#encoding:gbk

import ast
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, Optional, Type

from langchain.pydantic_v1 import BaseModel, Field, root_validator
from langchain.tools.base import BaseTool
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables.config import run_in_executor
from langchain_experimental.utilities.python import PythonREPL

class data_query_tool(BaseTool):
    name: str = "query_data_tool"
    description: str = (
        "一个数据查询工具，根据用户的问题总结出要查询的内容，便于使用向量查询"
        "只接受一个字符串输入query"
        "必须总结成中文语言"
        "总结出的内容必须有利于进行向量查询"
    )
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool."""
        return query
class data_choose_tool(BaseTool):
    name: str="data_chose_tool"
    description:str=(
        '一个数据问答工具，根据查询到的向量数据'
        )



