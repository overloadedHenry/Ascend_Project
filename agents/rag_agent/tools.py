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
        "һ�����ݲ�ѯ���ߣ������û��������ܽ��Ҫ��ѯ�����ݣ�����ʹ��������ѯ"
        "ֻ����һ���ַ�������query"
        "�����ܽ����������"
        "�ܽ�������ݱ��������ڽ���������ѯ"
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
        'һ�������ʴ𹤾ߣ����ݲ�ѯ������������'
        )



