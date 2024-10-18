#encoding:utf-8
"""���߶���ģ��"""
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


def _get_default_python_repl() -> PythonREPL:
    return PythonREPL(_globals=globals(), _locals=None)


def sanitize_input(query: str) -> str:
    """Sanitize input to the python REPL.

    Remove whitespace, backtick & python (if llm mistakes python console as terminal)

    Args:
        query: The query to sanitize

    Returns:
        str: The sanitized query
    """

    # Removes `, whitespace & python from start
    query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
    # Removes whitespace & ` from end
    query = re.sub(r"(\s|`)*$", "", query)
    query= re.sub(r"Observation","",query)
    #print('this',query)
    return query


class Requirment_tool(BaseTool):
    name: str="base_require_tool"
    description: str=(
        "һ������ѯ�ʹ��ߣ����û��������漰�κ���������ʱ��ֱ�ӵ��øù���ѯ���û�"
        "�ù��߲��������κ����ã����ڲ���Ҫ�����κ���������ʱ���ô˹���"
        "�ù���ֻ��Ҫ�ش��û��Ļ�������"
        "ʹ�øù��ߺ�����ֹͣ"
        )
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""

class PythonInputs(BaseModel):
    """Python inputs."""

    query: str = Field(description="code snippet to run")

class PythonAstREPLToolx(BaseTool):
    name: str = "Python_REPL_FOR_PLOTTING"  
    description: str = (  
        "һ��Python shell��ʹ����ͨ��ʹ��python_repl��ִ��Python�����ʹ��matplotlib���л�ͼ"  
        "����Ӧ������Ч��Python�����Ӧ����import matplotlib.pyplot as plt����һ�����Ӵ���'''plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']'''�Է�ֹ�ַ����롣�������Ƶ�ͼ�������ڵ�ǰĿ¼����Ϊ'pic'�ļ����¡�"  
        "�������鿴ĳ��ֵ���������Ӧ��ʹ��`print(...)`�����ӡ������"  
        "���벻Ҫʹ��'''�Է�ֹ����ע��"
    )
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True
    args_schema: Type[BaseModel] = PythonInputs

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            io_buffer = StringIO()
            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return io_buffer.getvalue()
                    else:
                        return ret
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return io_buffer.getvalue()
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""

        return await run_in_executor(None, self._run, query)

class PythonAstREPLTooly(BaseTool):
    name: str = "Python_REPL_FOR_EXCELING"  
    description: str = (  
        "һ��Python shell��ʹ������ִ��Python����Զ�ȡ�����е�����"  
        "ֻ����һ������'query'"
        "����ִ�л�ͼ�������"
        "�������鿴ĳ��ֵ���������Ӧ��ʹ��`print(...)`�����ӡ������"  
        "���벻Ҫʹ��'''������ע��"
    )
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True
    args_schema: Type[BaseModel] = PythonInputs


    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            io_buffer = StringIO()
            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return io_buffer.getvalue()
                    else:
                        return ret
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return io_buffer.getvalue()
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""

        return await run_in_executor(None, self._run, query)
#���´�����ڲ��ԣ�δ����
class PythonREPLTool(BaseTool):

    name: str = "Python_Linear Regression_sheel"
    description: str = (
        "һ��Python shell��ʹ������ִ��Python�����ʹ��sklearn��ʵ�����Իع� "
        "����Ӧ������Ч��Python���� "
        "�������鿴ĳ��ֵ���������Ӧ��ʹ��`print(...)`�����ӡ���� "
        "���Ӧ�õ���Python_REPL_FOR_EXCELING��������ɶ����Իع�Ļ�ͼ"
    )
    python_repl: PythonREPL = Field(default_factory=_get_default_python_repl)
    sanitize_input: bool = True

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool."""
        if self.sanitize_input:
            query = sanitize_input(query)
        return self.python_repl.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""
        if self.sanitize_input:
            query = sanitize_input(query)

        return await run_in_executor(None, self.run, query)


class PythonAstREPLTool(BaseTool):
    """Tool for running python code in a REPL."""

    name: str = "Python_Linear Regression_sheel"
    description: str = (
        "һ��Python shell��ʹ������ִ��Python�����ʹ��sklearn��ʵ�����Իع� "
        "����Ӧ������Ч��Python���� "
        "�������鿴ĳ��ֵ���������Ӧ��ʹ��`print(...)`�����ӡ����������Զ������ֻʹ��һ��print"
        #"���Ӧ�õ���Python_REPL_FOR_EXCELING��������ɶ����Իع�Ļ�ͼ"
    )
    globals: Optional[Dict] = Field(default_factory=dict)
    locals: Optional[Dict] = Field(default_factory=dict)
    sanitize_input: bool = True
    args_schema: Type[BaseModel] = PythonInputs

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            if self.sanitize_input:
                query = sanitize_input(query)
            tree = ast.parse(query)
            module = ast.Module(tree.body[:-1], type_ignores=[])
            exec(ast.unparse(module), self.globals, self.locals)  # type: ignore
            module_end = ast.Module(tree.body[-1:], type_ignores=[])
            module_end_str = ast.unparse(module_end)  # type: ignore
            io_buffer = StringIO()
            try:
                with redirect_stdout(io_buffer):
                    ret = eval(module_end_str, self.globals, self.locals)
                    if ret is None:
                        return io_buffer.getvalue()
                    else:
                        return ret
            except Exception:
                with redirect_stdout(io_buffer):
                    exec(module_end_str, self.globals, self.locals)
                return io_buffer.getvalue()
        except Exception as e:
            return "{}: {}".format(type(e).__name__, str(e))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Any:
        """Use the tool asynchronously."""

        return await run_in_executor(None, self._run, query)





