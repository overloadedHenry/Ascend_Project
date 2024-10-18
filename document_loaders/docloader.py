from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredExcelLoader
def Docloader(file:str):
    if file.endswith('.pdf') :
        try:
            loader = PyPDFLoader(file)
            return loader
        except Exception as e:
            print(e)
    elif file.endswith('.txt') or file.endswith('.docx'):
        try:
            loader=TextLoader(file)
            return loader
            
        except Exception as e:
            print(e)

    elif file.endswith('.xlsx'):
        try:
            loader = UnstructuredExcelLoader(file)
            return loader
        except Exception as e:
            print(e)
    else:
        raise NotImplementedError


