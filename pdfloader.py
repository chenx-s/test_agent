from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
import re

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
# loader = PyMuPDFLoader("D:/project/llm/test_agent/data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")
# loader = UnstructuredMarkdownLoader("D:/project/llm/test_agent/data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md")
# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载

# if __name__ == "__main__":


#     loader = PyMuPDFLoader("D:/project/llm/test_agent/data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")
#     pdf_pages = loader.load()
#     print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")

#     pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)

#     for doc in pdf_pages:
#         doc.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), doc.page_content)
#     print(pdf_pages.page_content)

#     pdf_page = pdf_pages[1]
#     print(f"每一个元素的类型：{type(pdf_page)}.", 
#         f"该文档的描述性数据：{pdf_page.metadata}", 
#         f"查看该文档的内容:\n{pdf_page.page_content}", 
#         sep="\n------\n")
import re
from langchain_community.document_loaders import PyMuPDFLoader

if __name__ == "__main__":
    loader = PyMuPDFLoader(r"D:\project\llm\test_agent\data_base\knowledge_db\pumkin_book\pumpkin_book.pdf")
    pdf_pages = loader.load()
    print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")

    pattern = re.compile(r"\n+")
    for pdf_page in pdf_pages:
        # 第一步：清除换行
        pdf_page.page_content = pattern.sub("", pdf_page.page_content)
        # 新增：清除项目符号•、清除空格
        pdf_page.page_content = pdf_page.page_content.replace('•', '')
        pdf_page.page_content = pdf_page.page_content.replace(' ', '')
        pdf_page.page_content = pdf_page.page_content.replace('\n\n', '\n')


        print(pdf_page.page_content)

    # 查看处理后的第1页
    print("====处理完成，取第1页查看====")
    one_page = pdf_pages[1]
    print(f"元数据：{one_page.metadata}")
    print(f"内容：{one_page.page_content}")
