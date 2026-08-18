''' 
* RecursiveCharacterTextSplitter 递归字符文本分割
RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
    这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
RecursiveCharacterTextSplitter需要关注的是4个参数：

* separators - 分隔符字符串数组
* chunk_size - 每个文档的字符数量限制
* chunk_overlap - 两份文档重叠区域的长度
* length_function - 长度计算函数
'''
import re
from langchain_community.document_loaders import PyMuPDFLoader
#导入文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter

if __name__ == "__main__":
    loader = PyMuPDFLoader(r"D:\project\llm\test_agent\data_base\knowledge_db\pumkin_book\pumpkin_book.pdf")
    pdf_pages = loader.load()
    print(f"载入后的变量类型为：{type(pdf_pages)}，该 PDF 一共包含 {len(pdf_pages)} 页")

    pattern = re.compile(r"\n+")
    for doc in pdf_pages:
        doc.page_content = pattern.sub("\n", doc.page_content)
        doc.page_content = doc.page_content.replace('•', '')
        # 【重要】注释掉全部删除空格，保留空格，给分割器语义分隔标记
        # doc.page_content = doc.page_content.replace(' ', '')

    # 知识库中单段文本长度
    CHUNK_SIZE = 500
    # 知识库中相邻文本重合长度
    OVERLAP_SIZE = 50

    # 使用递归字符文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )

    # 测试片段，取第0页前1000字符；接收split_text返回结果
    test_text = pdf_pages[0].page_content[0:1000]
    test_split_result = text_splitter.split_text(test_text)
    print("\n====测试切分片段====")
    for idx, seg in enumerate(test_split_result):
        print(f"test_chunk_{idx}: {seg}\n")

    # 对全部文档做分割
    split_docs = text_splitter.split_documents(pdf_pages)

    # 简单校验
    if len(split_docs) == 0:
        raise RuntimeError("切分结果为空，请检查PDF加载与文本清洗逻辑！")

    print(f"\n切分后的文件数量：{len(split_docs)}")
    print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

    # 打印前两个chunk，肉眼校验切分效果
    print("\n====样例chunk输出====")
    print(split_docs[0].page_content)
    print("-"*60)
    print(split_docs[1].page_content)