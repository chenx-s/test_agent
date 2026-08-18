# ''' 
# * RecursiveCharacterTextSplitter 递归字符文本分割
# RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
#     这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
# RecursiveCharacterTextSplitter需要关注的是4个参数：

# * separators - 分隔符字符串数组
# * chunk_size - 每个文档的字符数量限制
# * chunk_overlap - 两份文档重叠区域的长度
# * length_function - 长度计算函数
# '''
# #导入文本分割器
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# # 知识库中单段文本长度
# CHUNK_SIZE = 500

# # 知识库中相邻文本重合长度
# OVERLAP_SIZE = 50
# # 使用递归字符文本分割器
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=CHUNK_SIZE,
#     chunk_overlap=OVERLAP_SIZE
# )

# # 将字面量列表赋值给变量，构造Document对象模拟pdf_pages输入
# raw_text_list = [
#     '前言\n“周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读\n者通过西瓜书对机器学习有所了解,所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推\n导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充\n具体的推导细节。”\n读到这里，大家可能会疑问为啥前面这段话加了引号，因为这只是我们最初的遐想，后来我们了解到，周\n老师之所以省去这些推导细节的真实原因是，他本尊认为“理工科数学基础扎实点的大二下学生应该对西瓜书\n中的推导细节无困难吧，要点在书里都有了，略去的细节应能脑补或做练习”。所以......本南瓜书只能算是我\n等数学渣渣在自学的时候记下来的笔记，希望能够帮助大家都成为一名合格的“理工科数学基础扎实点的大二\n下学生”。\n使用说明\n南瓜书的所有内容都是以西瓜书的内容为前置知识进行表述的，所以南瓜书的最佳使用方法是以西瓜书\n为主线，遇到自己推导不出来或者看不懂的公式时再来查阅南瓜书；对于初学机器学习的小白，西瓜书第1章和第2章的公式强烈不建议深究，简单过一下即可，等你学得',
#     '有点飘的时候再回来啃都来得及；每个公式的解析和推导我们都力(zhi)争(neng)以本科数学基础的视角进行讲解，所以超纲的数学知识\n我们通常都会以附录和参考文献的形式给出，感兴趣的同学可以继续沿着我们给的资料进行深入学习；若南瓜书里没有你想要查阅的公式，或者你发现南瓜书哪个地方有错误，请毫不犹豫地去我们GitHub的\nIssues（地址：https://github.com/datawhalechina/pumpkin‑book/issues）进行反馈，在对应版块\n提交你希望补充的公式编号或者勘误信息，我们通常会在24小时以内给您回复，超过24小时未回复的\n话可以微信联系我们（微信号：at‑Sm1les）；\n配套视频教程：https://www.bilibili.com/video/BV1Mh411e7VU\n在线阅读地址：https://datawhalechina.github.io/pumpkin‑book（仅供第1版）\n最新版PDF获取地址：https://github.com/datawhalechina/pumpkin‑book/releases\n编委会',
#     '编委会\n主编：Sm1les、archwalker'
# ]

# # 构造Document对象，模拟pdf_pages
# pdf_pages = [Document(page_content=text) for text in raw_text_list]

# # 测试切分，接收返回值
# test_split_result = text_splitter.split_text(pdf_pages[0].page_content[0:1000])
# print("测试切分片段：", test_split_result)

# split_docs = text_splitter.split_documents(pdf_pages)

# # 增加空结果校验
# if not split_docs:
#     raise ValueError("切分结果为空，请检查输入文本与分割器参数")

# print(f"切分后的文件数量：{len(split_docs)}")
# print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

# # 打印样例chunk，方便观察切分效果
# print("\n-----样例chunk-----")
# print(split_docs[0].page_content)

import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
dotenv_path = find_dotenv()
load_ok = load_dotenv(dotenv_path)
print(f".env 文件路径：{dotenv_path}, 是否加载成功：{load_ok}")

# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = r"D:\project\llm\test_agent\data_base\knowledge_db"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print("前3个文件路径：", file_paths[:3])

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []
for file_path in file_paths:
    # 统一转小写后缀，兼容 .PDF / .Md
    file_type = file_path.split('.')[-1].lower()
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 读取本地文件并存储到text
texts = []
for loader in loaders:
    try:
        docs = loader.load()
        texts.extend(docs)
    except Exception as e:
        print(f"文件读取失败 {loader.file_path} , error: {e}")

# 边界校验，防止下标越界
if len(texts) >= 2:
    text = texts[1]
    # print(f"每一个元素的类型：{type(text)}.",
    #       f"该文档的描述性数据：{text.metadata}",
    #       f"查看该文档的内容:\n{text.page_content[0:]}",
    #       sep="\n------\n")
else:
    print(f"警告：读取得到文档总数：{len(texts)}，不足2个，无法取texts[1]")

print(f"\n总共加载Document对象数量：{len(texts)}")


from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
OVERLAP_SIZE = 50

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)

# 简单清洗：去除项目符号•
for doc in texts:
    doc.page_content = doc.page_content.replace("•", "")

split_docs = text_splitter.split_documents(texts)
# print(f"\n切分后chunk总数量：{len(split_docs)}")

if split_docs:
    print("\n====样例chunk输出====")
    print(split_docs[0].page_content[:400])


from zhipuai_embedding import ZhipuAIEmbeddings

# 定义 Embeddings
# embedding = OpenAIEmbeddings() 
embedding = ZhipuAIEmbeddings()
# embedding = QianfanEmbeddingsEndpoint()

# 定义持久化路径
persist_directory = 'D:\\project\\llm\\test_agent\\data_base\\vector_db\\chroma'

from langchain_community.vectorstores import Chroma

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# print(f"向量库中存储的数量：{vectordb._collection.count()}")


question="什么是大语言模型"


sim_docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(sim_docs)}")
for i, sim_doc in enumerate(sim_docs):
    print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")


mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")
