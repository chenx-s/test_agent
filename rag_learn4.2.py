import sys
sys.path.append("D:\\project\\llm\\test_agent") # 将父目录放入系统路径中

# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from zhipuai_embedding import ZhipuAIEmbeddings

from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())    # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# 向量数据库持久化路径
persist_directory = 'D:\\project\\llm\\test_agent\\data_base\\vector_db\\chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)
# print(f"向量库中存储的数量：{vectordb._collection.count()}")


# question = "什么是prompt engineering?"
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
# docs = retriever.invoke(question)
# print(f"检索到的内容数：{len(docs)}")
# for i, doc in enumerate(docs):
#     print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")

#4.2.2
from langchain_core.runnables import RunnableLambda
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

combiner = RunnableLambda(combine_docs)
retrieval_chain = retriever | combiner
# print(retrieval_chain.invoke("南瓜书是什么？"))

import os
from dotenv import load_dotenv, find_dotenv
from langchain_zhipu import ChatZhipuAI

# 加载 .env 环境变量
env_loaded = load_dotenv(find_dotenv())
if not env_loaded:
    print("警告：未找到 .env 文件，将尝试使用全局环境变量")

# 读取 API Key 并做空值校验
zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
if not zhipuai_api_key:
    raise ValueError("ZHIPUAI_API_KEY 未配置，请检查 .env 文件或系统环境变量")

# 初始化模型，显式指定参数
llm = ChatZhipuAI(
    api_key=zhipuai_api_key,
    model="glm-4-plus",       # 可按需替换为 glm-3-turbo / glm-4-plus 等
    temperature=0.1
)

# 打印确认初始化结果
# print(llm)

print(llm.invoke("请你自我介绍一下自己！").content)

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。请你在回答的最后说“谢谢你的提问！”。
{context}
问题: {input}
"""
# 将template通过 PromptTemplate 转为可以在LCEL中使用的类型
prompt = PromptTemplate(template=template)

qa_chain = (
    RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
    | prompt
    | llm
    | StrOutputParser()
)


question_1 = "什么是南瓜书？"
question_2 = "Prompt Engineering for Developer是谁写的？"

result = qa_chain.invoke(question_1)
print("大模型+知识库后回答 question_1 的结果：")
print(result)

result = qa_chain.invoke(question_2)
print("大模型+知识库后回答 question_2 的结果：")
print(result)


print("llm answer question1",llm.invoke(question_1).content)

print("llm answer question2",llm.invoke(question_2).content)
