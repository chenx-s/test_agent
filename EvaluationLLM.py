import sys
sys.path.append("D:\\project\\llm\\test_agent") # 将父目录放入系统路径中

# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from zhipuai_embedding import ZhipuAIEmbeddings

from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI

from langchain_zhipu import ChatZhipuAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())    # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
# OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# 定义 Embeddings
embedding = ZhipuAIEmbeddings()

# 向量数据库持久化路径
persist_directory = 'D:\\project\\llm\\test_agent\\data_base\\vector_db\\chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)
retriever = vectordb.as_retriever()
# 使用智谱 ChatZhipuAI 模型
llm = ChatZhipuAI(api_key=zhipuai_api_key, model_name = "glm-4.5-air", temperature = 0.1)

#5.1.2.1
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

template_v1 = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(template=template_v1)

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
retrievel_chain = retriever | RunnableLambda(combine_docs)
qa_chain = (
    RunnableParallel(context=retrievel_chain, question=RunnablePassthrough())
    |{
        "answer": QA_CHAIN_PROMPT | llm | StrOutputParser(),
        "context": lambda x: x["context"]
    }
)
print("问题一：")
question = "南瓜书和西瓜书有什么关系？"
result = qa_chain.invoke(question)
print(result["answer"])

print("问题二：")
question = "应该如何使用南瓜书？"
result = qa_chain.invoke(question)
print(result["answer"])

