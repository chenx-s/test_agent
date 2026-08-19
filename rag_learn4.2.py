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
# def combine_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)
# 重新定义 combine_docs
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"]) # 将 docs 改为 docs["context"]


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

# qa_chain = (
#     RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
#     | prompt
#     | llm
#     | StrOutputParser()
# )


# question_1 = "什么是南瓜书？"
# question_2 = "Prompt Engineering for Developer是谁写的？"

# result = qa_chain.invoke(question_1)
# print("大模型+知识库后回答 question_1 的结果：")
# print(result)

# result = qa_chain.invoke(question_2)
# print("大模型+知识库后回答 question_2 的结果：")
# print(result)


# print("llm answer question1",llm.invoke(question_1).content)

# print("llm answer question2",llm.invoke(question_2).content)
#4.2.5
from langchain_core.prompts import ChatPromptTemplate

# 问答链的系统prompt
system_prompt = (
    "你是一个问答任务的助手。 "
    "请使用检索到的上下文片段回答这个问题。 "
    "如果你不知道答案就说不知道。 "
    "请使用简洁的话语回答用户。"
    "\n\n"
    "{context}"
)
# 制定prompt template
qa_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)





# # 无历史记录
# messages = qa_prompt.invoke(
#     {
#         "input": "南瓜书是什么？",
#         "chat_history": [],
#         "context": ""
#     }
# )
# for message in messages.messages:
#     print(message.content)


# # 有历史记录
# messages = qa_prompt.invoke(
#     {
#         "input": "你可以介绍一下他吗？",
#         "chat_history": [
#             ("human", "西瓜书是什么？"),
#             ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
#         ],
#         "context": ""
#     }
# )
# for message in messages.messages:
#     print(message.content)

from langchain_core.runnables import RunnableBranch

# 压缩问题的系统 prompt
condense_question_system_template = (
    "请根据聊天记录完善用户最新的问题，"
    "如果用户最新的问题不需要完善则返回用户的问题。"
    )
# 构造 压缩问题的 prompt template
condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
# 构造检索文档的链
# RunnableBranch 会根据条件选择要运行的分支
retrieve_docs = RunnableBranch(
    # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
    (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
    # 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
    condense_question_prompt | llm | StrOutputParser() | retriever,
)



# 定义问答链
qa_chain = (
    RunnablePassthrough.assign(context=combine_docs) # 使用 combine_docs 函数整合 qa_prompt 中的 context
    | qa_prompt # 问答模板
    | llm
    | StrOutputParser() # 规定输出的格式为 str
)

# 定义带有历史记录的问答链
qa_history_chain = RunnablePassthrough.assign(
    context = (lambda x: x) | retrieve_docs # 将查询结果存为 content
    ).assign(answer=qa_chain) # 将最终结果存为 answer

# 不带聊天记录
# result1 =qa_history_chain.invoke({
#     "input": "西瓜书是什么？",
#     "chat_history": []
# })

# print("不带聊天记录的问答链结果：",result)
# 带聊天记录
result2 = qa_history_chain.invoke({
    "input": "南瓜书跟它有什么关系？",
    "chat_history": [
        ("human", "西瓜书是什么？"),
        ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
    ]
})
print("带聊天记录的问答链结果：",result2)