# import os
# from pathlib import Path
# from dotenv import load_dotenv
# from zhipuai import ZhipuAI

# # 加载.env文件
# env_path = Path(__file__).parent / '.env'
# load_dotenv(env_path)

# # 获取API密钥
# api_key = os.getenv("ZHIPUAI_API_KEY")

# # 如果还是获取不到，手动设置（临时测试用）
# if not api_key:
#     print("警告：无法从.env文件获取密钥，请检查文件内容")
#     # 取消下面这行的注释，并填入你的实际密钥进行测试
#     # api_key = "在这里填入你的实际API密钥"
#     exit()

# # 创建客户端
# client = ZhipuAI(api_key=api_key)

# # 测试调用
# try:
#     response = client.chat.completions.create(
#         model="glm-4",
#         messages=[
#             {"role": "user", "content": "你好，请介绍一下你自己"}
#         ]
#     )
#     print("调用成功！")
#     print("AI回复：", response.choices[0].message.content)
# except Exception as e:
#     print(f"调用失败：{e}")


# 使用 OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 使用百度千帆 Embedding
# from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使用

import os
from dotenv import load_dotenv, find_dotenv

# 【必须最先执行】加载.env
dotenv_path = find_dotenv()
ok = load_dotenv(dotenv_path)
print(f".env路径:{dotenv_path},加载成功:{ok}")
print("环境变量是否存在：", os.getenv("ZHIPUAI_API_KEY") is not None)

# 上面执行完，再导入你的自定义embedding
from zhipuai_embedding import ZhipuAIEmbeddings

embedding = ZhipuAIEmbeddings()
from zhipuai_embedding import ZhipuAIEmbeddings

# 定义 Embeddings
# embedding = OpenAIEmbeddings() 
embedding = ZhipuAIEmbeddings()
# embedding = QianfanEmbeddingsEndpoint()

# 定义持久化路径
persist_directory = '../../data_base/vector_db/chroma'

from langchain_community.vectorstores import Chroma

vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
print(f"向量库中存储的数量：{vectordb._collection.count()}")