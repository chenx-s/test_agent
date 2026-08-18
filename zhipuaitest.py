import os

from dotenv import load_dotenv, find_dotenv
from zhipuai import ZhipuAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
# 读取本地/项目的环境变量。

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

from zhipuai import ZhipuAI

client = ZhipuAI(
    api_key=os.environ["ZHIPUAI_API_KEY"]
)

def gen_glm_params(prompt):
    '''
    构造 GLM 模型请求参数 messages

    请求参数：
        prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages


def get_completion(prompt, model="glm-4-plus", temperature=0.95):
    '''
    获取 GLM 模型调用结果

    请求参数：
        prompt: 对应的提示词
        model: 调用的模型，默认为 glm-4，也可以按需选择 glm-3-turbo 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0.0-1.0。温度系数越低，输出内容越一致。
    '''

    messages = gen_glm_params(prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    if len(response.choices) > 0:
        return response.choices[0].message.content
    return "generate answer error"

def zhipu_embedding(text: str):

    api_key = os.environ['ZHIPUAI_API_KEY']
    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
        model="embedding-3",
        input=text,
    )
    return response


# ==========在这里调用函数==========
if __name__ == "__main__":

    # prompt = f"""
    # 给我一些研究LLM长度外推的论文，包括论文标题、主要内容和链接
    # """

    # response = get_completion(prompt)
    # print(response)
    # loader = PyMuPDFLoader("D:/project/llm/test_agent/data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")
    # pdf_pages = loader.load()
    # print(f"载入后的变量类型为：{type(pdf_pages)}，",  f"该 PDF 一共包含 {len(pdf_pages)} 页")


    # pdf_page = pdf_pages[1]
    # print(f"每一个元素的类型：{type(pdf_page)}.", 
    #     f"该文档的描述性数据：{pdf_page.metadata}", 
    #     f"查看该文档的内容:\n{pdf_page.page_content}", 
    #     sep="\n------\n")





    
    text = '"你好，请介绍一下你自己"'
    response = zhipu_embedding(text=text)
    print(f'response类型为：{type(response)}')
    print(f'embedding类型为：{response.object}')
    print(f'生成embedding的model为：{response.model}')
    print(f'生成的embedding长度为：{len(response.data[0].embedding)}')
    print(f'embedding（前10）为: {response.data[0].embedding[:10]}')


    # output = get_completion("你好，讲一个简短的小故事")
    # print(output)