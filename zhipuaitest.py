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
    #导入文本分割器
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    # 知识库中单段文本长度
    CHUNK_SIZE = 500

    # 知识库中相邻文本重合长度
    OVERLAP_SIZE = 50
    # 使用递归字符文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )
    ['前言\n“周志华老师的《机器学习》（西瓜书）是机器学习领域的经典入门教材之一，周老师为了使尽可能多的读\n者通过西瓜书对机器学习有所了解,所以在书中对部分公式的推导细节没有详述，但是这对那些想深究公式推\n导细节的读者来说可能“不太友好”，本书旨在对西瓜书里比较难理解的公式加以解析，以及对部分公式补充\n具体的推导细节。”\n读到这里，大家可能会疑问为啥前面这段话加了引号，因为这只是我们最初的遐想，后来我们了解到，周\n老师之所以省去这些推导细节的真实原因是，他本尊认为“理工科数学基础扎实点的大二下学生应该对西瓜书\n中的推导细节无困难吧，要点在书里都有了，略去的细节应能脑补或做练习”。所以......本南瓜书只能算是我\n等数学渣渣在自学的时候记下来的笔记，希望能够帮助大家都成为一名合格的“理工科数学基础扎实点的大二\n下学生”。\n使用说明\n南瓜书的所有内容都是以西瓜书的内容为前置知识进行表述的，所以南瓜书的最佳使用方法是以西瓜书\n为主线，遇到自己推导不出来或者看不懂的公式时再来查阅南瓜书；对于初学机器学习的小白，西瓜书第1章和第2章的公式强烈不建议深究，简单过一下即可，等你学得',
    '有点飘的时候再回来啃都来得及；每个公式的解析和推导我们都力(zhi)争(neng)以本科数学基础的视角进行讲解，所以超纲的数学知识\n我们通常都会以附录和参考文献的形式给出，感兴趣的同学可以继续沿着我们给的资料进行深入学习；若南瓜书里没有你想要查阅的公式，或者你发现南瓜书哪个地方有错误，请毫不犹豫地去我们GitHub的\nIssues（地址：https://github.com/datawhalechina/pumpkin-book/issues）进行反馈，在对应版块\n提交你希望补充的公式编号或者勘误信息，我们通常会在24小时以内给您回复，超过24小时未回复的\n话可以微信联系我们（微信号：at-Sm1les）；\n配套视频教程：https://www.bilibili.com/video/BV1Mh411e7VU\n在线阅读地址：https://datawhalechina.github.io/pumpkin-book（仅供第1版）\n最新版PDF获取地址：https://github.com/datawhalechina/pumpkin-book/releases\n编委会',
    '编委会\n主编：Sm1les、archwalker']

    text_splitter.split_text(pdf_page.page_content[0:1000])
    split_docs = text_splitter.split_documents(pdf_pages)
    print(f"切分后的文件数量：{len(split_docs)}")
    print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")



    
    # text = '"你好，请介绍一下你自己"'
    # response = zhipu_embedding(text=text)
    # print(f'response类型为：{type(response)}')
    # print(f'embedding类型为：{response.object}')
    # print(f'生成embedding的model为：{response.model}')
    # print(f'生成的embedding长度为：{len(response.data[0].embedding)}')
    # print(f'embedding（前10）为: {response.data[0].embedding[:10]}')


    # output = get_completion("你好，讲一个简短的小故事")
    # print(output)
