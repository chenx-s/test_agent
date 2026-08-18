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
    model="glm-4",       # 可按需替换为 glm-3-turbo / glm-4-plus 等
    temperature=0.1
)

# 打印确认初始化结果
print(llm)

#4.1
# output = llm.invoke("请你自我介绍一下自己！")
# print(f"模型输出结果：{output}")

# 这里我们要求模型对给定文本进行中文翻译
# prompt = """请你将由三个反引号分割的文本翻译成英文！\
# text: ```{text}```
# """
# text = "我带着比身体重的行李，\
# 游入尼罗河底，\
# 经过几道闪电 看到一堆光圈，\
# 不确定是不是这里。\
# "
# completion=prompt.format(text=text)
# print(f"completion:\n{completion}")

from langchain_core.prompts import ChatPromptTemplate

# prompt = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."

prompt = ChatPromptTemplate.from_template(
    "请将以下 {input_language} 内容翻译为 {output_language}：\n{text}"
)

# ✅ partial：提前固化部分模板变量，外部只需要传入 text
prompt_fixed = prompt.partial(input_language="英文", output_language="中文")
human_template = "{text}"

# chat_prompt = ChatPromptTemplate([
#     ("system", template),
#     ("human", human_template),
# ])

# text = "我带着比身体重的行李，\
# 游入尼罗河底，\
# 经过几道闪电 看到一堆光圈，\
# 不确定是不是这里。\
# "
# messages  = chat_prompt.invoke({"input_language": "中文", "output_language": "英文", "text": text})
# print(messages)
# output  = llm.invoke(messages)
# print(output)

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
# parsed_output = output_parser.invoke(output)
# # print(parsed_output)

# chain = chat_prompt | llm | output_parser
# chain = chain.invoke({"input_language":"中文", "output_language":"英文","text": text})
# print(f"chain.invoke()输出结果：{chain}")

# text = 'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.'
# chain2 = chain.invoke({"input_language": "英文", "output_language": "中文","text": text})
# print(f"chain2.invoke()输出结果：{chain2}")


from langchain_core.runnables import RunnablePassthrough,RunnableParallel

# 定义两个子链
translate_chain = prompt_fixed | llm | StrOutputParser()
summary_prompt = ChatPromptTemplate.from_template("总结以下内容：{text}")
summary_chain = summary_prompt | llm | StrOutputParser()

# 并行执行 + 原始数据透传
parallel_chain = RunnableParallel(
    {
    "翻译结果": translate_chain,    # 分支1：执行翻译
    "内容总结": summary_chain,      # 分支2：执行总结
    "原始文本": RunnablePassthrough() # 分支3：原样透传输入
    }
)

# 调用后返回包含三个key的字典
result = parallel_chain.invoke({"text": "Hello World"})

print("\n====并行链输出====")
for k,v in result.items():
    print(f"{k}: {v}")