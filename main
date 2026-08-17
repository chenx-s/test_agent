import os
from pathlib import Path
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 加载.env文件
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# 获取API密钥
api_key = os.getenv("ZHIPUAI_API_KEY")

# 如果还是获取不到，手动设置（临时测试用）
if not api_key:
    print("警告：无法从.env文件获取密钥，请检查文件内容")
    # 取消下面这行的注释，并填入你的实际密钥进行测试
    # api_key = "在这里填入你的实际API密钥"
    exit()

# 创建客户端
client = ZhipuAI(api_key=api_key)

# 测试调用
try:
    response = client.chat.completions.create(
        model="glm-4",
        messages=[
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ]
    )
    print("调用成功！")
    print("AI回复：", response.choices[0].message.content)
except Exception as e:
    print(f"调用失败：{e}")