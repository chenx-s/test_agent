# 类型提示导入，用于标注变量类型，提升代码可读性、IDE语法检查
from typing import Any, Dict, Iterator, List, Optional
# 导入智谱官方SDK客户端
from zhipuai import ZhipuAI
# LangChain回调管理器，LLM运行时回调（例如新token流出、日志记录）
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
# LangChain大模型基类，自定义LLM必须继承这个基类，遵守langchain协议规范
from langchain_core.language_models import BaseChatModel
# LangChain消息对象：各种角色消息类型
from langchain_core.messages import (
    AIMessage,          # AI返回消息（非流式完整结果）
    AIMessageChunk,     # AI流式分片消息，stream模式下每一小块内容
    BaseMessage,        # 所有消息的抽象基类
    SystemMessage,      # system系统提示消息
    ChatMessage,        # 通用聊天消息，可以自定义role
    HumanMessage        # 用户人类消息
)
# token用量元数据，记录输入、输出、总token消耗
from langchain_core.messages.ai import UsageMetadata
# LangChain输出对象：封装大模型返回结果
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
import time  # 用于统计API接口耗时


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """把LangChain的消息格式转为智谱API所需要的字典格式
    Args:
        message: The LangChain message.
    Returns:
        转换完成，符合智谱接口要求的字典 {"role":"xxx","content":"xxx"}
    """
    # 初始化字典，content直接赋值消息文本内容
    message_dict: Dict[str, Any] = {"content": message.content}
    # 如果消息存在name字段，把name放进字典（部分agent场景会用到）
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # 根据消息实例类型映射智谱要求的role字段
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"       # 用户消息 → 智谱role: user
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant" # AI消息 → 智谱role: assistant
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"    # 系统提示消息 → 智谱role: system
    else:
        # 遇到未知消息类型直接抛出异常
        raise TypeError(f"Got unknown type {message}")
    return message_dict


# 继承 LangChain 的 BaseChatModel 基类
class ZhipuaiLLM(BaseChatModel):
    """自定义Zhipuai聊天模型。
    实现langchain标准接口：支持普通调用、流式输出，可直接接入RAG、Chain。
    """
    # 模型名称，例如 "glm‑4‑plus"
    model_name: str = None
    # 温度参数，控制生成随机性
    temperature: Optional[float] = None
    # max_tokens：最大输出token数量
    max_tokens: Optional[int] = None
    # http请求超时时间
    timeout: Optional[int] = None
    # stop 停止词列表，遇到指定词大模型停止生成
    stop: Optional[List[str]] = None
    # 请求失败最大重试次数（本代码目前逻辑没有写重试逻辑，只是预留字段）
    max_retries: int = 3
    # 智谱API密钥
    api_key: str | None = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """【非流式同步调用】通过调用智谱API从而响应输入。
        Args:
            messages: 由messages列表组成的prompt，langchain消息对象列表
            stop: 在模型生成的回答中有该字符串列表中的元素则停止响应
            run_manager: 一个为LLM提供回调的运行管理器
        返回 ChatResult：langchain标准封装好的大模型完整返回对象
        """
        # 列表推导式，把langchain消息全部转换成智谱接口字典格式
        messages = [_convert_message_to_dict(message) for message in messages]
        # 记录调用开始时间，统计接口耗时
        start_time = time.time()
        # 实例化智谱客户端，发起非流式对话请求
        response = ZhipuAI(api_key=self.api_key).chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            messages=messages
        )
        # 计算接口总耗时（秒）
        time_in_seconds = time.time() - start_time

        # 将智谱原始返回，封装为LangChain的AIMessage对象
        message = AIMessage(
            content=response.choices[0].message.content, # 模型回答文本
            additional_kwargs={},
            response_metadata={
                "time_in_seconds": round(time_in_seconds, 3), # 保存接口耗时
            },
            # token消耗统计
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,      # 输入token
                "output_tokens": response.usage.completion_tokens,  # 输出token
                "total_tokens": response.usage.total_tokens,        # 总token
            },
        )
        # ChatGeneration包装单条回答
        generation = ChatGeneration(message=message)
        # ChatResult是langchain标准返回对象，外层容器，支持多候选结果
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """【流式输出】调用智谱API，返回迭代器，逐块返回回答内容。
        Args:
            messages: 由messages列表组成的prompt
            stop: 在模型生成的回答中有该字符串列表中的元素则停止响应
            run_manager: 一个为LLM提供回调的运行管理器
        yield ChatGenerationChunk：每一块流式分片对象，生成器迭代输出
        """
        # 消息格式转换
        messages = [_convert_message_to_dict(message) for message in messages]
        # stream=True开启流式，返回迭代器对象
        response = ZhipuAI(api_key=self.api_key).chat.completions.create(
            model=self.model_name,
            stream=True,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            messages=messages
        )
        start_time = time.time()
        usage_metadata = None
        # 循环遍历流式返回分片
        for res in response:
            # 流式最后一个分片会带上usage用量统计，捕获token消耗
            if res.usage:
                usage_metadata = UsageMetadata(
                    {
                        "input_tokens": res.usage.prompt_tokens,
                        "output_tokens": res.usage.completion_tokens,
                        "total_tokens": res.usage.total_tokens,
                    }
                )
            # 把delta增量文本封装成分块消息对象
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=res.choices[0].delta.content)
            )

            if run_manager:
                # langchain回调：每流出一块token触发回调，用于日志、监控
                run_manager.on_llm_new_token(res.choices[0].delta.content, chunk=chunk)
            # yield输出分片，作为生成器，上层可以for循环迭代获取
            yield chunk

        time_in_sec = time.time() - start_time
        # 流全部结束，发送一个空内容的结束分片，携带耗时、token用量元数据
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": round(time_in_sec, 3)}, usage_metadata=usage_metadata)
        )
        if run_manager:
            run_manager.on_llm_new_token("", chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """获取此聊天模型使用的语言模型类型。
        langchain内部属性，用于识别模型类别。
        """
        return self.model_name

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回一个标识参数的字典。
        该信息由LangChain回调系统使用，用于跟踪、日志记录。
        """
        return {
            "model_name": self.model_name,
        }