from typing import List, Optional
from langchain_core.embeddings import Embeddings


class ZhipuAIEmbeddings(Embeddings):
    """`Zhipuai Embeddings` embedding models，适配langchain接口"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "embedding-3",
        timeout: int = 60,
        batch_size: int = 64
    ):
        """
        实例化智谱Embedding客户端
        Args:
            api_key: 智谱API‑key，不传则自动读取环境变量 ZHIPUAI_API_KEY
            model_name: embedding模型名称
            timeout: http请求超时时间，单位秒
            batch_size: 单次接口最大条数，智谱限制最大64
        """
        from zhipuai import ZhipuAI

        self.model_name = model_name
        self.timeout = timeout
        self.batch_size = batch_size

        if api_key:
            self.client = ZhipuAI(api_key=api_key)
        else:
            self.client = ZhipuAI()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量生成文档 embedding，自动分批，每批不超过64条
        Args:
            texts: 待向量化文本列表
        Returns:
            List[List[float]]: 二维向量列表
        """
        if not texts:
            return []

        # 过滤空文本
        valid_texts = [t for t in texts if t.strip()]
        if len(valid_texts) == 0:
            return []

        all_embeddings = []
        # 切片分批，每一批最多batch_size条
        for i in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[i:i + self.batch_size]
            try:
                resp = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    timeout=self.timeout
                )
                batch_vec = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_vec)
            except Exception as e:
                raise RuntimeError(f"智谱embedding接口调用失败：{str(e)}") from e

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        生成查询文本 embedding，用于用户提问向量化
        Args:
            text: 用户查询文本
        Returns:
            List[float]: 一维向量
        """
        if not text.strip():
            raise ValueError("查询文本不能为空")

        vec_list = self.embed_documents([text])
        return vec_list[0]