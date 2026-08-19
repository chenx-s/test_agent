import streamlit as st
from langchain_zhipu import ChatZhipuAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

# ----------------------密钥兼容：本地.env + 云端st.secrets----------------------
load_dotenv(find_dotenv())

# 优先读环境变量，其次读streamlit secrets
zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
if not zhipuai_api_key:
    try:
        zhipuai_api_key = st.secrets["ZHIPUAI_API_KEY"]
    except Exception:
        zhipuai_api_key = None

# 把key写入os.environ，供zhipuai_embedding内部读取
if zhipuai_api_key:
    os.environ["ZHIPUAI_API_KEY"] = zhipuai_api_key


def build_vector_store_from_pdf(pdf_bytes):
    """内存构建向量库，不持久化到磁盘，适配streamlit cloud"""
    # 保存临时pdf
    temp_path = "/tmp/temp_upload.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_bytes)

    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(docs)
    embedding = ZhipuAIEmbeddings()
    # persist_directory=None → 纯内存模式，不写磁盘
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=None
    )
    return vectordb.as_retriever()


def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])


@st.cache_resource(show_spinner="正在初始化大模型...")
def get_llm():
    llm = ChatZhipuAI(
        model="glm-4-plus",
        temperature=0.1,
        api_key=zhipuai_api_key
    )
    return llm


def get_qa_history_chain(retriever):
    llm = get_llm()
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate.from_messages([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever,),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context=retrieve_docs,
    ).assign(answer=qa_chain)
    return qa_history_chain


def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]


def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发')

    # 密钥校验，页面友好报错，不会直接黑屏崩溃
    if not zhipuai_api_key:
        st.error("❌ ZHIPUAI_API_KEY 未配置！\n本地请检查.env，云端请在Settings-Secrets配置密钥")
        st.stop()

    # 初始化会话状态
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = None

    # 侧边栏上传PDF构建知识库
    with st.sidebar:
        st.subheader("📄 上传PDF知识库")
        uploaded_file = st.file_uploader("请上传PDF文档", type=["pdf"])
        if uploaded_file is not None:
            with st.spinner("正在解析PDF，构建向量知识库..."):
                retriever = build_vector_store_from_pdf(uploaded_file.read())
                st.session_state.qa_history_chain = get_qa_history_chain(retriever)
            st.success("✅ 知识库构建完成，可以开始提问！")

    messages = st.container(height=550)
    # 渲染历史对话
    for message in st.session_state.messages:
        with messages.chat_message(message[0]):
            st.write(message[1])

    prompt = st.chat_input("Say something")
    if prompt:
        if st.session_state.qa_history_chain is None:
            st.warning("⚠️请先在侧边栏上传PDF知识库！")
        else:
            st.session_state.messages.append(("human", prompt))
            with messages.chat_message("human"):
                st.write(prompt)
            answer = gen_response(
                chain=st.session_state.qa_history_chain,
                input=prompt,
                chat_history=st.session_state.messages
            )
            with messages.chat_message("ai"):
                output = st.write_stream(answer)
            st.session_state.messages.append(("ai", output))


if __name__ == "__main__":
    main()