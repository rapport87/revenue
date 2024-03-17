from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os

os.makedirs("./.cache/files", exist_ok=True)
file_path = "./.cache/files/container.txt"
cache_dir = LocalFileStore("./.cache/embeddings/revenue")
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=2000,
    chunk_overlap=200,
)

st.set_page_config(
    page_title="국세청 GPT",
    page_icon="📃",
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-3.5-turbo-0125",
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            되도록 context의 내용을 사용하여 답변하세요
            답변할 수 없는 질문에는 답변을 만들어내지 마세요
            답변할 수 없다면 '질문에 대한 내용이 없습니다' 라고 답변하세요.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


@st.cache_data(show_spinner="Embedding file...")
def load_website(target_urls):
    loader = AsyncHtmlLoader(target_urls)
    loader.requests_per_second = 1
    docs = loader.load()
    container = ""
    for doc in docs:
        html_content = doc.page_content
        soup = BeautifulSoup(html_content, "html.parser")

        # class명이 'locationWrap'인 요소 제거
        for location_wrap in soup.find_all(class_="locationWrap"):
            location_wrap.decompose()

        # class명이 'zoomBox'인 요소 제거
        for zoom_box in soup.find_all(class_="zoomBox"):
            zoom_box.decompose()

        if container != "":
            container = container + "\n\n"

        container = (
            soup.find(id="container").get_text().replace("\n", " ").replace("\xa0", " ")
        )

        with open(file_path, "a", encoding="utf-8") as file:
            result_text = splitter.split_text(container)
            if result_text:
                # 리스트의 항목을 하나의 문자열로 결합
                combined_text = " ".join(result_text)
                file.write(combined_text)
                file.write("\n\n---\n\n")


def get_contents():
    loader = UnstructuredFileLoader(file_path)
    result = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(result, cached_embeddings)
    retriever = vector_store.as_retriever()
    return retriever


def get_urls():
    url = [
        "https://www.nts.go.kr/nts/nurisitemap.do?mi=6789",  # 전체포탈
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=12242&cntntsId=8631",  # 지급명세서 제출
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=6769&cntntsId=8165",  # 근로소득
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=6441&cntntsId=7877",  # 퇴직소득
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=6449&cntntsId=7885",  # 연금소득
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=6454&cntntsId=7890",  # 기타소득
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=6464&cntntsId=7900",  # 사업소득
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=6469&cntntsId=7905",  # 금융(이자·배당)소득
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=6480&cntntsId=7916",  # 비거주자 원천징수
    ]
    loader = AsyncHtmlLoader(url)
    docs = loader.load()
    print("Extracting content with LLM")

    all_urls = []

    for doc in docs:
        html_content = doc.page_content
        soup = BeautifulSoup(html_content, "html.parser")
        urls = [a.get("href") for a in soup.find_all("a", href=True)]
        filtered_urls = [url for url in urls if url.startswith("/nts")]
        filtered_urls = list(set(filtered_urls))
        all_urls += filtered_urls
        target_urls = [f"https://nts.go.kr{all_url}" for all_url in all_urls]
    target_urls = list(set(target_urls))
    return target_urls


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def reload_page():
    st.experimental_rerun()


# 초기화 버튼
if st.button("초기화"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    reload_page()

st.title("국세청AI")

st.markdown(
    """
안녕하세요!
            
국세청 사이트의 내용을 AI에게 질문을 할 수 있는 국세청AI(ChatGPT3.5) 입니다
    """
)

if not os.path.exists(file_path):
    final_urls = get_urls()
    contents = load_website(final_urls)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

retriever = get_contents()

if retriever:
    send_message("준비되었습니다 무엇이든 물어보세요", "ai", save=False)
    paint_history()
    message = st.chat_input("국세청에 대해 궁금한것을 물어보세요")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)
