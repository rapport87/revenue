from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import SeleniumURLLoader
import streamlit as st
import os

os.makedirs("./.cache/files", exist_ok=True)
file_path = "./.cache/files/container.txt"
cache_dir = LocalFileStore("./.cache/embeddings/revenue")

llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-3.5-turbo-0125",
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    오직 정보에 있는 내용만 사용하여 사용자의 질문에 답하세요
    만일 정보에 적절한 답변이 없다면, 다른 작업을 하지 말고, "모르겠습니다"라고 답하세요.
    
    그후, 답변의 점수를 0에서 5 사이로 매기세요.

    답변이 사용자의 질문에 대한 답변이라면 점수는 높아야 하고, 그렇지 않다면 낮아야 합니다.

    답변의 점수가 0이더라도 항상 포함되도록 하세요.

    정보: {context}
                                                  
    예제:
    -----         
    정보: 지구에서 달까지의 거리는 384,400 Km 떨어져 있습니다.

    질문: 지구에서 달까지 얼마나 떨어져 있나요?
    답변: 달은 384,400 km 떨어져 있습니다.
    점수: 5

    질문 : 지구에서 달까지는 얼마나 떨어져 있나요?
    답변 : 지구와 달은 멀리 떨어져 있습니다
    점수 : 3

    질문: 지구에서 태양까지 얼마나 떨어져 있나요?
    답변: 모르겠습니다
    점수: 0
    -----

    위의 예제를 참고하여 답변하세요.

    질문: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            기존에 존재하는 답변만 사용하여 사용자의 질문에 답하세요.

            내용이 가장 자세하고 점수가 높은 답변을 선택하세요.

            답변: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(f"{answer['answer']}\n" for answer in answers)
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


def filter_fn(document):
    soup = BeautifulSoup(document.content, "html.parser")

    # 헤더와 푸터 제거
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.extract()
    if footer:
        footer.extract()

    return str(soup.get_text())


@st.cache_resource()
def get_urls():
    url = "https://www.nts.go.kr/nts/nurisitemap.do?mi=6789"
    loader = AsyncHtmlLoader(url)
    docs = loader.load()
    print("Extracting content with LLM")

    html_content = docs[0].page_content

    soup = BeautifulSoup(html_content, "html.parser")

    urls = [a.get("href") for a in soup.find_all("a", href=True)]
    filtered_urls = [url for url in urls if url.startswith("/nts")]
    filtered_urls = list(set(filtered_urls))
    # target_urls = [f"https://nts.go.kr{filtered_url}" for filtered_url in filtered_urls]
    target_urls = [
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2265&cntntsId=7690",
        "https://www.nts.go.kr/nts/cm/cntnts/cntntsView.do?mi=2272&cntntsId=7693",
    ]
    return target_urls


@st.cache_data(show_spinner="Embedding file...")
def load_website(target_urls):
    loader = AsyncHtmlLoader(target_urls)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000,
        chunk_overlap=200,
    )
    loader.requests_per_second = 1
    docs = loader.load()
    container = ""
    for doc in docs:
        html_content = doc.page_content  # 각 문서의 HTML 내용
        soup = BeautifulSoup(html_content, "html.parser")

        # class 이름이 'locationWrap'인 모든 요소를 찾아서 제거합니다.
        for location_wrap in soup.find_all(class_="locationWrap"):
            location_wrap.decompose()

        # class 이름이 'zoomBox'인 모든 요소를 찾아서 제거합니다.
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

    loader = UnstructuredFileLoader(file_path)
    result = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(result, cached_embeddings)
    vector_store.as_retriever()
    return result


urls = get_urls()
retriever = load_website(urls)

if retriever:
    st.write("Loaded successfully")

st.markdown(
    """
안녕하세요!
            
국세청에서 제공하는 정보를 검색할 수 있는 Renenue 챗봇입니다

아래에 검색할 내용을 입력하시면,

국세청 홈페이지에서 검색하여 답변을 제공해드립니다.
"""
)

# query = st.text_input("국세청 홈페이지에서 검색할 내용을 입력하세요.")
# if query:
#     chain = (
#         {
#             "docs": retriever,
#             "question": RunnablePassthrough(),
#         }
#         | RunnableLambda(get_answers)
#         | RunnableLambda(choose_answer)
#     )
#     result = chain.invoke(query)
#     st.markdown(result.content.replace("$", "\$"))
