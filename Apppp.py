import os
import pandas as pd
import streamlit as st
import requests

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain


# ---------------- VECTOR STORE CACHE ----------------
@st.cache_resource
def build_vector_store(df):

    url_columns = ['plant_url', 'Image URLs']
    plant_columns = [col for col in df.columns if col not in url_columns]

    df['plant_text'] = df[plant_columns].astype(str).agg(' '.join, axis=1)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.from_texts(
        df['plant_text'].tolist(),
        embeddings
    )

    return vector_store, df


def main():

    st.set_page_config(page_title="Medicinal Plant Chatbot", layout="wide")
    st.header("🌿 Medicinal Plant Knowledge Chatbot")


    # ---------------- API KEY ----------------
    filepath = os.getcwd()

    try:
        with open(os.path.join(filepath, "OpenAI_API_Key.txt"), "r") as f:
            openai_api_key = f.read().strip()

        os.environ["OPENAI_API_KEY"] = openai_api_key

    except FileNotFoundError:
        st.error("Please ensure OpenAI_API_Key.txt exists.")
        return


    # ---------------- CHAT MEMORY ----------------
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # ---------------- SIDEBAR ----------------
    with st.sidebar:

        st.title("Settings")

        dataset = st.file_uploader(
            "Upload Medicinal Plant Dataset (CSV)",
            type="csv"
        )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


    # ---------------- LOAD DATASET ----------------
    knowledge_base = None
    df = None

    if dataset is not None:

        df = pd.read_csv(dataset)

        knowledge_base, df = build_vector_store(df)

        st.sidebar.success("Dataset loaded! RAG is active.")


    # ---------------- DISPLAY CHAT HISTORY ----------------
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # ---------------- USER QUESTION ----------------
    if user_question := st.chat_input("Ask about medicinal plants..."):

        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user"):
            st.markdown(user_question)


        # ---------------- CONVERSATION CONTEXT ----------------
        conversation_context = ""

        for msg in st.session_state.messages[-4:]:
            role = msg["role"]
            content = msg["content"]
            conversation_context += f"{role}: {content}\n"


        contextual_question = f"""
        Conversation so far:
        {conversation_context}

        Based on the conversation answer the latest question:

        {user_question}
        """


        # ---------------- ASSISTANT RESPONSE ----------------
        with st.chat_message("assistant"):

            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=openai_api_key
            )

            answer = ""
            plant_row = None


            # -------- 1️⃣ RAG SEARCH --------
            if knowledge_base:

                docs = knowledge_base.similarity_search(
                    contextual_question,
                    k=2
                )

                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.invoke({
                    "input_documents": docs,
                    "question": contextual_question
                })

                answer = response["output_text"]


            # -------- 2️⃣ FIND MATCHING PLANT --------
            if df is not None:

                plant_row = df[
                    df['plant_text'].str.contains(
                        user_question,
                        case=False,
                        na=False
                    )
                ]


            # -------- 3️⃣ USE plant_url IF DATASET INFO MISSING --------
            missing_keywords = [
                "not available",
                "no information",
                "unknown",
                "not mentioned"
            ]

            if (
                plant_row is not None
                and not plant_row.empty
                and any(k in answer.lower() for k in missing_keywords)
            ):

                plant_url = plant_row.iloc[0]['plant_url']

                try:

                    page = requests.get(plant_url, timeout=5)

                    if page.status_code == 200:

                        page_text = page.text[:4000]

                        url_prompt = f"""
                        Use the webpage content to answer the question.

                        Webpage content:
                        {page_text}

                        Question:
                        {user_question}
                        """

                        response = llm.invoke(url_prompt)

                        answer = response.content

                except:
                    pass


            # -------- 4️⃣ DISPLAY IMAGE --------
            if plant_row is not None and not plant_row.empty:

                image_url = plant_row.iloc[0]['Image URLs']

                if isinstance(image_url, str) and image_url.startswith("http"):
                    st.image(image_url, caption="Plant Image", width=300)


            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )


if __name__ == '__main__':
    main()