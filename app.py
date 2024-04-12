import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import GPT4AllEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if pdf.name[-4::1] == '.pdf':
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()             
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = GPT4AllEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatGroq(temperature=1.0,
     groq_api_key=os.getenv("groq_api_key"),
      model_name="llama2-70b-4096")
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if st.session_state.conversation == None:
        st.error("Upload a file first please.")
        return
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your pdfs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.title('PDF Chat App💬')
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                if not raw_text:
                    st.error("No text found in your PDFs or No PDF uploaded.")
                    return
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                if len(text_chunks) == 0:
                    st.error("No text found in your PDFs or No PDF uploaded.")
                    return


                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

    st.header("Chat with your PDFs 💬")
    user_question = st.text_input("Ask any question about your documents:")
    if user_question:
        handle_userinput(user_question)


if __name__ == '__main__':
    main()
