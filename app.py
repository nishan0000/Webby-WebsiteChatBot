import streamlit as st

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

def create_vectorstore(url):
    
    # Implementing loader
    loader = WebBaseLoader(url)
    doc = loader.load()
    
    # Splitting the documents using recursive character text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=60,
        length_function=len
    )
    chunks = splitter.split_documents(doc)
    
    # Creating Embeddings
    embedddings = OpenAIEmbeddings()
    
    # Creating a vetorstore
    vectorstore = Chroma.from_documents(chunks, embedddings)
    
    # Returning the vectorstore
    return vectorstore

def create_context_retriever_chain(vectorstore):
    
    # Creating LLM instance
    llm = ChatOpenAI(temperature=0.0)
    
    # Creating a retriever
    retriever = vectorstore.as_retriever()
    
    # Creating the prompt
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    # Creating the retriever chain
    retriever_chain = create_history_aware_retriever(llm, retriever=retriever, prompt=prompt)
    
    # Returning the retriever chain
    return retriever_chain

def create_conversation_rag_chain(retriever_chain):
    
    # Creating LLM instance
    llm = ChatOpenAI(temperature=0.0)
    
    # Creating the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name='chat_history'),
        ("user", "{input}")
    ])
    
    # Creating stuff chain
    stuff_doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    # Creating the final retriever chain
    retrieval_chain = create_retrieval_chain(retriever_chain, stuff_doc_chain)
    
    # Returning the retrieval chain
    return retrieval_chain

def get_response(user_input):
    
    # Creating retriever chain
    retriever_chain = create_context_retriever_chain(st.session_state.vectorstore)
    # Creating conversation rag chain
    conversation_rag_chain = create_conversation_rag_chain(retriever_chain)
    
    # Collecting response for input query and saving the chat history
    response = conversation_rag_chain.invoke({
        'chat_history' : st.session_state.chat_history, 
        'input' : user_input
    })

    # Return answer
    return response['answer']

# Configuring the streamlit app
st.set_page_config(page_title='Webchat', page_icon='🤖')
st.title('Chat With Websites')

# Creating a sidebar
with st.sidebar:
    st.header("Settings")
    url = st.text_input("Enter a Website URL")
    
# Checking if link is valid or not
if url == None or url == "":
    st.info("Enter a Valid Website Link")
else:
    # Adding chat_history to session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a Webby. How can i help you?")]
    # Adding vectorstore to session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = create_vectorstore(url)
        
        
    # Getting answer for input from user
    user_question = st.chat_input("Type your question here..")
    if user_question!=None and user_question!="":
        response = get_response(user_question)
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=response))
        
    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)



# In[ ]:




