import streamlit as st
import json
import uuid
from base64 import b64decode
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_together import ChatTogether
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatHuggingFace
#from langchain_community.chat_models import ChatPerplexity
from langchain_anthropic import ChatAnthropic
from langchain_perplexity import ChatPerplexity
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import traceback
from operator import itemgetter
import torch
import os

os.environ["PYTORCH_ENABLE_META_TENSOR"] = "0"

st.title("Course : Digital AI strategy")

st.subheader ("Week 2: Contemporary Digital Technologies Overview, such as AI, Chatbot, NLP.")

# Sidebar: Choose provider & keys
provider = st.sidebar.selectbox(
    "Choose LLM Provider:",
    ("OpenAI", "Together", "Groq", "Hugging Face", "Anthropic", "Perplexity")
)
api_key = st.sidebar.text_input(f"{provider} API Key", type="password")
model_name = st.sidebar.text_input("Model name (optional)", "")

# Spacer to push warning downward (optional)
st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)

# AI warning box at bottom
st.sidebar.markdown(
    """
    <div style='
        border: 1px solid red;
        background-color: #ffe6e6;
        color: red;
        padding: 10px;
        font-size: 0.85em;
        border-radius: 5px;
        margin-top: 30px;
    '>
        ⚠️ This is an AI chat bot. Use caution when interpreting its responses. ⚠️
    </div>
    """,
    unsafe_allow_html=True,
)

### Change the below chroma DB path for changing the the vector DB

# Load prebuilt chroma DB path 
PERSIST_DIRECTORY = "./Week_2_16Sep2025"

### --------------------

# Sample Questions Section - Available without API key
with st.expander("💡 Sample Questions", expanded=False):
    st.markdown("### Get started with these example questions:")
    
    # Inference based
    st.markdown("**🦸 Case Study References**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎭 Case flow and Explanation", key="q1"):
            if 'sample_question' not in st.session_state:
                st.session_state.sample_question = None
            st.session_state.sample_question = "What are the chronological flow of events around design and use case of IBM watson in healthcare?"
    
    with col2:
        if st.button("Human-AI partnership question", key="q2"):
            if 'sample_question' not in st.session_state:
                st.session_state.sample_question = None
            st.session_state.sample_question = "In the collaboration between MD Anderson's doctors and IBM Watson, what were the biggest challenges to the human-AI team? "
    
    # Simple Explanations
    st.markdown("**👶 Simple Explanations**")
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("🧒 Explain to a 13-year-old", key="q3"):
            if 'sample_question' not in st.session_state:
                st.session_state.sample_question = None
            st.session_state.sample_question = "Explain the learnings from this session to a thirteen year old"
    
    with col4:
        if st.button("🦸‍♂️ Superhero Team Powers", key="q4"):
            if 'sample_question' not in st.session_state:
                st.session_state.sample_question = None
            st.session_state.sample_question = "If IBM Watson's cognitive computing was a superpower, what would be its kryptonite?"
    
    # Class Prep
    st.markdown("**🔗 Class Preparation**")
    col5, col6 = st.columns(2)
    
    with col5:
        if st.button("🧱 Direct question on case decision", key="q5"):
            if 'sample_question' not in st.session_state:
                st.session_state.sample_question = None
            st.session_state.sample_question = "What were the key factors that led to the failure of the IBM Watson for Oncology project at MD Anderson, and what broader lessons does this case offer for the adoption of AI in high-stakes industries?"
    
    with col6:
        if st.button("Technical Concepts", key="q6"):
            if 'sample_question' not in st.session_state:
                st.session_state.sample_question = None
            st.session_state.sample_question = "What are the technical concepts covered in this case?"

model = None

if api_key:
    try:
        if provider == "OpenAI" and api_key.startswith("sk-"):
            model = ChatOpenAI(
                api_key=api_key,
                model=model_name or "gpt-4o-mini",
                temperature=0.7
            )

        elif provider == "Together":
            model = ChatTogether(
                together_api_key=api_key,
                model=model_name or "mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.7
            )

        elif provider == "Groq":
            model = ChatGroq(
                groq_api_key=api_key,
                model_name=model_name or "llama-3.1-8b-instant",
                temperature=0.7
            )

        elif provider == "Hugging Face":
            # Typical model e.g. "HuggingFaceH4/zephyr-7b-beta"
            model = ChatHuggingFace(
                huggingfacehub_api_token=api_key,
                repo_id=model_name or "HuggingFaceH4/zephyr-7b-beta",
                temperature=0.7
            )

        elif provider == "Anthropic" and api_key.startswith("sk-ant-"):
            model = ChatAnthropic(
                anthropic_api_key=api_key,
                model_name=model_name or "claude-3-haiku-20240307",
                temperature=0.7
            )

        elif provider == "Perplexity" and api_key.startswith("pplx-"):
            model = ChatPerplexity(
                api_key=api_key,
                model=model_name or "sonar-pro",
                temperature=0.7
            )
        else:
            st.error("Unsupported provider or invalid API key format.")
    except Exception as e:
        st.error(f"Error initializing model: {e}")

if model:
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load vectorstore from disk instead of recreating it
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    # Cleaner parse_docs with expander
    def parse_docs(docs):
        return {"texts": docs}

    # Replace retriever with a RunnableLambda that does similarity_search
    def run_similarity_search(query):
        # k=5 to get top 5 docs, adjust as needed
        results = vectorstore.similarity_search(query, k=5)
        return results

    # Build prompt with expander
    def build_prompt(kwargs):
        ctx = kwargs["context"]
        question = kwargs["question"]
        context_text = "\n".join([d.page_content for d in ctx["texts"]])
        prompt_template = f"""
            Role: You are a helpful assistant for advance undergratuate students taking the Digital and AI strategy course. Your purpose is to help students understand the provided lecture notes and examples.
            Instructions:
            1.  Answer question only using the provided context. Do not use outside knowledge.
            2.  Maintain a polite and encouraging tone.
            3.  If a student asks a question that is not covered in the context, inform them that the question is outside the current topic of this session.
            4.  Suggest that they can search the web for more information if they are curious.
            5.  If a question is a duplicate, provide a more concise version of the previous answer.
            6.  Do not provide any in-text citations in your response without including reference list in the response.
            Context:
            {context_text}

            Question: {question}
            """

        return ChatPromptTemplate.from_messages(
            [{"role": "user", "content": prompt_template}]
        ).format_messages()

    # Compose chain using RunnableLambda for similarity_search + parse_docs
    chain = (
        {
            "context": itemgetter("question") | RunnableLambda(run_similarity_search) | RunnableLambda(parse_docs),
            "question": itemgetter("question")
        }
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )

    # Streamlit chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle sample question selection
    pending_question = None
    if 'sample_question' in st.session_state and st.session_state.sample_question:
        pending_question = st.session_state.sample_question
        st.session_state.sample_question = None  # Clear it after using
    
    user_input = st.chat_input("Ask a question...")
    if user_input:
        pending_question = user_input
    
    if pending_question:
        # Add the question to messages
        st.session_state.messages.append({"role": "user", "content": pending_question})
        st.chat_message("user").write(pending_question)

        try:
            answer = chain.invoke({"question": pending_question})
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
        except Exception as e:
            st.error(f"Error running RAG chain: {e}")
            st.error(traceback.format_exc())

else:
    # Handle sample question selection even without API key
    if 'sample_question' in st.session_state and st.session_state.sample_question:
        st.info(f"You selected: '{st.session_state.sample_question}' - Please enter your API key above to get an answer!")
        st.session_state.sample_question = None  # Clear it after showing
    
    st.warning("Please enter your API key and choose a provider.", icon="⚠")
