import streamlit as st
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
# from langchain_together import ChatTogether
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_perplexity import ChatPerplexity
import traceback
import os
import shutil
import tempfile

os.environ["PYTORCH_ENABLE_META_TENSOR"] = "0"

st.set_page_config(page_title="Week 2 · Digital AI Strategy", page_icon="🎓")

st.title("Course : Digital AI strategy")

st.subheader ("Week 2: Contemporary Digital Technologies Overview, such as AI, Chatbot, NLP.")

# Provider settings: API key prefix (None = not checked), default model, where to get a key
PROVIDERS = {
    "OpenAI": {"prefix": "sk-", "default": "gpt-4o-mini", "keys": "https://platform.openai.com/api-keys"},
    # "Together" is disabled; see the commented-out block further down
    "Groq": {"prefix": None, "default": "openai/gpt-oss-20b", "keys": "https://console.groq.com/keys"},
    "Hugging Face": {"prefix": None, "default": "openai/gpt-oss-20b", "keys": "https://huggingface.co/settings/tokens"},
    "Anthropic": {"prefix": "sk-ant-", "default": "claude-haiku-5-5", "keys": "https://platform.claude.com/settings/keys"},
    "Perplexity": {"prefix": "pplx-", "default": "the 'fast' preset", "keys": "https://console.perplexity.ai"},
}

# Sidebar: Choose provider & keys
provider = st.sidebar.selectbox("Choose LLM Provider:", list(PROVIDERS))
settings = PROVIDERS[provider]
api_key = st.sidebar.text_input(f"{provider} API Key", type="password").strip()
st.sidebar.caption(f"Need a key? [Get a {provider} API key]({settings['keys']})")
model_name = st.sidebar.text_input(
    "Model name (optional)",
    "",
    placeholder=f"Default: {settings['default']}",
    help="Leave blank to use the default model."
    + (" Perplexity model names look like 'openai/gpt-5.6-sol'." if provider == "Perplexity" else ""),
).strip()

if st.sidebar.button("🗑️ New chat"):
    st.session_state.messages = []

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
            st.session_state.sample_question = "what can I learn from this session for overall digital and AI strategy course?"
    
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
            st.session_state.sample_question = "What are the technical concepts covered in this case? Explain in simple terms too"

model = None

if api_key:
    key_error = None
    if provider == "OpenAI" and api_key.startswith("sk-ant-"):
        key_error = "That looks like an Anthropic key. Choose Anthropic as the provider, or paste an OpenAI key."
    elif settings["prefix"] and not api_key.startswith(settings["prefix"]):
        key_error = (
            f"{provider} API keys start with '{settings['prefix']}'. "
            "Check that you chose the right provider and pasted the whole key."
        )

    if key_error:
        st.error(key_error)
    else:
        try:
            if provider == "OpenAI":
                model = ChatOpenAI(
                    api_key=api_key,
                    model=model_name or "gpt-4o-mini",
                    temperature=0.7
                )

            #elif provider == "Together":
            #    model = ChatTogether(
            #        together_api_key=api_key,
            #        model=model_name or "mistralai/Mistral-7B-Instruct-v0.2",
            #        temperature=0.7
            #    )

            elif provider == "Groq":
                model = ChatGroq(
                    groq_api_key=api_key,
                    model_name=model_name or "openai/gpt-oss-20b",
                    temperature=0.7
                )

            elif provider == "Hugging Face":
                # Any chat model served by Hugging Face Inference Providers
                llm = HuggingFaceEndpoint(
                    huggingfacehub_api_token=api_key,
                    repo_id=model_name or "openai/gpt-oss-20b",
                    provider="auto",
                    max_new_tokens=1024,
                    temperature=0.7
                )
                model = ChatHuggingFace(llm=llm)

            elif provider == "Anthropic":
                # The Anthropic API no longer accepts temperature, so none is passed
                model = ChatAnthropic(
                    anthropic_api_key=api_key,
                    model_name=model_name or "claude-haiku-5-5"
                )

            elif provider == "Perplexity":
                # Sonar models were retired; Perplexity now serves the Agent API.
                # The "fast" preset replaces "sonar"; a model name must be "provider/model".
                if model_name:
                    model = ChatPerplexity(api_key=api_key, model=model_name, use_responses_api=True)
                else:
                    model = ChatPerplexity(api_key=api_key, use_responses_api=True, model_kwargs={"preset": "fast"})
        except Exception as e:
            st.error(f"Could not set up the {provider} model: {e}")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_SOURCE = os.path.join(APP_DIR, PERSIST_DIRECTORY)
DB_COPY = os.path.join(tempfile.gettempdir(), "chroma_" + os.path.basename(os.path.normpath(DB_SOURCE)))

# Load the embedding model and vectorstore once and share them across all
# sessions. Chroma opens a fresh, writable copy of the database so it never
# writes into the app's source folder.
@st.cache_resource(show_spinner="Loading course materials...")
def load_vectorstore():
    shutil.rmtree(DB_COPY, ignore_errors=True)
    shutil.copytree(DB_SOURCE, DB_COPY, ignore=shutil.ignore_patterns("*-wal", "*-shm", "*-journal"))
    for root, dirs, files in os.walk(DB_COPY):
        os.chmod(root, 0o755)
        for name in files:
            os.chmod(os.path.join(root, name), 0o644)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=DB_COPY,
        embedding_function=embedding_model
    )

def describe_folder(path):
    if not os.path.isdir(path):
        return f"{path}: missing"
    lines = [f"{path}:"]
    for root, dirs, files in os.walk(path):
        for name in sorted(files):
            full = os.path.join(root, name)
            lines.append(f"  {os.path.relpath(full, path)}  {os.path.getsize(full)} bytes")
    return "\n".join(lines)

# Earlier question/answer pairs sent along with each question, so follow-ups work
HISTORY_TURNS = 3

def build_prompt(question, docs, history):
    context_text = "\n\n".join(d.page_content for d in docs)
    history_text = "\n\n".join(
        f"{'Student' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-2 * HISTORY_TURNS:]
    ) or "(none)"
    prompt = f"""
Role: You are a helpful assistant for advanced undergraduate students taking the Digital and AI strategy course. Your purpose is to help students understand the provided lecture notes and examples.
Instructions:
1.  Answer questions only using the provided context and the conversation so far. Do not use outside knowledge.
2.  Maintain a polite and encouraging tone.
3.  If a student asks a question that is not covered in the context, inform them that the question is outside the current topic of this session.
4.  Suggest that they can search the web for more information if they are curious.
5.  If a question repeats an earlier one, provide a more concise version of the previous answer.
6.  Do not provide any in-text citations in your response without including reference list in the response.

Conversation so far:
{history_text}

Context:
{context_text}

Question: {question}
"""
    return [HumanMessage(content=prompt)]

def friendly_error(e):
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    name = model_name or settings["default"]
    if status in (401, 403):
        return (
            f"{provider} rejected your API key. Check that it was copied correctly, "
            "is still active, and that your account has API access or credit."
        )
    if status == 404:
        return f"The model '{name}' isn't available from {provider}. Clear the Model name box to use the default, or check the spelling."
    if status == 429:
        return f"{provider} says you've hit a usage limit or run out of credit. Wait a minute and try again, or check your {provider} account."
    if status == 400:
        return f"{provider} couldn't process the request. This is often an unsupported model name, so try clearing the Model name box."
    if status and status >= 500:
        return f"{provider} is having problems right now. Please try again in a minute, or choose another provider."
    return "Something went wrong while getting an answer. Please try again, or choose another provider."

def show_text(text):
    # Show "$" literally: Streamlit would read "$5 to $10" as a LaTeX formula
    st.markdown(text.replace("$", "\\$"))

def show_sources(sources):
    if sources:
        with st.expander("📚 Sources from the course materials"):
            for i, snippet in enumerate(sources, 1):
                # Escape characters Streamlit would read as Markdown or LaTeX
                for ch in "\\`*_#[]<>$":
                    snippet = snippet.replace(ch, "\\" + ch)
                st.markdown(f"**{i}.** {snippet}")

if model:

    try:
        vectorstore = load_vectorstore()
    except Exception as e:
        # Show the real error: Streamlit hides the message of uncaught exceptions
        import chromadb
        st.error("Could not load the course materials. Please share the details below with the app maintainer.")
        st.code(
            f"{type(e).__name__}: {e}\n\n"
            f"chromadb {chromadb.__version__}, Python {sys.version.split()[0]}\n"
            f"cwd: {os.getcwd()}\n"
            f"{describe_folder(DB_SOURCE)}\n"
            f"{describe_folder(DB_COPY)}"
        )
        st.stop()

    # Streamlit chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            show_text(msg["content"])
            show_sources(msg.get("sources"))

    # Handle sample question selection
    pending_question = None
    if 'sample_question' in st.session_state and st.session_state.sample_question:
        pending_question = st.session_state.sample_question
        st.session_state.sample_question = None  # Clear it after using

    user_input = st.chat_input("Ask a question about this week's session...")
    if user_input:
        pending_question = user_input

    if pending_question:
        history = list(st.session_state.messages)
        # Add the question to messages
        st.session_state.messages.append({"role": "user", "content": pending_question})
        with st.chat_message("user"):
            show_text(pending_question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    docs = vectorstore.similarity_search(pending_question, k=5)
                    answer = (model | StrOutputParser()).invoke(build_prompt(pending_question, docs, history))
                sources = [
                    (text[:300] + "…") if len(text) > 300 else text
                    for text in (" ".join(d.page_content.split()) for d in docs)
                ]
                show_text(answer)
                show_sources(sources)
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
            except Exception as e:
                st.error(friendly_error(e))
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

else:
    # Keep a sample question chosen before the API key is entered, and answer it once a key is in
    if 'sample_question' in st.session_state and st.session_state.sample_question:
        st.info(
            f"You selected: '{st.session_state.sample_question}'. "
            "It will be answered as soon as you enter your API key."
        )

    if not api_key:
        st.warning(
            "Please choose a provider and enter your API key in the sidebar. "
            "On a phone, tap the arrow at the top-left of the screen to open it.",
            icon="⚠",
        )
