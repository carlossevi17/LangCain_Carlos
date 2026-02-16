import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

# --- CONFIGURACIÓN ESTÉTICA ---
st.set_page_config(page_title="Agente 007 Pro", page_icon="🕵️‍♂️", layout="wide")

st.markdown("""
    <style>
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    .stSpinner { border-top-color: #4CAF50 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.title("⚙️ Configuración")
    st.info("Este agente utiliza Google Gemini, Wikipedia y DuckDuckGo.")
    
    if "GOOGLE_API_KEY" in st.secrets:
        st.success("✅ API Key cargada correctamente")
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        api_key = st.text_input("Introduce tu Google API Key", type="password")
    
    if st.button("🗑️ Limpiar Historial"):
        st.session_state.chat_messages = []
        st.rerun()

# --- INICIALIZACIÓN DE COMPONENTES (Basado en tu Notebook) ---
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    chat = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=0.7)
    
    # Herramientas
    search = DuckDuckGoSearchResults()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [search, wikipedia]

    # Prompt con memoria (Apartado 'Agent with memory' del notebook)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente experto. Utiliza el historial para dar contexto y las herramientas para datos actuales."),
        ("placeholder", "{history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Agente y Ejecutor
    agent = create_tool_calling_agent(chat, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Memoria de Streamlit
    msgs = StreamlitChatMessageHistory(key="chat_messages")

    # Formateo de salida (Tu lógica del notebook)
    def ensure_string_output(agent_result: dict) -> dict:
        output_value = agent_result.get('output')
        if isinstance(output_value, list):
            concatenated_text = "".join([item.get('text', '') if isinstance(item, dict) else str(item) for item in output_value])
            agent_result['output'] = concatenated_text
        return agent_result

    agent_with_history = RunnableWithMessageHistory(
        agent_executor | RunnableLambda(ensure_string_output),
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="history",
    )

    # --- INTERFAZ DE CHAT ---
    st.title("🕵️‍♂️ Agente Inteligente con Memoria")

    # Mostrar mensajes previos
    for msg in msgs.messages:
        role = "user" if msg.type == "human" else "assistant"
        st.chat_message(role).write(msg.content)

    # Input de usuario
    if user_query := st.chat_input("¿Qué quieres investigar hoy?"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            # Contenedor para ver los "pasos internos"
            with st.status("🔍 Investigando en la red...", expanded=False) as status:
                try:
                    config = {"configurable": {"session_id": "streamlit_user"}}
                    response = agent_with_history.invoke({"input": user_query}, config)
                    status.update(label="✅ Investigación completada", state="complete", expanded=False)
                    st.write(response["output"])
                except Exception as e:
                    status.update(label="❌ Error en la búsqueda", state="error")
                    st.error(f"Hubo un problema: {str(e)}")
else:
    st.warning("⚠️ Por favor, introduce tu API Key para empezar.")
