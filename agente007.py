import streamlit as st
import os
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

# --- CONFIGURACIÓN DE FECHA ---
# Esto soluciona que el agente sepa qué día es hoy
fecha_actual = datetime.now().strftime("%d de %B de %Y")

st.set_page_config(page_title="Agente con Password", page_icon="🔑")
st.title("🕵️‍♂️ Agente Investigador")

# --- BARRA LATERAL PARA LA API KEY ---
with st.sidebar:
    st.header("Seguridad")
    # Entrada de texto modo password
    user_api_key = st.text_input("Introduce tu Google API Key", type="password")
    if not user_api_key:
        st.warning("Introduce la llave para activar al agente.")

# --- INICIALIZACIÓN (Solo si hay API Key) ---
if user_api_key:
    os.environ["GOOGLE_API_KEY"] = user_api_key
    
    # Configuramos el modelo (Temperatura 0 para evitar errores en fechas)
    chat = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=0)

    # Herramientas (Basado en el notebook)
    search = DuckDuckGoSearchResults()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [search, wikipedia]

    # Prompt con memoria y fecha inyectada
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Eres un asistente preciso. Hoy es {fecha_actual}. "
                   "Para fechas de nacimiento o datos de famosos, USA SIEMPRE Wikipedia."),
        ("placeholder", "{history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Crear agente usando la estructura del notebook
    agent = create_tool_calling_agent(chat, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Memoria de Streamlit
    msgs = StreamlitChatMessageHistory(key="chat_history")

    # Limpieza de salida (Lógica del notebook
