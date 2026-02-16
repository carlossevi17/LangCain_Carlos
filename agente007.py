import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

st.title("🤖 Agente Sencillo LangChain")

# Pide la clave de API en la barra lateral
with st.sidebar:
    apikey = st.text_input("Introduce tu Google API Key", type="password")

if not apikey:
    st.info("Introduce la API Key para empezar.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = apikey

# 1. Configuración del Modelo (Gemini 1.5 Flash es el estándar actual)
chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# 2. Definición de Herramientas (Como tu ejemplo de conchita_coins)
@tool
def convert_to_galactic_credits(usd_amount: float) -> float:
    """Utiliza esta herramienta para convertir dólares USD a Créditos Galácticos."""
    return usd_amount * 1.5

# Herramienta de búsqueda (como en tu notebook)
search = DuckDuckGoSearchRun()
tools = [convert_to_galactic_credits, search]

# 3. Prompt sencillo (siguiendo tu ejemplo de AgentExecutor)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un asistente útil que usa herramientas para responder."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. Creación del Agente
agent = create_tool_calling_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Interfaz de usuario
user_query = st.text_input("Hazme una pregunta (ej: ¿Cuántos créditos galácticos son 50 dólares?)")

if user_query:
    with st.spinner("Pensando..."):
        # Ejecución directa sin memoria compleja para no complicar
        response = agent_executor.invoke({"input": user_query})
        st.markdown(f"**Respuesta:** {response['output']}")
