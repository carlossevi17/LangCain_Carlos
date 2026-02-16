import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

st.title("🤖 Mi Agente LangChain")

# Configuración de API Key
with st.sidebar:
    api_key = st.text_input("Introduce tu Google API Key", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # 1. Herramientas (Copiadas de tu notebook)
    # Tu herramienta personalizada
    @tool
    def conchita_coins(usd_input: float) -> float:
        """Usa esta herramienta para convertir USD a Conchita Academy coins."""
        return 1.3 * float(usd_input)

    # Wikipedia (que es más estable que DuckDuckGo en Streamlit)
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    
    tools = [conchita_coins, wikipedia]

    # 2. Modelo (Versión estable)
    chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # 3. Prompt (Estructura de tu notebook)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente que usa Wikipedia para buscar info o convierte monedas."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 4. Agente
    agent = create_tool_calling_agent(chat, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Interfaz de chat
    user_input = st.text_input("¿En qué puedo ayudarte?")

    if user_input:
        with st.spinner("Buscando..."):
            response = agent_executor.invoke({"input": user_input})
            st.write(response["output"])
else:
    st.warning("Por favor, introduce tu API Key en la barra lateral.")
