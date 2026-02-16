import streamlit as st
import os
from datetime import datetime

# Importaciones base
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Configuración de fecha real
hoy = datetime.now().strftime("%d de %B de %Y")

st.set_page_config(page_title="Agente Fix", page_icon="🤖")
st.title("🤖 Agente 007: Versión Estable")

# Sidebar para la Key
with st.sidebar:
    st.header("Seguridad")
    key_input = st.text_input("Introduce tu API Key", type="password")

if key_input:
    # Limpieza total de la clave
    api_key = key_input.strip()
    
    try:
        # INICIALIZACIÓN DEL MODELO
        # Forzamos transport="rest" para evitar el error 'line 1 column 1'
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=api_key,
            temperature=0,
            transport="rest"
        )

        # Herramientas
        search = DuckDuckGoSearchResults()
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools = [search, wikipedia]

        # Prompt optimizado (como en tu notebook)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Eres un asistente experto. Hoy es {hoy}. Si no sabes un dato o una fecha, USA LAS HERRAMIENTAS."),
            ("placeholder", "{history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Crear agente y ejecutor
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True
        )

        # Memoria persistente en Streamlit
        msgs = StreamlitChatMessageHistory(key="chat_history_v3")

        # Agente con historial
        agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: msgs,
            input_messages_key="input",
            history_messages_key="history",
        )

        # --- INTERFAZ DE CHAT ---
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if user_input := st.chat_input("Pregúntame algo..."):
            st.chat_message("human").write(user_input)
            
            with st.chat_message("ai"):
                # Invocación con configuración de sesión
                config = {"configurable": {"session_id": "any"}}
                response = agent_with_history.invoke({"input": user_input}, config)
                st.write(response["output"])

    except Exception as e:
        st.error(f"Fallo de conexión: {e}")
else:
    st.info("Introduce tu API Key en la izquierda.")
