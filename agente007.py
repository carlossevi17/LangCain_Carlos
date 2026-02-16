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

# 1. Fecha real para que no alucine
hoy = datetime.now().strftime("%d de %B de %Y")

st.set_page_config(page_title="Agente Fijo", page_icon="🛠️")
st.title("🛠️ Agente 007: Corrección Total")

# 2. Entrada de clave en la barra lateral
with st.sidebar:
    st.header("Seguridad")
    key_input = st.text_input("Introduce tu API Key", type="password")

if key_input:
    # Limpieza de clave
    api_key = key_input.strip()
    os.environ["GOOGLE_API_KEY"] = api_key
    
    try:
        # 3. EL TRUCO: transport="rest" evita que la conexión se quede vacía
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=api_key,
            temperature=0,
            transport="rest"  # <--- ESTO ES LO QUE FALTA
        )

        # 4. Herramientas sencillas
        tools = [
            DuckDuckGoSearchResults(),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        ]

        # 5. Prompt directo (basado en tu notebook)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Eres un asistente que sabe que hoy es {hoy}. Si no sabes una fecha o dato de un famoso, búscala en las herramientas."),
            ("placeholder", "{history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 6. Agente y Memoria (Estructura del notebook)
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        msgs = StreamlitChatMessageHistory(key="chat_history_unique")

        agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: msgs,
            input_messages_key="input",
            history_messages_key="history",
        )

        # --- INTERFAZ ---
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if user_input := st.chat_input("Pregúntame algo"):
            st.chat_message("human").write(user_input)
            with st.chat_message("ai"):
                # Invocación directa
                response = agent_with_history.invoke(
                    {"input": user_input}, 
                    config={"configurable": {"session_id": "temp"}}
                )
                st.write(response["output"])

    except Exception as e:
        st.error(f"Error técnico: {e}")
else:
    st.info("Pon la API Key a la izquierda para activar el agente.")
