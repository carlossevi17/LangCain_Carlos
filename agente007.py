import streamlit as st
import os
from datetime import datetime

# Importaciones específicas del notebook
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- 1. CONFIGURACIÓN DE FECHA REAL ---
# Esto es clave para que no te diga que hoy es junio de 2024
hoy = datetime.now().strftime("%d de %B de %Y")

st.set_page_config(page_title="Agente Histórico", page_icon="📜")
st.title("📜 Agente Experto en Fechas")
st.write(f"📅 **Fecha de hoy en el sistema:** {hoy}")

# --- 2. ENTRADA DE API KEY ---
with st.sidebar:
    api_key = st.text_input("Introduce tu Google API Key", type="password")

if api_key:
    # Limpiamos la clave y la configuramos
    os.environ["GOOGLE_API_KEY"] = api_key.strip()
    
    try:
        # --- 3. CONFIGURACIÓN DEL MODELO Y HERRAMIENTAS ---
        # Gemini 1.5 Flash es el modelo que mejor funciona para esto
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        search = DuckDuckGoSearchResults()
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools = [search, wikipedia]

        # --- 4. EL PROMPT (Aquí está la magia de las fechas) ---
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Eres un historiador experto. 
            CONTEXTO: Hoy es día {hoy}.
            REGLA: Si te preguntan por una fecha exacta, edad de alguien o eventos pasados, 
            NO ADIVINES. Usa Wikipedia o Search para confirmar el año exacto."""),
            ("placeholder", "{history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # --- 5. EL AGENTE ---
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # Memoria para Streamlit
        msgs = StreamlitChatMessageHistory(key="chat_history")

        # Unimos todo con el historial
        agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: msgs,
            input_messages_key="input",
            history_messages_key="history",
        )

        # --- 6. INTERFAZ DE USUARIO ---
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if user_query := st.chat_input("Ej: ¿Qué pasó tal día como hoy? o ¿En qué año nació CR7?"):
            st.chat_message("human").write(user_query)
            
            with st.chat_message("ai"):
                with st.spinner("Consultando archivos..."):
                    config = {"configurable": {"session_id": "sesion_unica"}}
                    response = agent_with_history.invoke({"input": user_query}, config)
                    st.write(response["output"])

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Por favor, introduce tu API Key en la barra lateral.")
