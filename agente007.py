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
fecha_actual = datetime.now().strftime("%d de %B de %Y")

st.set_page_config(page_title="Agente 007", page_icon="🕵️‍♂️")
st.title("🕵️‍♂️ Agente Investigador")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Seguridad")
    # Añadimos una 'key' para que Streamlit mantenga el valor al recargar
    user_api_key = st.text_input("Introduce tu Google API Key", type="password", key="password_input")
    
    if st.button("Borrar historial"):
        st.session_state.chat_history = []
        st.rerun()

# --- INICIALIZACIÓN ---
if user_api_key:
    # Limpiar espacios y configurar entorno
    api_key_limpia = user_api_key.strip()
    os.environ["GOOGLE_API_KEY"] = api_key_limpia
    
    try:
        # MODELO CORREGIDO: Usamos 1.5-flash (el 3 no existe)
        chat = ChatGoogleGenerativeAI(
            model='gemini-3-flash-preview', 
            google_api_key=api_key_limpia,
            temperature=0
        )

        # Herramientas
        search = DuckDuckGoSearchResults()
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools = [search, wikipedia]

        # Prompt con memoria (exacto al notebook)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Eres un asistente preciso. Hoy es {fecha_actual}. "
                       "Para fechas de nacimiento o datos de famosos, USA SIEMPRE Wikipedia o Search."),
            ("placeholder", "{history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Agente
        agent = create_tool_calling_agent(chat, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        # Memoria persistente de Streamlit
        msgs = StreamlitChatMessageHistory(key="chat_history")

        # Función de limpieza de salida (Lógica de tu notebook)
        def ensure_string_output(agent_result: dict) -> dict:
            output_value = agent_result.get('output')
            if isinstance(output_value, list):
                concatenated_text = ""
                for item in output_value:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        concatenated_text += item.get('text', '')
                    elif isinstance(item, str):
                        concatenated_text += item
                agent_result['output'] = concatenated_text
            return agent_result

        # Encadenar con historial
        agent_with_history = RunnableWithMessageHistory(
            agent_executor | RunnableLambda(ensure_string_output),
            lambda session_id: msgs,
            input_messages_key="input",
            history_messages_key="history",
        )

        # --- INTERFAZ DE CHAT ---
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if user_input := st.chat_input("¿Qué quieres investigar?"):
            st.chat_message("human").write(user_input)
            
            with st.chat_message("ai"):
                with st.status("Consultando fuentes...") as status:
                    config = {"configurable": {"session_id": "any"}}
                    response = agent_with_history.invoke({"input": user_input}, config)
                    status.update(label="Investigación terminada", state="complete")
                st.write(response["output"])

    except Exception as e:
        st.error(f"Error de conexión: {e}. Revisa tu API Key y el modelo.")
else:
    st.info("Introduce tu API Key en la barra lateral para comenzar.")
