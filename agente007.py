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

# 1. Configuración de fecha real
fecha_actual = datetime.now().strftime("%d de %B de %Y")

st.set_page_config(page_title="Agente con Password", page_icon="🔑")
st.title("🕵️‍♂️ Agente Investigador")

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Configuración")
    # Usamos session_state para que la clave no se borre al recargar
    user_api_key = st.text_input("Introduce tu Google API Key", type="password", key="api_key_input")
    
    if st.button("Limpiar Chat"):
        st.session_state.chat_history_pro = []
        st.rerun()

# --- VALIDACIÓN Y CARGA DEL AGENTE ---
if user_api_key:
    # Configurar el entorno
    os.environ["GOOGLE_API_KEY"] = user_api_key
    
    # Definir componentes (Basado en tu notebook)
    try:
        chat = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=0)
        
        search = DuckDuckGoSearchResults()
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools = [search, wikipedia]

        # Prompt del apartado 'Agent with memory'
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Eres un asistente preciso. Hoy es {fecha_actual}. "
                       "Para datos de famosos (como CR7) o fechas, USA SIEMPRE Wikipedia o Search."),
            ("placeholder", "{history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Crear agente
        agent = create_tool_calling_agent(chat, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        # Memoria persistente
        msgs = StreamlitChatMessageHistory(key="chat_history_pro")

        # Función de limpieza (del notebook)
        def ensure_string_output(agent_result: dict) -> dict:
            output_value = agent_result.get('output')
            if isinstance(output_value, list):
                concatenated_text = "".join([i.get('text', '') if isinstance(i, dict) else str(i) for i in output_value])
                agent_result['output'] = concatenated_text
            return agent_result

        agent_with_history = RunnableWithMessageHistory(
            agent_executor | RunnableLambda(ensure_string_output),
            lambda session_id: msgs,
            input_messages_key="input",
            history_messages_key="history",
        )

        # --- INTERFAZ DE CHAT ---
        # Mostrar mensajes antiguos
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        # Entrada de chat
        if user_input := st.chat_input("¿Qué quieres saber?"):
            st.chat_message("human").write(user_input)
            
            with st.chat_message("ai"):
                with st.status("Investigando...") as status:
                    config = {"configurable": {"session_id": "session_123"}}
                    response = agent_with_history.invoke({"input": user_input}, config)
                    status.update(label="✅ Listo", state="complete")
                st.write(response["output"])
                
    except Exception as e:
        st.error(f"Error de conexión: Verifica que tu API Key sea válida. Detalle: {e}")

else:
    st.info("Para empezar, introduce tu API Key en la barra lateral izquierda 👈")
    st.image("https://img.icons8.com/clouds/200/lock-landscape.png") # Ilustración de bloqueo
