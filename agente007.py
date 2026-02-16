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

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Agente 007 Preciso", page_icon="📅")

# 1. Obtener fecha actual para el contexto
fecha_actual = datetime.now().strftime("%d de %B de %Y")

# 2. Cargar API Key (Secrets)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Falta la GOOGLE_API_KEY en los Secrets.")
    st.stop()

# --- MODELO Y HERRAMIENTAS ---
chat = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=0) # Temp 0 para más precisión

search = DuckDuckGoSearchResults()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search, wikipedia]

# --- PROMPT MEJORADO ---
# Aquí es donde solucionamos lo de CR7 y la fecha actual
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""Eres un asistente de alta precisión. 
    CONTEXTO TEMPORAL: Hoy es {fecha_actual}.
    REGLA DE ORO: Si el usuario pregunta por fechas de nacimiento, eventos actuales o datos biográficos de personas famosas, DEBES usar Wikipedia o DuckDuckGo Search antes de responder. No confíes en tu memoria interna para años exactos."""),
    ("placeholder", "{history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Construcción del Agente
agent = create_tool_calling_agent(chat, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True
)

msgs = StreamlitChatMessageHistory(key="chat_history_pro")

# --- LÓGICA DE EJECUCIÓN ---
def ensure_string_output(agent_result: dict) -> dict:
    # Mantenemos tu lógica del notebook para procesar la salida
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

# --- INTERFAZ ---
st.title("🕵️‍♂️ Agente con Conciencia Temporal")
st.caption(f"📅 Fecha del sistema: {fecha_actual}")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if user_input := st.chat_input("Pregúntame por CR7 o qué día es hoy..."):
    st.chat_message("human").write(user_input)
    
    with st.chat_message("ai"):
        with st.status("Verificando datos reales...") as status:
            config = {"configurable": {"session_id": "user_1"}}
            response = agent_with_history.invoke({"input": user_input}, config)
            status.update(label="Datos verificados", state="complete")
        st.write(response["output"])
