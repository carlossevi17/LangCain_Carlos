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

# --- CONFIGURACIÓN INICIAL (Igual al Notebook) ---
st.title("Agente LangChain del Notebook")

# Usar Secrets de Streamlit para la API Key
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Configura GOOGLE_API_KEY en los Secrets de Streamlit.")
    st.stop()

# Inicializar Chat (Corregido a 1.5-flash ya que 2.5 no existe)
chat = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')

# Herramientas (DuckDuckGo y Wikipedia)
search = DuckDuckGoSearchResults()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [search, wikipedia]

# --- LÓGICA DEL AGENTE CON MEMORIA ---

# Prompt exacto del apartado "Agent with memory"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Based on user query and the chat history, look for information using DuckDuckGo Search and Wikipedia and then give the final answer"),
    ("placeholder", "{history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Crear el agente usando langchain-classic
agent = create_tool_calling_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Historial persistente en la sesión de Streamlit
msgs = StreamlitChatMessageHistory(key="chat_messages")

# Función para limpiar el output (del notebook)
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

agent_executor_with_formatted_output = agent_executor | RunnableLambda(ensure_string_output)

# Configurar el ejecutable con historial
agent_with_history = RunnableWithMessageHistory(
    agent_executor_with_formatted_output,
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="history",
)

# --- INTERFAZ DE USUARIO ---
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if user_input := st.chat_input():
    st.chat_message("human").write(user_input)
    
    with st.chat_message("ai"):
        config = {"configurable": {"session_id": "sess1"}}
        response = agent_with_history.invoke({"input": user_input}, config)
        st.write(response["output"])
