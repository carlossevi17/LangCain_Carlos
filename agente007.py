import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Título de la App (Basado en tu notebook)
st.title("Conchita EDEM Agent 🤖")

# 1. Configuración de API Key (Sustituye a getpass)
with st.sidebar:
    api_key = st.text_input("Provide your Google API Key:", type="password")

if not api_key:
    st.info("Por favor, introduce tu API Key para comenzar.")
    st.stop()

os.environ['GOOGLE_API_KEY'] = api_key

# 2. Definición de Herramientas (Copiadas exactamente de tu .ipynb)
@tool
def conchita_coins(input: float) -> float:
    """Use this tool to convert USD to Conchita Academy coins"""
    return 1.3 * (float(input))

# Configuración de Wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [conchita_coins, wikipedia]

# 3. Configuración del Modelo
# Nota: Usamos 'gemini-1.5-flash' porque el nombre '2.5' causa el ValidationError que recibiste.
chat = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

# 4. Prompt (Misma estructura que tu notebook)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. For answering the user query, look for information using Wikipedia and then give the final answer"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 5. Agente y Executor
agent = create_tool_calling_agent(chat, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. Interfaz de entrada
user_input = st.text_input("Haz tu consulta:")

if user_input:
    with st.spinner("El agente está pensando..."):
        try:
            # Invocación final idéntica a tu notebook
            result = agent_executor.invoke({"input": user_input})
            st.markdown(f"**Resultado:** {result['output']}")
        except Exception as e:
            st.error(f"Error: {e}")
