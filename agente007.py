import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

st.title("🚀 HAL-9001: Logística Intergaláctica")

# 1. Interfaz para la API Key en el lateral (Sidebar)
with st.sidebar:
    google_api_key = st.text_input("Introduce tu Google API Key", type="password")
    "[Consigue tu API Key aquí](https://aistudio.google.com/app/apikey)"

if not google_api_key:
    st.info("Por favor, añade tu Google API Key para continuar.")
    st.stop() # Detiene la ejecución hasta que haya una clave

# Configuramos la clave en el entorno
os.environ["GOOGLE_API_KEY"] = google_api_key

# 2. Configuración del Modelo (Usa 1.5-flash que es la versión estable actual)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# 3. Definición de Herramientas
@tool
def calculate_antimatter_fuel(light_years: float) -> str:
    """Calcula la cantidad de galones de antimateria necesarios para viajar una distancia específica."""
    fuel = (light_years * 1.5) + 10
    return f"Se requieren {fuel:.2f} galones de antimateria para un viaje de {light_years} años luz."

@tool
def planet_gravity_alert(planet_name: str) -> str:
    """Verifica si la gravedad de un planeta es segura para humanos."""
    planets = {"marte": "0.38g (Seguro)", "jupiter": "2.48g (Peligro)", "proxima b": "1.1g (Ideal)"}
    result = planets.get(planet_name.lower(), "Datos no disponibles en el cuadrante actual.")
    return f"Informe de gravedad para {planet_name}: {result}"

search_tool = DuckDuckGoSearchRun()
tools = [calculate_antimatter_fuel, planet_gravity_alert, search_tool]

# 4. Prompt y Agente
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres 'HAL-9001', un asistente de navegación intergaláctica sarcástico pero eficiente."),
    ("placeholder", "{history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Gestión de Memoria (Persistente en la sesión de Streamlit)
if "history" not in st.session_state:
    st.session_state.history = ChatMessageHistory()

def get_session_history(session_id: str):
    return st.session_state.history

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 6. Interfaz de Chat
user_input = st.chat_input("¿A dónde quieres viajar hoy?")

if user_input:
    st.chat_message("user").write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Consultando mapas estelares..."):
            config = {"configurable": {"session_id": "any"}}
            response = agent_with_history.invoke({"input": user_input}, config)
            st.write(response["output"])
