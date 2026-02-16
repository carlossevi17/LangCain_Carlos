import os
import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Configuración de la API Key (Pide la clave si no existe)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Introduce tu Google API Key: ")

# 2. Configuración del Modelo (Corregido a versión estable)
# El error de Pydantic suele ser por nombres de modelo incorrectos
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# 3. Definición de Herramientas (Astro-Logistics)
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
    ("system", "Eres 'HAL-9001', un asistente de navegación intergaláctica. Eres eficiente y un poco irónico."),
    ("placeholder", "{history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Memoria
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- Prueba de ejecución ---
config = {"configurable": {"session_id": "vuelo_01"}}
print("\n--- Sistema Iniciado ---\n")
pregunta = "Hola, necesito ir a Marte, ¿cuánto combustible gasto si está a 0.00002 años luz?"
respuesta = agent_with_history.invoke({"input": pregunta}, config)
