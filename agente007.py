import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Configuración del Modelo
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 2. Definición de Herramientas (Tools)
@tool
def calculate_antimatter_fuel(light_years: float) -> str:
    """Calcula la cantidad de galones de antimateria necesarios para viajar una distancia específica."""
    # Fórmula inventada: 1.5 galones por año luz + 10 galones de reserva para el salto inicial
    fuel = (light_years * 1.5) + 10
    return f"Se requieren {fuel:.2f} galones de antimateria para un viaje de {light_years} años luz."

@tool
def planet_gravity_alert(planet_name: str) -> str:
    """Verifica si la gravedad de un planeta es segura para humanos (Simulado)."""
    # Aquí podrías conectar una API real, pero para el ejemplo lo haremos temático
    planets = {"marte": "0.38g (Seguro)", "jupiter": "2.48g (Peligro: Aplastamiento)", "proxima b": "1.1g (Ideal)"}
    result = planets.get(planet_name.lower(), "Datos no disponibles en la base de datos galáctica.")
    return f"Informe de gravedad para {planet_name}: {result}"

search_tool = DuckDuckGoSearchRun()
tools = [calculate_antimatter_fuel, planet_gravity_alert, search_tool]

# 3. Prompt con Memoria y Personalidad
prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres 'HAL-9001', un asistente de navegación intergaláctica sarcástico pero eficiente. "
               "Utilizas tus herramientas para ayudar a los viajeros espaciales. "
               "Si te preguntan por noticias espaciales, usa la búsqueda."),
    ("placeholder", "{history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 4. Creación del Agente
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Gestión de Memoria (Store)
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

# --- Ejemplo de ejecución ---
# config = {"configurable": {"session_id": "piloto_1"}}
# agent_with_history.invoke({"input": "Hola, planeo ir a Proxima B. ¿Cuánto combustible necesito si está a 4.2 años luz?"}, config)
