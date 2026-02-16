import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# Configuración visual de la página
st.set_page_config(page_title="Mi Agente LangChain", page_icon="🤖")
st.title("🤖 Mi Agente LangChain (Modo Estable)")

# 1. Entrada de la API Key en la barra lateral
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu Google API Key", type="password")
    st.info("Obtén tu clave en: https://aistudio.google.com/app/apikey")

if not api_key:
    st.warning("Por favor, introduce tu API Key en la barra lateral para comenzar.")
    st.stop()

# Establecer la API Key en el entorno
os.environ["GOOGLE_API_KEY"] = api_key

# 2. Definición de Herramientas (Basado en tu notebook)
@tool
def conchita_coins(usd_input: float) -> float:
    """Usa esta herramienta para convertir dólares (USD) a Conchita Academy coins."""
    return 1.3 * float(usd_input)

# Herramienta de Wikipedia (Más estable que DuckDuckGo en la nube)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [conchita_coins, wikipedia]

# 3. Configuración del Modelo con parámetros de seguridad (Para evitar el ClientError)
try:
    chat = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        safety_settings={
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
    )

    # 4. Prompt (Estructura de tu notebook)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente que responde dudas usando Wikipedia o convierte moneda con tus herramientas."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # 5. Creación del Agente
    agent = create_tool_calling_agent(chat, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 6. Interfaz de Usuario (Chat)
    user_input = st.text_input("¿Qué quieres saber o convertir?")

    if user_input:
        with st.spinner("El agente está procesando tu solicitud..."):
            try:
                # Ejecutamos el agente (invoke)
                result = agent_executor.invoke({"input": user_input})
                st.markdown(f"### Respuesta:\n{result['output']}")
            except Exception as e:
                st.error(f"Hubo un problema al procesar la respuesta: {e}")
                st.info("Asegúrate de que tu API Key sea válida y no tenga restricciones.")

except Exception as init_error:
    st.error(f"Error al inicializar el modelo: {init_error}")
