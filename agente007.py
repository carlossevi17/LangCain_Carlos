import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import DuckDuckGoSearchResults

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Agente con Memoria", page_icon="🤖")
st.title("🤖 Mi Agente con Memoria")

# Input para la API Key (más seguro para despliegue)
api_key = st.sidebar.text_input("Introduce tu Google API Key", type="password")

if api_key:
    # 1. Configurar el Modelo
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    search = DuckDuckGoSearchResults()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools = [search, wikipedia]

    # 3. Configurar la Memoria (Específica para Streamlit)
    # Esto guarda los mensajes automáticamente en st.session_state
    msgs = StreamlitChatMessageHistory(key="chat_history")

    # 4. Definir el Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil que usa herramientas para dar respuestas precisas."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 5. Crear el Agente
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 6. Agente con historial
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="history",
    )

    # --- INTERFAZ DE CHAT ---
    # Mostrar historial previo
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Entrada del usuario
    if user_input := st.chat_input("¿En qué puedo ayudarte?"):
        st.chat_message("human").write(user_input)
        
        with st.chat_message("ai"):
            with st.spinner("Pensando..."):
                # Llamada al agente
                config = {"configurable": {"session_id": "any"}}
                response = agent_with_history.invoke({"input": user_input}, config)
                st.write(response["output"])

else:
    st.info("Por favor, añade tu Google API Key en la barra lateral para comenzar.")
