import streamlit as st
import os
from datetime import datetime
# Forzamos la configuración de LangChain antes de importar lo demás
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

# Contexto temporal real
fecha_actual = datetime.now().strftime("%d de %B de %Y")

st.set_page_config(page_title="Agente 007 Fix", page_icon="🚀")

with st.sidebar:
    st.title("Configuración")
    user_api_key = st.text_input("Google API Key", type="password", key="api_key")

if user_api_key:
    # 1. Limpieza absoluta de la clave
    api_key_clean = user_api_key.strip()
    
    try:
        # 2. Configuración del modelo con transporte forzado y nombre estable
        # Si insistes en gemini-3 y tu SDK lo soporta, cámbialo, 
        # pero gemini-1.5-flash es el estándar que no falla.
        llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview", 
            google_api_key=api_key_clean,
            temperature=0,
            transport="rest"  # <--- ESTO arregla el error 'line 1 column 1' en muchas redes
        )

        # 3. Herramientas
        search = DuckDuckGoSearchResults()
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools = [search, wikipedia]

        # 4. Prompt idéntico a tu notebook
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Eres un asistente preciso. Hoy es {fecha_actual}. Si no sabes algo, usa Wikipedia."),
            ("placeholder", "{history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 5. Agente
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 6. Memoria
        msgs = StreamlitChatMessageHistory(key="chat_history")

        def ensure_string(res: dict) -> dict:
            if isinstance(res.get('output'), list):
                res['output'] = str(res['output'][0].get('text', ''))
            return res

        chain = agent_executor | RunnableLambda(ensure_string)

        agent_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: msgs,
            input_messages_key="input",
            history_messages_key="history",
        )

        # --- INTERFAZ ---
        st.write(f"📅 **Hoy es:** {fecha_actual}")
        
        for msg in msgs.messages:
            st.chat_message(msg.type).write(msg.content)

        if prompt_user := st.chat_input():
            st.chat_message("human").write(prompt_user)
            with st.chat_message("ai"):
                # Invocación limpia
                response = agent_with_history.invoke(
                    {"input": prompt_user}, 
                    config={"configurable": {"session_id": "default"}}
                )
                st.write(response["output"])

    except Exception as e:
        st.error(f"Fallo técnico: {e}")
else:
    st.info("Introduce la clave.")
