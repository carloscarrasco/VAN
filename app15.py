import streamlit as st
import os
import PyPDF2
import pdfplumber
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

# 🔑 Configurar API Key de OpenAI
os.environ["OPENAI_API_KEY"] = "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"

# 📌 Configuración de la Página Web
st.set_page_config(page_title="📖 Chat con la Ley de Pensiones", layout="wide")
st.title("📖 Chat con la Ley de Pensiones 📚🤖")

st.write("📂 **Sube un PDF y chatea con ChatGPT para extraer información específica, incluidas imágenes y tablas.**")

# 📂 Subir Archivo PDF
pdf_file = st.file_uploader("📂 Sube tu PDF", type=["pdf"])

if pdf_file:
    st.success("✅ PDF cargado correctamente")

    # 📜 Leer y Extraer Texto del PDF
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # 🔍 Extraer Tablas del PDF con pdfplumber
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                df = pd.DataFrame(table)
                tables.append(df)

    # 🔍 Crear Base de Conocimiento del PDF
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # 📚 Convertir Texto a Embeddings para GPT-4
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    vectorstore.save_local("faiss_index")  # Guardamos para consultas futuras

    st.success("📚 El PDF se ha convertido en una base de conocimiento.")

    # 🔍 Inicializar Historial de Conversación
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 🔍 Chat Interactivo
    st.subheader("💬 Chat con la Ley de Pensiones")
    user_question = st.text_input("Escribe tu pregunta aquí y presiona Enter:")

    if user_question:
        vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever()
        llm = OpenAI(temperature=0)

        # 🚀 🔥 **EXTRAER EL CONTEXTO RELEVANTE DEL PDF**
        relevant_docs = retriever.get_relevant_documents(user_question)
        pdf_context = "\n\n".join([doc.page_content for doc in relevant_docs])  # Fragmentos relevantes

        # 🔧 **Nuevo Prompt Mejorado**
        custom_prompt = f"""
        Eres un asistente experto en leyes de pensiones. 
        Solo responde con información contenida en el PDF cargado por el usuario.

        📄 **Información relevante del documento:**  
        {pdf_context}  

        Historial de la conversación:
        {st.session_state.chat_history}

        Pregunta del usuario: {user_question}

        ⚠️ **Si la respuesta debe incluir una tabla, devuélvela en formato CSV sin ningún otro texto adicional**.
        - La primera fila debe contener los nombres de las columnas.
        - Cada fila debe estar separada por una nueva línea.
        - Las columnas deben estar separadas por comas.

        ⚠️ **Si no hay tabla, responde en texto normal.**
        """

      # 📌 **Obtener Respuesta de GPT-4**
        response = llm.predict(custom_prompt).strip()

        # 💾 Guardar Conversación en Historial
        st.session_state.chat_history.append(f"👤 Usuario: {user_question}")
        st.session_state.chat_history.append(f"🤖 GPT-4: {response}")

        # 🔍 **Detectar si la respuesta es una tabla**
        if "," in response and "\n" in response and len(response.split("\n")) > 2:
            try:
                # 💡 **Corregir el formato de la tabla**
                lines = response.split("\n")
                clean_lines = [line.strip() for line in lines if line.strip()]  # Eliminar líneas vacías

                # Verificar si la primera fila tiene encabezados
                headers = clean_lines[0].split(",")  # Primera fila como encabezado
                data = [row.split(",") for row in clean_lines[1:]]  # Resto de las filas

                # 📊 Convertir a DataFrame
                df_table = pd.DataFrame(data, columns=headers)

                st.subheader("📊 Tabla Extraída:")
                st.dataframe(df_table)  # Mostrar tabla en la web

                # 📥 Descargar la tabla en Excel
                st.download_button(
                    label="📥 Descargar Tabla en Excel",
                    data=df_table.to_csv(index=False).encode("utf-8"),
                    file_name="tabla_pensiones.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.warning("⚠️ No se pudo procesar la tabla correctamente. Mostrando respuesta en texto.")
                st.write(response)

        else:
            # 📜 Si la respuesta es solo texto, mostrarla sin errores
            st.subheader("📜 Respuesta de ChatGPT:")
            st.write(response)

            # 📥 Convertir Respuesta en Excel para Descargar
            df_text = pd.DataFrame({"Pregunta": [user_question], "Respuesta": [response]})
            st.download_button(
                label="📥 Descargar Respuesta en Excel",
                data=df_text.to_csv(index=False).encode("utf-8"),
                file_name="respuesta_pensiones.csv",
                mime="text/csv"
            )
        # 📜 Mostrar Historial de Conversación
    
    st.subheader("📖 Historial de Conversación")
    for msg in st.session_state.chat_history:
        st.write(msg)

    # 📥 Descargar Historial Completo en Excel
    if st.session_state.chat_history:
        df_chat = pd.DataFrame({"Conversación": st.session_state.chat_history})
        st.download_button(
            label="📥 Descargar Historial Completo en Excel",
            data=df_chat.to_csv(index=False).encode("utf-8"),
            file_name="historial_chat_pensiones.csv",
            mime="text/csv"
        )
