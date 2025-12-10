import streamlit as st
import pandas as pd
import os
from langchain_groq import ChatGroq
from herramientas import crear_herramientas

# --------------------------------------------
# CONFIGURACIÓN DE LA APP
# --------------------------------------------
st.set_page_config(page_title="Asistente de Análisis de Datos con IA", layout="centered")
st.title("🦜 Asistente de Análisis de Datos con IA")

st.info("""
Esta herramienta permite generar reportes, responder preguntas sobre los datos 
y crear gráficos usando un DataFrame cargado desde un archivo CSV.
""")

# --------------------------------------------
# SUBIR ARCHIVO
# --------------------------------------------
st.markdown("### 📁 Cargar archivo CSV")
archivo_cargado = st.file_uploader("Selecciona un archivo CSV", type="csv", label_visibility="collapsed")

if archivo_cargado:
    df = pd.read_csv(archivo_cargado)
    st.success("Archivo cargado exitosamente!")
    st.dataframe(df.head())

    # --------------------------------------------
    # LLM (Modelo económico)
    # --------------------------------------------
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # --------------------------------------------
    # CARGAR HERRAMIENTAS
    # --------------------------------------------
    tools = crear_herramientas(df)
    tool_dict = {t.name: t for t in tools}

    # --------------------------------------------
    # FUNCIÓN REDUCIDA DE INVOCACIÓN (SIN REACT)
    # --------------------------------------------
    def preguntar_llm(mensaje):
        respuesta = llm.invoke(mensaje)
        return respuesta.content

    # --------------------------------------------
    # 1️⃣ INFORME GENERAL
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 📄 Informe General del Dataset")

    if st.button("Generar Informe General"):
        with st.spinner("Generando informe general…"):
            tool = tool_dict.get("Informaciones DF")
            if tool:
                informe_general = tool.run({
                    "pregunta": "Dame un reporte general del DataFrame",
                    "df": df
                })
                st.markdown(informe_general)
                st.download_button(
                    "📥 Descargar Informe General",
                    informe_general,
                    "informe_general.md"
                )

    # --------------------------------------------
    # 2️⃣ INFORME ESTADÍSTICO
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 Informe Estadístico")

    if st.button("Generar Informe Estadístico"):
        with st.spinner("Generando informe estadístico…"):
            tool = tool_dict.get("Resumen Estadístico")
            if tool:
                informe_estadistico = tool.run({
                    "pregunta": "Genera un resumen estadístico del DataFrame",
                    "df": df
                })
                st.markdown(informe_estadistico)
                st.download_button(
                    "📥 Descargar Informe Estadístico",
                    informe_estadistico,
                    "informe_estadistico.md"
                )

    # --------------------------------------------
    # 3️⃣ GENERAR GRÁFICO
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 Crear gráfico")

    pregunta_grafico = st.text_input("Describe el gráfico que deseas generar:")
    if st.button("Generar gráfico"):
        if pregunta_grafico.strip() == "":
            st.warning("Por favor, describe el gráfico que deseas generar.")
        else:
            with st.spinner("Generando gráfico…"):
                tool_grafico = tool_dict.get("Generar Gráfico")
                if tool_grafico:
                    tool_grafico.run({
                        "pregunta": pregunta_grafico,
                        "df": df
                    })

    # --------------------------------------------
    # 4️⃣ INFORME DE INSIGHTS
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## ✨ Informe de Insights del Dataset")

    if st.button("Generar Informe de Insights"):
        with st.spinner("Generando informe de insights…"):
            tool_insights = tool_dict.get("Informe de Insights")
            if tool_insights:
                insights = tool_insights.run({
                    "pregunta": "Genera un informe con los principales insights del dataset",
                    "df": df
                })
                st.markdown(insights)
                st.download_button(
                    "📥 Descargar Informe de Insights",
                    insights,
                    "informe_insights.md"
                )

    # --------------------------------------------
    # PREGUNTA DIRECTA AL LLM
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 🔎 Preguntas directas sobre los datos")

    pregunta = st.text_input("Escribe tu pregunta:")
    if st.button("Responder pregunta"):
        if pregunta.strip() == "":
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.spinner("Analizando datos…"):
                respuesta = preguntar_llm(
                    f"""Eres un analista experto. 
                    Responde la siguiente pregunta usando este DataFrame:
                    Columnas: {list(df.columns)}
                    Pregunta: {pregunta}"""
                )
                st.markdown(respuesta)