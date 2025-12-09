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
    # ACCIONES RÁPIDAS
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## ⚡ Acciones rápidas")

    # ---- Reporte General ----
    if st.button("📄 Reporte de Informaciones Generales"):
        with st.spinner("Generando reporte…"):
            tool = tool_dict["Informaciones DF"]
            resultado = tool.run({
                "pregunta": "Dame un reporte general del DataFrame",
                "df": df
            })
            st.session_state["reporte_general"] = resultado

    if "reporte_general" in st.session_state:
        with st.expander("Resultado: Reporte de Informaciones Generales"):
            st.markdown(st.session_state["reporte_general"])
            st.download_button(
                "📥 Descargar Reporte",
                st.session_state["reporte_general"],
                "reporte_informacion_general.md"
            )

    # ---- Reporte Estadístico ----
    if st.button("📄 Reporte de estadísticas descriptivas"):
        with st.spinner("Generando reporte…"):
            tool = tool_dict["Resumen Estadístico"]
            resultado = tool.run({
                "pregunta": "Genera un resumen estadístico del DataFrame",
                "df": df
            })
            st.session_state["reporte_estadisticas"] = resultado

    if "reporte_estadisticas" in st.session_state:
        with st.expander("Resultado: Reporte de estadísticas descriptivas"):
            st.markdown(st.session_state["reporte_estadisticas"])
            st.download_button(
                "📥 Descargar Reporte",
                st.session_state["reporte_estadisticas"],
                "reporte_estadisticas.md"
            )

    # --------------------------------------------
    # PREGUNTA SOBRE LOS DATOS
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 🔎 Preguntas sobre los datos")

    pregunta = st.text_input("Escribe tu pregunta:")
    if st.button("Responder pregunta"):
        with st.spinner("Analizando datos…"):
            respuesta = preguntar_llm(
                f"""Eres un analista experto. 
                Responde la siguiente pregunta usando este DataFrame:
                Columnas: {list(df.columns)}
                Pregunta: {pregunta}"""
            )
            st.markdown(respuesta)

    # --------------------------------------------
    # GENERACIÓN DE GRÁFICOS
    # --------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 Crear gráfico")

    pregunta_grafico = st.text_input("¿Qué gráfico deseas generar?")
    if st.button("Generar gráfico"):
        with st.spinner("Generando gráfico…"):
            tool = tool_dict["Generar Gráfico"]
            tool.run({
                "pregunta": pregunta_grafico,
                "df": df
            })