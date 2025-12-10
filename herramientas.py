import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from langchain_experimental.tools import PythonAstREPLTool

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------------------------------------------------
# INFORMACIÓN DEL DATAFRAME — VERSIÓN MEJORADA
# ---------------------------------------------------------
def informacion_df(pregunta: str, df: pd.DataFrame) -> str:

    n_filas, n_columnas = df.shape
    tipos = df.dtypes.astype(str)

    tabla_tipos = (
        pd.DataFrame({"Columna": tipos.index, "Tipo de dato": tipos.values})
        .to_markdown(index=False)
    )

    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df) * 100).round(2)

    tabla_nulos = (
        pd.DataFrame({
            "Columna": nulos.index,
            "Nulos": nulos.values,
            "% Nulos": porcentaje_nulos.values
        })
        .sort_values("% Nulos", ascending=False)
        .to_markdown(index=False)
    )

    duplicados = df.duplicated().sum()

    plantilla = PromptTemplate(
        template='''
        Eres un analista de datos senior. Con base en la siguiente información del dataset:

        - Pregunta del usuario: {pregunta}
        - Filas: {n_filas}
        - Columnas: {n_columnas}

        Genera un texto narrativo breve, claro y profesional que incluya:
        1. Una explicación general del dataset.
        2. Qué tipo de análisis pueden realizarse.
        3. Recomendaciones de preprocesamiento.
        4. Qué insights se pueden extraer.

        NO repitas tablas ni dimensiones — solo narrativa.
        ''',
        input_variables=["pregunta", "n_filas", "n_columnas"]
    )

    cadena = plantilla | llm | StrOutputParser()
    narrativa = cadena.invoke({
        "pregunta": pregunta,
        "n_filas": n_filas,
        "n_columnas": n_columnas
    })

    informe = f"""
# 📊 Informe General del Dataset

Este informe responde a la solicitud: **{pregunta}**  

---

## 🔹 Resumen General

| Métrica | Valor |
|--------|-------|
| **Total de filas** | {n_filas} |
| **Total de columnas** | {n_columnas} |
| **Filas duplicadas** | {duplicados} |

---

## 🔹 Tipos de columnas
{tabla_tipos}

---

## 🔹 Valores nulos por columna
{tabla_nulos}

---

## ✨ Análisis Narrativo
{narrativa}

---

## 🎯 Recomendación
Puedes continuar con análisis estadísticos, generación de gráficos o transformaciones usando las herramientas del asistente.
"""

    return informe


# ---------------------------------------------------------
# RESUMEN ESTADÍSTICO
# ---------------------------------------------------------
def resumen_estadistico(pregunta, df):

    resumen = df.describe(include='number').transpose().to_string()

    plantilla = PromptTemplate(
        template="""
        Eres un analista de datos encargado de interpretar estadísticas
        a partir de una pregunta: {pregunta}

        ================= ESTADÍSTICAS DESCRIPTIVAS =================
        {resumen}
        ============================================================

        Redacta un informe:
        - Título: "## Informe de estadísticas descriptivas"
        - Explica cada columna
        - Detecta outliers
        - Sugiere próximos pasos
        """,
        input_variables=['pregunta','resumen']
    )

    cadena = plantilla | llm | StrOutputParser()
    return cadena.invoke({
        "pregunta": pregunta,
        "resumen": resumen
    })


# ---------------------------------------------------------
# GRÁFICOS
# ---------------------------------------------------------
def generar_grafico(pregunta, df):
    columnas_info = '\n'.join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
    num_filas = len(df)

    plantilla = PromptTemplate(
        template="""
        Eres un experto en visualización. Devuelve SOLO código Python válido.

        ## Solicitud:
        "{pregunta}"

        ## DataFrame:
        Filas: {num_filas}
        Columnas:
        {columnas}

        ## Reglas:
        - Usar SIEMPRE el df completo (no head(), no sample(), no iloc).
        - Usar plt y sns.
        - sns.set_theme()
        - figsize=(10,5)
        - Título y etiquetas.
        - sns.despine()
        - plt.show()

        Código:
        """,
        input_variables=['pregunta','columnas','num_filas']
    )

    cadena = plantilla | llm | StrOutputParser()
    script = cadena.invoke({
        "pregunta": pregunta,
        "columnas": columnas_info,
        "num_filas": num_filas
    })

    script = script.replace("```python", "").replace("```", "")

    exec_globals = {"df": df, "plt": plt, "sns": sns}
    exec(script, exec_globals)

    fig = plt.gcf()
    st.pyplot(fig)

    return ""


# ---------------------------------------------------------
# HERRAMIENTA PYTHON INTELIGENTE (CORRELACIONES)
# ---------------------------------------------------------
def ejecutar_python_inteligente(pregunta: str, df=None):

    pregunta_lower = pregunta.lower()

    # Detectar si el usuario pregunta por correlaciones
    if "correl" in pregunta_lower or "relación" in pregunta_lower or "correlation" in pregunta_lower:

        # buscar columna objetivo tipo "tiempo"
        col_obj = None
        for c in df.columns:
            if "tiempo" in c.lower():
                col_obj = c
                break

        if col_obj is None:
            return "❌ No encontré ninguna columna relacionada con tiempo."

        corr = df.corr(numeric_only=True)[col_obj].sort_values(ascending=False)
        mejor = corr.drop(col_obj).idxmax()
        valor = corr.drop(col_obj).max()

        tabla = corr.to_frame("Correlación").to_markdown()

        return f"""
# 🔎 Análisis de correlación para **{col_obj}**

## 📘 Todas las correlaciones
{tabla}

---

## 🥇 Variable más correlacionada
**{mejor}** con **{valor:.4f}**
"""

    # Si no es correlación, ejecutar Python normalmente
    repl = PythonAstREPLTool(locals={"df": df})
    return repl.run(pregunta)


# ---------------------------------------------------------
# CREAR HERRAMIENTAS
# ---------------------------------------------------------
def crear_herramientas(df):

    herramienta_info = StructuredTool.from_function(
        name="Informaciones DF",
        func=lambda pregunta: informacion_df(pregunta, df),
        description="Devuelve información general del dataframe.",
        return_direct=True
    )

    herramienta_resumen = StructuredTool.from_function(
        name="Resumen Estadístico",
        func=lambda pregunta: resumen_estadistico(pregunta, df),
        description="Devuelve un análisis estadístico del dataframe.",
        return_direct=True
    )

    herramienta_grafico = StructuredTool.from_function(
        name="Generar Gráfico",
        func=lambda pregunta: generar_grafico(pregunta, df),
        description="Genera un gráfico en base a una pregunta del usuario.",
        return_direct=True
    )

    herramienta_python = StructuredTool.from_function(
        name="Herramienta Códigos de Python",
        func=lambda pregunta: ejecutar_python_inteligente(pregunta, df),
        description="Ejecuta cálculos en Python directamente sobre df.",
        return_direct=True
    )

    return [
        herramienta_info,
        herramienta_resumen,
        herramienta_grafico,
        herramienta_python
    ]