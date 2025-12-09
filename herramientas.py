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

# -------------------------------------------------------------------
# HERRAMIENTA: INFORMACIÓN DEL DATAFRAME
# -------------------------------------------------------------------
def informacion_df(pregunta, df):
    shape = df.shape
    columns = df.dtypes
    nulos = df.isnull().sum()
    nans_str = df.apply(lambda col: col[~col.isna()].astype(str).str.strip().str.lower().eq('nan').sum())
    duplicados = df.duplicated().sum()

    plantilla = PromptTemplate(
        template="""
        Eres un analista de datos encargado de presentar un resumen informativo
        sobre un **DataFrame** a partir de una pregunta: {pregunta}

        ================= INFORMACIÓN DEL DATAFRAME =================

        Dimensiones: {shape}

        Columnas y tipos de datos:
        {columns}

        Valores nulos por columna:
        {nulos}

        Cadenas 'nan' por columna:
        {nans_str}

        Filas duplicadas: {duplicados}

        ============================================================

        Ahora redacta un informe:
        - Título: "## Informe de información general sobre el dataset"
        - Explica cada punto de forma clara
        - Recomienda análisis posibles
        - Recomienda tratamientos de datos
        """,
        input_variables=['pregunta','shape','columns','nulos','nans_str','duplicados']
    )

    cadena = plantilla | llm | StrOutputParser()
    return cadena.invoke({
        "pregunta": pregunta,
        "shape": shape,
        "columns": columns,
        "nulos": nulos,
        "nans_str": nans_str,
        "duplicados": duplicados
    })

# -------------------------------------------------------------------
# HERRAMIENTA: RESUMEN ESTADÍSTICO
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# HERRAMIENTA: GENERACIÓN DE GRÁFICOS (SIN @tool)
# -------------------------------------------------------------------
def generar_grafico(pregunta, df):
    """
    Genera un gráfico usando SIEMPRE el DataFrame completo.
    Devuelve string vacío porque Streamlit muestra el gráfico.
    """

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

# -------------------------------------------------------------------
# CREAR HERRAMIENTAS
# -------------------------------------------------------------------
def crear_herramientas(df):

    herramienta_info = StructuredTool.from_function(
        name="Informaciones DF",
        func=informacion_df,
        description="Devuelve información general del dataframe.",
        return_direct=True
    )

    herramienta_resumen = StructuredTool.from_function(
        name="Resumen Estadístico",
        func=resumen_estadistico,
        description="Devuelve un análisis estadístico del dataframe.",
        return_direct=True
    )

    herramienta_grafico = StructuredTool.from_function(
        name="Generar Gráfico",
        func=generar_grafico,
        description="Genera un gráfico en base a una pregunta del usuario.",
        return_direct=True
    )

    herramienta_python = PythonAstREPLTool(locals={"df": df})
    herramienta_python.name = "Herramienta Códigos de Python"

    return [
        herramienta_info,
        herramienta_resumen,
        herramienta_grafico,
        herramienta_python
    ]