"""Frontend: interfaz web de alerta temprana de cambio de carrera."""
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Alerta temprana de cambio de carrera",
                   page_icon="🎓", layout="wide")


@st.cache_data
def obtener(ruta):
    r = requests.get(f"{API}{ruta}", timeout=30)
    r.raise_for_status()
    return r.json()


try:
    salud = obtener("/salud")
except Exception:
    st.error(f"No se pudo conectar con el backend en {API}. "
             "Inicie primero la API con: uvicorn api:app")
    st.stop()

st.title("🎓 Alerta temprana de cambio de carrera")
st.caption(f"Región del Biobío · Modelo: {salud['modelo']} · Umbral de alerta: {salud['umbral']}")

pestana_eval, pestana_mon = st.tabs(["Evaluar estudiante", "Monitoreo"])

# ---------------- Pestaña 1: evaluación ----------------
with pestana_eval:
    programas = pd.DataFrame(obtener("/programas"))
    fechas = obtener("/fechas")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Programa de estudios")
        inst = st.selectbox("Institución", sorted(programas["nomb_inst"].unique()))
        p = programas[programas["nomb_inst"] == inst]
        sede = st.selectbox("Sede", sorted(p["nomb_sede"].unique()))
        p = p[p["nomb_sede"] == sede]
        carrera = st.selectbox("Carrera", sorted(p["nomb_carrera"].unique()))
        p = p[p["nomb_carrera"] == carrera].copy()
        p["opcion"] = p["jornada"] + " · " + p["modalidad"] + " · código " + p["codigo_unico"]
        opcion = st.selectbox("Jornada y modalidad", p["opcion"])
        codigo = p.loc[p["opcion"] == opcion, "codigo_unico"].iloc[0]

    with col2:
        st.subheader("2. Datos del estudiante")
        genero = st.radio("Género", ["Femenino", "Masculino"], horizontal=True)
        fecha = st.selectbox("Fecha de nacimiento (año y mes)", fechas,
                             index=len(fechas) // 2)

    if st.button("Evaluar riesgo", type="primary"):
        r = requests.post(f"{API}/predecir", timeout=30, json={
            "codigo_unico": codigo, "gen_alu": genero, "fec_nac_alu": fecha})

        if r.status_code != 200:
            st.error(r.json().get("detail", "Error en la predicción."))
        else:
            res = r.json()
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Puntaje de riesgo", f"{res['puntaje_riesgo']:.3f}")
            m2.metric("Umbral de alerta", f"{res['umbral']:.2f}")
            m3.metric("Rango de edad", res["rango_edad"])

            if res["alerta"]:
                st.error(f"🔴 {res['mensaje']}")
            else:
                st.success(f"🟢 {res['mensaje']}")

            st.subheader("¿Por qué el modelo entrega este resultado?")
            factores = pd.DataFrame(res["factores"])
            st.bar_chart(factores.set_index("variable")["contribucion"])
            st.dataframe(factores.rename(columns={
                "variable": "Variable", "contribucion": "Contribución SHAP",
                "efecto": "Efecto"}), hide_index=True, use_container_width=True)
            st.caption("Valores positivos aumentan el riesgo y valores negativos lo disminuyen. "
                       "El puntaje es un indicador de riesgo relativo, no una probabilidad calibrada.")

# ---------------- Pestaña 2: monitoreo ----------------
with pestana_mon:
    if st.button("Actualizar monitoreo"):
        st.rerun()
    mon = requests.get(f"{API}/monitoreo", timeout=30).json()

    if mon["total"] == 0:
        st.info("Aún no se han realizado predicciones.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Predicciones", mon["total"])
        b.metric("Alertas emitidas", mon["alertas"])
        c.metric("Tasa de alertas", f"{mon['tasa_alertas']:.1%}",
                 f"esperada {mon['tasa_alertas_esperada']:.1%}", delta_color="off")
        d.metric("Tiempo de respuesta", f"{mon['latencia_promedio_ms']} ms")

        if mon["posible_desviacion"]:
            st.warning("⚠️ La tasa de alertas difiere en más de 10 puntos de la esperada. "
                       "Podría existir un cambio en los datos (drift); se recomienda revisar el modelo.")

        ultimas = pd.DataFrame(mon["ultimas"])
        ultimas["puntaje"] = ultimas["puntaje"].astype(float)
        st.subheader("Puntajes de las últimas predicciones")
        st.line_chart(ultimas["puntaje"])
        st.dataframe(ultimas, hide_index=True, use_container_width=True)
