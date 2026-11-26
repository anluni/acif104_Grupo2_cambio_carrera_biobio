"""Backend: API REST para la alerta temprana de cambio de carrera."""
import csv
import os
import time
from datetime import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy import sparse
from xgboost import XGBClassifier

RUTA_MODELO = os.getenv("RUTA_MODELO", "modelo_cambio_carrera.joblib")
RUTA_XGB = os.getenv("RUTA_XGB", "modelo_xgboost.json")
RUTA_LOG = os.getenv("RUTA_LOG", "registro_predicciones.csv")

# ---------- Carga del modelo (una sola vez, al iniciar) ----------
art = joblib.load(RUTA_MODELO)
prep = art["preprocesador"]
UMBRAL = art["umbral"]
COLUMNAS = art["columnas"]
GRUPOS = np.array(art["grupos"])
FECHAS = art["fechas"]
TASA_ESPERADA = art["tasa_alertas_esperada"]

clf = XGBClassifier()
clf.load_model(RUTA_XGB)
booster = clf.get_booster()

CATALOGO = art["catalogo"].copy()
CATALOGO["codigo_unico"] = CATALOGO["codigo_unico"].astype(str)
CATALOGO = CATALOGO.set_index("codigo_unico")

app = FastAPI(
    title="API de alerta temprana de cambio de carrera",
    description="Predice el riesgo de cambio de carrera (Biobío, MINEDUC 2025) y explica cada predicción con SHAP.",
    version="1.0",
)


class Estudiante(BaseModel):
    codigo_unico: str
    gen_alu: str
    fec_nac_alu: str


# ---------- Funciones internas ----------
def armar_fila(e: Estudiante) -> pd.DataFrame:
    """Construye el registro completo que espera el modelo."""
    if e.codigo_unico not in CATALOGO.index:
        raise HTTPException(404, f"Programa '{e.codigo_unico}' no encontrado en el catálogo.")
    if e.gen_alu not in ("Masculino", "Femenino"):
        raise HTTPException(422, "gen_alu debe ser 'Masculino' o 'Femenino'.")
    if e.fec_nac_alu not in FECHAS:
        raise HTTPException(422, "Fecha de nacimiento fuera del rango conocido por el modelo.")

    fila = CATALOGO.loc[[e.codigo_unico]].reset_index()
    fila["gen_alu"] = e.gen_alu
    fila["fec_nac_alu"] = FECHAS[e.fec_nac_alu]["valor"]
    fila["rango_edad"] = FECHAS[e.fec_nac_alu]["rango_edad"]
    return fila[COLUMNAS]


def transformar(fila: pd.DataFrame):
    """Aplica el preprocesamiento y devuelve una matriz dispersa."""
    Xs = prep.transform(fila)
    return Xs if sparse.issparse(Xs) else sparse.csr_matrix(Xs)


def explicar(Xs, top: int = 5) -> list:
    """Valores SHAP exactos (TreeSHAP), agregados por variable original."""
    contrib = booster.predict(xgb.DMatrix(Xs), pred_contribs=True)[0, :-1]
    por_variable = pd.Series(contrib).groupby(GRUPOS).sum()
    principales = por_variable.abs().sort_values(ascending=False).index[:top]
    return [
        {
            "variable": v,
            "contribucion": round(float(por_variable[v]), 4),
            "efecto": "aumenta el riesgo" if por_variable[v] > 0 else "disminuye el riesgo",
        }
        for v in principales
    ]


def registrar(e: Estudiante, puntaje: float, alerta: bool, ms: float) -> None:
    """Guarda cada predicción en un CSV para el monitoreo."""
    nuevo = not os.path.exists(RUTA_LOG)
    with open(RUTA_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha_hora", "codigo_unico", "gen_alu", "fec_nac_alu",
                        "puntaje", "alerta", "latencia_ms"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), e.codigo_unico,
                    e.gen_alu, e.fec_nac_alu, round(puntaje, 4), int(alerta), round(ms, 1)])


# ---------- Endpoints ----------
@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo": art["nombre"], "umbral": UMBRAL,
            "programas": len(CATALOGO)}


@app.get("/programas")
def programas():
    cols = ["nomb_inst", "nomb_sede", "nomb_carrera", "jornada", "modalidad"]
    return CATALOGO[cols].fillna("").reset_index().to_dict(orient="records")


@app.get("/fechas")
def fechas():
    return sorted(FECHAS.keys())


@app.post("/predecir")
def predecir(e: Estudiante):
    inicio = time.perf_counter()
    fila = armar_fila(e)
    Xs = transformar(fila)
    puntaje = float(clf.predict_proba(Xs)[0, 1])
    alerta = puntaje >= UMBRAL
    factores = explicar(Xs)
    ms = (time.perf_counter() - inicio) * 1000
    registrar(e, puntaje, alerta, ms)
    return {
        "puntaje_riesgo": round(puntaje, 4),
        "umbral": UMBRAL,
        "alerta": bool(alerta),
        "mensaje": ("Riesgo alto: se recomienda seguimiento preventivo." if alerta
                    else "Riesgo bajo: no se requiere acción inmediata."),
        "rango_edad": str(fila["rango_edad"].iloc[0]),
        "factores": factores,
        "latencia_ms": round(ms, 1),
    }


@app.post("/predecir_lote")
def predecir_lote(estudiantes: List[Estudiante]):
    return [predecir(e) for e in estudiantes]


@app.get("/monitoreo")
def monitoreo():
    if not os.path.exists(RUTA_LOG):
        return {"total": 0, "tasa_alertas_esperada": round(TASA_ESPERADA, 4)}
    log = pd.read_csv(RUTA_LOG)
    tasa = float(log["alerta"].mean())
    return {
        "total": int(len(log)),
        "alertas": int(log["alerta"].sum()),
        "tasa_alertas": round(tasa, 4),
        "tasa_alertas_esperada": round(TASA_ESPERADA, 4),
        "puntaje_promedio": round(float(log["puntaje"].mean()), 4),
        "latencia_promedio_ms": round(float(log["latencia_ms"].mean()), 1),
        "posible_desviacion": bool(len(log) >= 30 and abs(tasa - TASA_ESPERADA) > 0.10),
        "ultimas": log.tail(20).astype(str).to_dict(orient="records"),
    }
