# Diplamado en Python y Ciencia de Datos
# Instituto Politécnico Nacional
# Centro de Investigación en Computación
# Departamento de Diplomados y Extensión Profesional

# Ciudad de México
# Sábado 29, Agosto 2026.

# Alan Badillo Salas

# Módulo IV - Deep Learning
# Introducción a las Redes Neuronales de una Capa

# Práctica 1 - Perceptrón Lineal y Logístico

# > Espacio de características
#   - Análisis de una característica
#     * Categórica
#     * Numérica
#   - Contraste entre características
#     * Categórica / Categórica (A / B) - Matriz de conteos
#     * Categórica / Numérica (A / Y) - Matriz de estadísticos
#     * Numérica / Categórica (X / B) - Matriz de densidades
#     * Numérica / Numérica (X / Y) - Matriz de correlación
# > Espacio numérico
#   - Expansión por dummies(one-hot)
#     * Remoción de la clase base
#     * Matriz de características
#   - Transformación directa
#     * Ponderación natural
#     * Chances promedios
#     * Mediana
#     * Promedio y desviación estándar
# > Espacio de análisis (X, Y)
#   - Separación en características predictivas y respuestas
#    * Corte del conjunto de entrenamiento y validación
#   - Análisis de espacios separados
#    * Eje de observación
#    * Densidades de respuesta
#   - Análisis del espacio contiguo
#    * Dispersión
#    * Contornos de superficie
# > Espacio lineal (z)
#   - Transformación lineal ponderada (predictor)
#    * Ecuaciones normales
#    * Linealidad
#    * Centralidad
# > Perceptrón lineal (z)
#   - Optimización de las betas
#    * Modelo lineal
#    * Error y pérdida
#    * Algoritmo de optimización por descenso del gradiente
# > Perceptrón logístico (sigma(z))
#   - Optimización de las betas
#    * Modelo logístico
#    * Error y pérdida
#    * Algoritmo de optimización por descenso del gradiente

# -------------------------------------------------------

import numpy
import pandas

from matplotlib import pyplot
import seaborn

# -------------------------------------------------------

# Problema - Estrés en las personas
datos1 = pandas.DataFrame([
    [25, "Hombre", "Desempleado", 1.83, "POCO", 2, 0.3], # Observación 1
    [13, "Mujer", "Estudiante", 1.35, "NADA", 1, 0], # Observación 2
    [28, "Mujer", "Empleado", 1.56, "POCO", 2.5, .25], # Observación 3
    [55, "Hombre", "Empleado", 1.72, "MUCHO", 5, 0.8], # Observación 4
    [43, "Mujer", "Empleado", 1.60, "MUCHO", 3, 0.6], # Observación 5
    [18, "Hombre", "Desempleado", 1.76, "POCO", 1.5, 0.2], # Observación 6
], 
    columns=[
        "EDAD", "SEXO", "OCUPACION", 
        "ESTATURA", "NIVEL_ESTRES", 
        "FACTOR_ESTRES", "PORCENTAJE_ESTRES"
    ]
)

print(datos1)

# EDAD - Numérica
# SEXO - Categórica
# OCUPACION - Categórica
# ESTATURA - Numérica
# NIVEL_ESTRES - Categórica
# FACTOR_ESTRES - Numérica
# PORCENTAJE_ESTRES - Numérica

# Análisis de una característica

# Categóricas

# SEXO - Categórica
# OCUPACION - Categórica
# NIVEL_ESTRES - Categórica

print(
    datos1[["SEXO"]].groupby("SEXO").size()
)

# SEXO
# Hombre    3
# Mujer     3

print(
    (datos1[["SEXO"]].groupby("SEXO").size() / len(datos1)).round(3)
)

# SEXO
# Hombre    0.5
# Mujer     0.5

print(
    datos1[["OCUPACION"]].groupby("OCUPACION").size()
)
print(
    (datos1[["OCUPACION"]].groupby("OCUPACION").size() / len(datos1)).round(3)
)

# OCUPACION
# Desempleado    2
# Empleado       3
# Estudiante     1
# --------------------
# OCUPACION
# Desempleado    0.333
# Empleado       0.500
# Estudiante     0.167

print(
    datos1[["NIVEL_ESTRES"]].groupby("NIVEL_ESTRES").size()
)
print(
    (datos1[["NIVEL_ESTRES"]].groupby("NIVEL_ESTRES").size() / len(datos1)).round(3)
)

# NIVEL_ESTRES
# MUCHO    2
# NADA     1
# POCO     3
# --------------
# NIVEL_ESTRES
# MUCHO    0.333
# NADA     0.167
# POCO     0.500

figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
datos1[["NIVEL_ESTRES"]].groupby("NIVEL_ESTRES").size().sort_values().plot.bar(ax=axes)
pyplot.title("Frecuencia del nivel de estrés")
pyplot.xlabel("Nivel de estrés")
pyplot.ylabel("Frecuencia")

figure.savefig("nivel_estres.png", dpi=300)

# Numéricas

# EDAD - Numérica
# ESTATURA - Numérica
# FACTOR_ESTRES - Numérica
# PORCENTAJE_ESTRES - Numérica

# n, mín, mean, desv.est., Q1-0.25, Q2-0.5 (mediana), Q3-0.75, máx
print(
    datos1[["EDAD", "ESTATURA", "FACTOR_ESTRES", "PORCENTAJE_ESTRES"]].describe().round(3)
)

#          EDAD  ESTATURA  FACTOR_ESTRES  PORCENTAJE_ESTRES
# count   6.000     6.000          6.000              6.000
# mean   30.333     1.637          2.500              0.358
# std    15.845     0.173          1.414              0.291
# min    13.000     1.350          1.000              0.000
# 25%    19.750     1.570          1.625              0.213
# 50%    26.500     1.660          2.250              0.275
# 75%    39.250     1.750          2.875              0.525
# max    55.000     1.830          5.000              0.800

figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
seaborn.boxplot(datos1[["EDAD"]])
pyplot.title("Distribución de la edad")
pyplot.xlabel("Edad")
pyplot.ylabel("Distribución")

figure.savefig("edad.png", dpi=300)

#     * Categórica / Categórica (A / B) - Matriz de conteos

# SEXO / NIVEL_ESTRES

print(
    datos1[["SEXO", "NIVEL_ESTRES"]].groupby(["SEXO", "NIVEL_ESTRES"]).size().unstack()
)

# NIVEL_ESTRES  MUCHO  NADA  POCO
# SEXO                           
# Hombre          1.0   NaN   2.0
# Mujer           1.0   1.0   1.0

figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
seaborn.heatmap(
    datos1[["SEXO", "NIVEL_ESTRES"]].groupby(["SEXO", "NIVEL_ESTRES"]).size().unstack(),
    cmap="coolwarm"
)
pyplot.title("Sexo contra nivel de estrés")
pyplot.xlabel("Sexo")
pyplot.ylabel("Nivel de estrés")

figure.savefig("sexo_vs_nivel_estres.png", dpi=300)

#     * Categórica / Numérica (A / Y) - Matriz de estadísticos

figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
seaborn.boxplot(
    datos1[["OCUPACION", "PORCENTAJE_ESTRES"]], 
    x="OCUPACION", 
    y="PORCENTAJE_ESTRES", 
    hue="OCUPACION"
)
pyplot.title("Ocupación vs Porcentaje de estrés")
pyplot.xlabel("Ocupación")
pyplot.ylabel("Porcentaje de estrés")

figure.savefig("ocupacion_vs_porcentaje_estres.png", dpi=300)

#     * Numérica / Categórica (X / B) - Matriz de densidades
#     * Numérica / Numérica (X / Y) - Matriz de correlación

# Espacio numérico

# EDAD - Numérica
# ESTATURA - Numérica
# FACTOR_ESTRES - Numérica
# PORCENTAJE_ESTRES - Numérica

c1 = datos1["EDAD"] # 25, 13, 28, 55, ...
c2 = datos1["ESTATURA"] # 1.83, 1.35, 1.56, 1.72, ...
c3 = datos1["FACTOR_ESTRES"] # 2, 1, 2.5, 5, ...
c4 = datos1["PORCENTAJE_ESTRES"] # 0.3, 0, 0.25, 0.8, ...

# * Transformación de características categóricas

# SEXO - Categórica
# OCUPACION - Categórica
# NIVEL_ESTRES - Categórica

# - Expansión por dummies (one-hot)

c5_hombre = (datos1["SEXO"] == "Hombre").astype(int)
c5_mujer = (datos1["SEXO"] == "Mujer").astype(int) # BASE

c6_estudiante = (datos1["OCUPACION"] == "Estudiante").astype(int)
c6_desempleado = (datos1["OCUPACION"] == "Desempleado").astype(int)
c6_empleado = (datos1["OCUPACION"] == "Empleado").astype(int) # BASE

# - Transformación directa

# Ponderación natural

C7_natural = datos1["NIVEL_ESTRES"].map({
    "NADA": 0,
    "POCO": 1,
    "MUCHO": 2,
})

# Chances/oportunidades promedio

promedio_nada = datos1[datos1["NIVEL_ESTRES"] == "NADA"]["PORCENTAJE_ESTRES"].mean()
promedio_poco = datos1[datos1["NIVEL_ESTRES"] == "POCO"]["PORCENTAJE_ESTRES"].mean()
promedio_mucho = datos1[datos1["NIVEL_ESTRES"] == "MUCHO"]["PORCENTAJE_ESTRES"].mean()

C7_chances = datos1["NIVEL_ESTRES"].map({
    "NADA": (promedio_nada + 1) / (promedio_nada + 1),
    "POCO": (promedio_poco + 1) / (promedio_nada + 1),
    "MUCHO": (promedio_mucho + 1) / (promedio_nada + 1),
})

# Mediana

C7_mediana = datos1["NIVEL_ESTRES"].map({
    "NADA": datos1[datos1["NIVEL_ESTRES"] == "NADA"]["PORCENTAJE_ESTRES"].median(),
    "POCO": datos1[datos1["NIVEL_ESTRES"] == "POCO"]["PORCENTAJE_ESTRES"].median(),
    "MUCHO": datos1[datos1["NIVEL_ESTRES"] == "MUCHO"]["PORCENTAJE_ESTRES"].median(),
})

# Promedio

C7_promedio = datos1["NIVEL_ESTRES"].map({
    "NADA": datos1[datos1["NIVEL_ESTRES"] == "NADA"]["PORCENTAJE_ESTRES"].mean(),
    "POCO": datos1[datos1["NIVEL_ESTRES"] == "POCO"]["PORCENTAJE_ESTRES"].mean(),
    "MUCHO": datos1[datos1["NIVEL_ESTRES"] == "MUCHO"]["PORCENTAJE_ESTRES"].mean(),
})

# Desviación estándar

C7_desviacion = datos1["NIVEL_ESTRES"].map({
    "NADA": datos1[datos1["NIVEL_ESTRES"] == "NADA"]["PORCENTAJE_ESTRES"].std(),
    "POCO": datos1[datos1["NIVEL_ESTRES"] == "POCO"]["PORCENTAJE_ESTRES"].std(),
    "MUCHO": datos1[datos1["NIVEL_ESTRES"] == "MUCHO"]["PORCENTAJE_ESTRES"].std(),
})

# Espacio de análisis (X, Y)

x1 = c1 # Edad

y1 = C7_natural # Nivel estrés natural

# - Espacios separados

figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
seaborn.kdeplot(
    x=x1,
)
pyplot.title("Densidad de x1 (Edad)")
pyplot.xlabel("Edad")
pyplot.ylabel("Densidad de la edad")

figure.savefig("densidad_x1.png", dpi=300)

figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
seaborn.kdeplot(
    x=y1,
)
pyplot.title("Densidad de y1 (Nivel de estrés)")
pyplot.xlabel("Nivel de estrés")
pyplot.ylabel("Densidad del nivel de estrés")

figure.savefig("densidad_y1.png", dpi=300)

# - Espacio contiguo

figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
seaborn.scatterplot(
    x=x1,
    y=y1,
    ax=axes,
)
pyplot.title("x1 vs y1")
pyplot.xlabel("Edad")
pyplot.ylabel("Nivel de estrés")

figure.savefig("x1_vs_y1.png", dpi=300)

print(x1)
print(y1)

# Espacio lineal

figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
seaborn.regplot(
    x=x1,
    y=y1,
    ax=axes,
    line_kws={"color": "red"}
)
pyplot.title("x1 vs y1")
pyplot.xlabel("Edad")
pyplot.ylabel("Nivel de estrés")

figure.savefig("regresion1.png", dpi=300)

print(x1)
print(y1)

# Betas

# y = b0 + b1 * x1
# betas = (b0, b1)
betas = numpy.array([0.001, -0.05]) # numpy.random.normal(0, 1, 2)

# Matriz de diseño (X = [1; x1])

X = numpy.array([numpy.ones_like(x1), x1]).T

print(
    pandas.DataFrame(X)
)

#    0   1
# 0  1  25
# 1  1  13
# 2  1  28
# 3  1  55
# 4  1  43
# 5  1  18

# Predictor (z)
z = X.dot(betas)
Xp = numpy.array([
    numpy.ones(20),
    numpy.linspace(0, 80, 20)
]).T
zp = Xp.dot(betas)

figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
seaborn.scatterplot(
    x=z,
    y=y1,
    ax=axes,
)
seaborn.lineplot(
    x=zp,
    y=zp,
    ax=axes,
)
pyplot.title("z vs y1")
pyplot.xlabel("z = X • betas")
pyplot.ylabel("Nivel de estrés")

figure.savefig("z_vs_y1.png", dpi=300)

# Perceptrón lineal

alpha = 0.000001

for epoca in range(1000):
    z = X.dot(betas)

    yp = z

    e = y1 - yp

    L = (1 / 2) * e.dot(e)

    print(epoca, L)

    # --- GRÁFICA DE z vs y1 POR ÉPOCA
    if epoca % 100 == 0:
        Xp = numpy.array([
            numpy.ones(20),
            numpy.linspace(0, 80, 20)
        ]).T
        zp = Xp.dot(betas)

        figure, axes = pyplot.subplots(1, 1, figsize=(10, 10))
        seaborn.scatterplot(
            x=z,
            y=y1,
            ax=axes,
        )
        seaborn.lineplot(
            x=zp,
            y=zp,
            ax=axes,
        )
        pyplot.title("z vs y1")
        pyplot.xlabel("z = X • betas")
        pyplot.ylabel("Nivel de estrés")

        figure.savefig(f"z_vs_y1_epoca_{epoca}.png", dpi=300)

    # --- TERMINA LA GRÁFICA ---

    betas = betas + alpha * e.dot(X)

