# Diplamado en Python y Ciencia de Datos
# Instituto Politécnico Nacional
# Centro de Investigación en Computación
# Departamento de Diplomados y Extensión Profesional

# Ciudad de México
# Sábado 29, Agosto 2026.

# Alan Badillo Salas

# Módulo IV - Deep Learning
# Introducción a las Redes Neuronales Multicapa

# Práctica 2 - Red neuronal de una capa

# Perceptrón lineal yp = z = x • betas

# Perceptrón logístico yp = sigma(z) = e^z / (1 + e^z) = sigma(x • betas)

# x1 -> (1, x1) • (b0, b1)
# x1, x2 -> (1, x1, x2) • (b0, b1, b2)
# x1, x2, ..., xk -> (1, x1, x2, ..., xk) • (b0, b1, b2, ..., bk)

# z = x • betas = (1, x1) • (b0, b1) = b0 + b1 * x1
# z = x • betas = (1, x1, x2) • (b0, b1, b2) = b0 + b1 * x1 + b2 * x2
# z = x • betas = (1, x1, x2, ..., xk) • (b0, b1, b2, ..., bk) = b0 + b1 * x1 + b2 * x2 + ... + bk * xk

# yp = z --> LINEAL
# yp = sigma(z) --> LOGÍSTICO

# Ejemplo 1: Tiempo en minutos que tarda un litro de helado en derretirse según su precio
# betas = (1, 1)
# z = 1 + 1 * x1
# e = y - yp
# x1  |  y |  z  |  yp |  e  |   l
# ----------------------------------
# 20  | 11 |  21 |  21 | -10 |   50
# 40  | 17 |  41 |  41 | -24 |  288
# 60  | 25 |  61 |  61 | -36 |  648
# 80  | 31 |  81 |  81 | -50 | 1250
# 100 | 34 | 101 | 101 | -67 | 2244.5
# -----------------------------------
# L = 4480.5 | MSE: 896.1 RMSE: 29.93

# MSE  - Mean Square Error / Error Cuadrático Medio
# RMSE - Root Mean Square Error / Raíz del Error Cuadrático Medio

# Hay un error promedio lineal en la predicción de casi 30 minutos

# betas = (5, 0.3)
# z = 5 + 0.3 * x1
# e = y - yp
# x1  |  y |  z | yp | e  |  l
# ----------------------------------
# 20  | 11 | 11 | 11 |  0 |  0
# 40  | 17 | 17 | 17 |  0 |  0
# 60  | 25 | 23 | 23 |  2 |  2
# 80  | 31 | 29 | 29 |  2 |  2
# 100 | 34 | 35 | 35 | -1 |  0.5
# -----------------------------------
# L = 4.5   | MSE: 0.9 RMSE: 0.94

# Hay un error promedio lineal en la predicción de casi 1 minuto

# Un helado que cuesta $300, se derretirá en 95 ± 1 minutos según el modelo

import numpy
import pandas
from matplotlib import pyplot
import seaborn

x1 = numpy.array([20, 40, 60, 80, 100])
y = numpy.array([11, 17, 25, 31, 34])

pyplot.clf()

figure, axes = pyplot.subplots(1, 1, figsize=(5, 5))

seaborn.lineplot(x=x1, y=y, ax=axes, 
                 marker="o", 
                 linestyle="--", 
                 markerfacecolor="red",
                 color="gray")

pyplot.savefig("s203p/graficas/x1_vs_y.png", dpi=300)

x1p = numpy.linspace(0, 120, 100) # Vector de todas las x's no observadas

yp = 5 + 0.3 * x1p # Vector de prección para las x's no observadas

pyplot.clf()

figure, axes = pyplot.subplots(1, 1, figsize=(5, 5))

seaborn.lineplot(x=x1p, y=yp, ax=axes, color="red")
seaborn.lineplot(x=x1, y=y, ax=axes, color="yellow")

pyplot.savefig("s203p/graficas/x1p_vs_yp.png", dpi=300)

# betas* = betas + eta * e • (1, x1, ..., xk)









