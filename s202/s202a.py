import numpy

# Problema - Kilómetros recorridos en mujeres por edad (años) y peso (kilogramos)

# n = 13 observaciones en 2 características numéricas predictivas y 
#           1 característica numérica no acotada de respuesta

# Edades (años)
x1 = numpy.array([
    18,
    19,
    20,
    21,
    22,
    23,
    25,
    26,
    30,
    31,
    40,
    41,
    50,
    51,
])

# Pesos (kilogramos)
x2 = numpy.array([
    45,
    52,
    48,
    55,
    52,
    60,
    57,
    63,
    60,
    70,
    62,
    68,
    58,
    64,
])

# Distancias (kilómetros)
y = numpy.array([
    5.10,
    4.80,
    4.90,
    4.20,
    4.10,
    3.50,
    3.80,
    3.20,
    3.10,
    2.50,
    2.50,
    1.80,
    2.10,
    1.90,
])

# y <- (x1, x2)

from matplotlib import pyplot
import seaborn

pyplot.clf()
seaborn.scatterplot(x=x1, y=y)
pyplot.savefig("s202/graficas/x1_vs_y.png", dpi=300)

pyplot.clf()
seaborn.scatterplot(x=x2, y=y, color="orange")
pyplot.savefig("s202/graficas/x2_vs_y.png", dpi=300)

# pyplot.clf()
# seaborn.jointplot(x=x1, y=x2, hue=y)
# pyplot.savefig("s202/graficas/x1_x2_vs_y.png", dpi=300)

# Perceptrón: betas*

