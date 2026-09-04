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

import pandas

# b0, b1, b2
betas = numpy.array([1, 1, 1]) # numpy.random.normal(0, 1, 3) - normal(mu, sigma, p)

n = 14 # len(x1) | len(x2) | len(y)

perdidas = []

for t in range(2000):
    z = numpy.zeros(n)
    yp = numpy.zeros(n)

    for i in range(0, n):
        xi = numpy.array([1, x1[i], x2[i]]) # Vector de diseño que considera la b0 | bias
        # b0 = betas[0]
        # b1 = betas[1]
        # b2 = betas[2]
        # z = b0 + b1 * x1[i] + b2 * x2[i]
        z[i] = xi.dot(betas) # (1, x1_i, x2_i) • (betas_0, betas_1, betas_2)
                        # (1, 18, 45) • (0.1, -0.1, 0.2)
        yp[i] = z[i] # yp = z

    e = y - yp
    l = (1 / 2) * e ** 2

    # print(
    #     pandas.DataFrame({
    #         "x1": x1,
    #         "x2": x2,
    #         "y": y,
    #         "z": z,
    #         "yp": yp,
    #         "e": e,
    #         "L": l,
    #     })
    # )

    L = l.sum()

    print("betas", betas)
    print(f"L={L}")
    print("-" * 20)

    betas = betas + (0.00001) * e.dot(numpy.array([numpy.ones(n), x1, x2]).T)

    print("betas*", betas)
    print("=" * 20)

    perdidas.append((t, L))

Vt, VL = zip(*perdidas)

pyplot.clf()
seaborn.lineplot(x=Vt, y=numpy.log(VL))
pyplot.savefig("s202/graficas/perdida_tiempo.png", dpi=300)

# GRID SEARCH -> Buscar hiperparámetros (tasa, épocas, lotes, ...)