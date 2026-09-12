# Aprendizaje Supervisado

Conjunto de datos: X (carecterísticas), Y (respuestas)

- **Regresión** - Estimar una respuesta continua
- **Clasificación** - Identificar la clase a la que pertenece una respuesta

---

## Modelo de Aprendizaje

Espacio de carecterísticas (E) 
    -> Espacio numérico (C) 
    -> Espacio de análisis (X, Y) / (predictivas, respuestas)

Tipos de respuestas:

* **Respuesta abierta** - No acotada con potencial espacio de $(-\infty, \infty)$
* **Respuesta cerrada** - Acotada con potencial espacio de $[0, 1]$ (espacio binario o probabilístico)

Modelo de aprendizaje/predicción $\hat{y} = \phi(X, \beta)$

* **Modelo lineal** - $\beta_0 + \beta_1 \cdot x_1 + \beta_2 \cdot x_2 + \ldots +  \beta_k \cdot x_k$
* **Modelo logístico** - $\sigma(\beta_0 + \beta_1 \cdot x_1 + \beta_2 \cdot x_2 + \ldots +  \beta_k \cdot x_k)$
* **Modelo z** - $\phi(z) \qquad;\qquad z = x \cdot \beta$

## Perceptrón ($X, Y, \phi, \beta, L, \varepsilon, \lambda$)

1. $\hat{Y}$ - Crea una predicción $\hat{y} = \phi(x, \beta)$
2. $\varepsilon$ - Mide la diferencia entre la respuesta real/observada y la respuesta de predicción/modelo $\varepsilon = y - \hat{y}$
3. $L$ - Construye un modelo para optimizar el error, es decir, una función de pérdida sobre el error (qué tanto logro mejorar el error si cambio las $\beta$), se considera la función objetivo a minimizar para mejorar las $\beta$ (dirección del error)
4. $\lambda(\beta, L)$ - Optimiza las $\beta$ con la información de la pérdida para minimizar el error, la pérdida establece dónde está el error mínimo, luego, las $\beta$ cambian error y finalmente aumentan o disminuyen la pérdida, el optimizar busca las $\beta$ que siempre minimices la pérdida: $\beta^* = \beta + \eta \nabla_{\beta} L$

**Nota:**

* $-\nabla_{\beta} L$ - Dirección donde la pérdida se minimiza al cambiar las $\beta$
* $\eta$ - El tamaño de paso del gradiente o tasa de aprendizaje con la que cambiará la $\beta$, una tasa pequeña hace que la $\beta$ cambie muy poco y la optimización sea estable, pero necesitaremos más epocas o iteraciones para lograr el aprendizaje.

## Red neuronal de una capa

Una red neuronal une a muchos perceptrones en una capa, y les llama neuronas.


$$
\Phi(h^{(l)}, B^{(l)})
$$

La red neuronal de una capa, se puede ver como un vector de perceptrones, y si todos tienen la misma función de activación $\phi$, entonces $\Phi = (\phi_1, \phi_2, ..., \phi_{l_1})$

## Red neuronal multicapa

Una red neuronal une a muchos perceptrones en una capa, y les llama neuronas.


$$
\Phi_1(h^{(0)}, B^{(1)}), \Phi_2(h^{(1)}, B^{(2)}), ..., \Phi_{l + 1}(h^{(l)}, B^{(l + 1)})
$$

La red neuronal multicapa, consiste en capas ocultas llamas capas **Densas** de una misma activación

## Aplicación con Keras

```py
import keras

modelo = keras.Sequential([
    keras.Input(shape=(k,)),
    keras.layers.Dense(l1, activation="relu"),
    keras.layers.Dense(l2, activation="relu"),
    ...
    keras.layers.Dense(L, activation="relu"),
    keras.layers.Dense(s, activation="linear|sigmoid")
])

print(
    modelo.summary()
)

modelo.compile(
    optimizer="sgd|adam|...",
    loss="mse|binary_crossentropy|categorical_crossentropy|...",
    metrics=["mae|accuracy|..."]
)

historial = modelo.fit(X, Y, epochs=T)

# Ver cómo fluctúa el error y decae

Yp = modelo.predict(Xb)

# Comparar manualmente Yp vs Yb

# Guardar la red ya entrenada 
# y usuarla como motor de predicciones

Yc = modelo.predict(Xc)
```


