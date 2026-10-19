# EP2 Deep Learning (DLY0100) — CNN para clasificar personajes de Los Simpsons

**Asignatura:** Deep Learning · **Evaluación:** Parcial N°2 · **Semestre:** 2° de 2026
**Institución:** Duoc UC

## Objetivo y resultado

Entrenar una red neuronal convolucional (CNN) en TensorFlow/Keras que clasifique imágenes de
**18 personajes de Los Simpsons**, con una accuracy en test **de al menos 85%**.

El cuaderno `notebooks/EP2_DLY0100_CNN_Simpsons.ipynb` **es el informe**: cada decisión técnica está
justificada en celdas Markdown, y todos los números citados provienen de las salidas ejecutadas.

## Resultado obtenido

| Métrica (conjunto de test, 890 imágenes) | Valor |
|---|---|
| **Accuracy** | **97,87%** |
| Precision (macro) | 0,9794 |
| Recall (macro) | 0,9787 |
| F1 (macro) | 0,9787 |
| Imágenes mal clasificadas | **19 de 890** |
| Objetivo del enunciado | ≥ 85% — **cumplido con +12,87 puntos de margen** |

## El dataset

Imágenes extraídas de capítulos de la serie, recopiladas por Alexandre Attia. Se usan los 18
personajes con más imágenes (el dataset original trae 47 carpetas).

| | Cantidad |
|---|---|
| Imágenes de entrenamiento | 18.992 |
| Imágenes de test | 890 |
| Clases | 18 |
| Resolución de entrada | 64×64 px, RGB |
| Desbalanceo | 6,27× (Homer 2.246 vs Nelson 358) |
| Clasificador trivial (siempre Homer) | 11,8% |
| Azar | 5,56% |

Los tamaños originales van de 256 a 1912 píxeles de ancho, así que el `resize` a 64×64 es
obligatorio. Las imágenes **no vienen normalizadas**: los píxeles se llevan de 0–255 a 0–1.

## Cómo ejecutar

### 1. Entorno

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (bash)
# source .venv/bin/activate        # Linux / macOS
pip install -r requirements.txt
```

Versiones exactas con las que se ejecutó (Python 3.11): TensorFlow 2.21.0, Keras 3.15.1,
OpenCV 5.0.0, NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, Matplotlib 3.11.2.

### 2. Datos

Descargar los dos archivos indicados en el enunciado:

- **train:** https://onedrive.live.com/download?cid=C506CF0A4F373B0F&resid=C506CF0A4F373B0F%219337&authkey=AMzI92bJPx8Sd60 (~500 MB)
- **test:** https://onedrive.live.com/download?cid=C506CF0A4F373B0F&resid=C506CF0A4F373B0F%219341&authkey=ANnjK3Uq1FhuAe8 (~10 MB)

Descomprimirlos dentro de `data/raw/`:

```bash
mkdir -p data/raw
tar -xzf simpsons_train.tar.gz -C data/raw
tar -xzf simpsons_test.tar.gz  -C data/raw
```

La estructura resultante debe ser:

```
data/raw/simpsons/<personaje>/pic_XXXX.jpg        # 18.992 imagenes en 47 carpetas
data/raw/simpsons_testset/<personaje>_<n>.jpg     # 890 imagenes en una sola carpeta
```

### 3. Ejecución

Abrir `notebooks/EP2_DLY0100_CNN_Simpsons.ipynb` y ejecutar todas las celdas en orden
(`Kernel → Restart & Run All`), o desde la terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=5400 \
  notebooks/EP2_DLY0100_CNN_Simpsons.ipynb
```

**Tiempo esperado:** unos 75 minutos en una CPU de 8 núcleos (casi todo en entrenamientos). Con GPU
en Colab, mucho menos.

El cuaderno busca los datos automáticamente en `data/raw/`, en `../data/raw/` y en `/content/data/raw`
(para Colab), así que no hace falta editar rutas.

## Arquitectura

```
Entrada 64×64×3
  → Rescaling(1/255)                       normalizacion dentro del modelo
  → Conv 32 (3x3) + BN + ReLU + MaxPool    → 32x32x32
  → Conv 64 (3x3) + BN + ReLU + MaxPool    → 16x16x64
  → Conv 128 (3x3) + BN + ReLU + MaxPool   → 8x8x128
  → Flatten                                → 8.192
  → Dense 256 + BN + ReLU
  → Dropout 0.5
  → Dense 18 + softmax
```

**≈ 2,2 millones de parámetros.** Optimizador Adam (lr = 1e-3), pérdida `categorical_crossentropy`,
batch 128, parada temprana con paciencia 6 sobre `val_loss` y `ReduceLROnPlateau`.

El modelo final se entrenó **25 épocas en 25,7 minutos**, con una mejor accuracy de validación de
**0,9113** y una accuracy final en test de **0,9787**.

## Decisiones técnicas que vale la pena conocer

**1. `Rescaling` dentro del modelo en vez de convertir todo a `float32`.**
Medido en este equipo con las mismas 15.193 imágenes:

| Forma | RAM | Tiempo por época |
|---|---|---|
| `astype("float32") / 255` antes de entrenar | 760 MB | 187 s |
| `uint8` + `layers.Rescaling(1/255)` en el modelo | 190 MB | **47 s** |

Cuatro veces menos memoria y cuatro veces más rápido. El modelo guardado sabe procesar una imagen
cruda, sin que nadie tenga que recordar dividirla.

**2. `Flatten + Dense` en vez de `GlobalAveragePooling`.**
Se probó primero el GAP, que es la opción "moderna" porque reduce parámetros. Dio **62,25% en test**
con una brecha train−val de solo 0,024: estaba **subajustando**, no sobreajustando. GAP promedia cada
mapa de activación y pierde la información de **dónde** está cada rasgo — y para distinguir a Homer
de Flanders la disposición espacial del bigote y el pelo importa. Con `Flatten + Dense` el mismo
entrenamiento subió a **97,87%**.

**3. Un subconjunto fijo para los experimentos, dimensionado midiendo.**
Con las 18.992 imágenes una época tarda 47 s; trece configuraciones habrían costado más de 10 horas.
Se usa una muestra fija de 6.000 imágenes y 10 épocas para todas las comparaciones —misma muestra,
misma partición, misma semilla— y el modelo final se entrena con **todas** las imágenes.

El tamaño de la muestra se eligió con datos, no al azar: con 3.500 imágenes el modelo **memorizaba
antes de generalizar** (brecha de 0,84 y todas las configuraciones dando lo mismo), y con 8.000
llegaba a 0,73. Con 6.000 la comparación ya distingue.

**4. Tres espacios de datos distintos, y no se mezclan.**
`train` (15.193) para aprender, `validation` (3.799) para la parada temprana y las decisiones, `test`
(890) **solo** para la medición final. El test no participa en ninguna decisión, y por eso el número
es honesto.

## Experimentos que contiene el cuaderno

| # | Experimento | Qué se varía |
|---|---|---|
| 1 | CNN frente a red totalmente conectada | La presencia de capas convolucionales |
| 2 | Profundidad | 1, 2 o 3 bloques convolucionales |
| 3 | Optimizador | Adam frente a SGD |
| 4 | Aumento de datos | Con y sin `RandomFlip` / `RandomRotation` / `RandomZoom` |
| 5 | Regularización | Dropout, BatchNorm, ambas o ninguna |

En todos se cambia **una sola cosa a la vez**, con la misma muestra de 6.000 imágenes, las mismas 10
épocas, el mismo batch y la misma semilla.

### Resultados medidos

| Experimento | Accuracy de validación | Brecha train−val |
|---|---|---|
| Red densa (FC) | 0,1182 | +0,0005 |
| CNN, 1 bloque | 0,4367 | +0,5626 |
| CNN, 2 bloques | 0,5441 | +0,4557 |
| CNN, 3 bloques | 0,5928 | +0,4069 |
| Optimizador SGD | 0,4993 | +0,2588 |
| Optimizador Adam | 0,5928 | +0,4069 |
| Sin aumento de datos | 0,5928 | +0,4069 |
| **Con aumento de datos** | **0,6699** | **+0,2039** |
| Sin regularización | 0,7589 | +0,2411 |
| Solo Dropout 0,4 | 0,7305 | +0,2659 |
| Solo BatchNorm | 0,5957 | +0,4043 |
| Dropout + BatchNorm | 0,5928 | +0,4069 |

**Lecturas principales:**

1. **Convoluciones frente a capas densas:** 0,59 contra 0,12 con la misma muestra. La red densa tiene
   además 6,3 millones de parámetros contra 2,2 de la CNN. Es la demostración empírica de que este
   problema necesita compartir pesos.
2. **Profundidad:** cada bloque extra aporta (0,44 → 0,54 → 0,59), con rendimientos decrecientes.
3. **Optimizador:** Adam supera a SGD en casi 10 puntos con el mismo presupuesto de épocas.
4. **Aumento de datos:** sube la accuracy (0,59 → 0,67) **y reduce la brecha a la mitad** (0,41 →
   0,20). Es el resultado más limpio de todos.
5. **Regularización:** en esta muestra, **sin regularización es la mejor** (0,7589). BatchNorm
   empeora (0,5957) porque con 6.000 imágenes y 10 épocas apenas hay 47 actualizaciones de pesos por
   época, y eso no alcanza para estimar bien sus estadísticas móviles. Se verificó por separado: con
   3.500 imágenes, BatchNorm daba 0,28 y sin ella 0,63. **Es una limitación del experimento pequeño,
   documentada en el cuaderno en lugar de maquillada.**

## Estructura del repositorio

```
├── notebooks/
│   └── EP2_DLY0100_CNN_Simpsons.ipynb   <- el entregable (cuaderno ejecutado)
├── images/                              <- las figuras generadas, en PNG
├── data/
│   ├── tabla_experimentos.csv
│   └── metricas_por_clase.csv
├── models/
│   └── cnn_simpsons_final.keras         <- el modelo entrenado
├── requirements.txt
├── .gitignore
└── README.md
```

El dataset **no se versiona** (523 MB comprimido): se descarga siguiendo las instrucciones de arriba.

## Limitaciones reconocidas

- El `resize` a 64×64 **deforma las imágenes panorámicas** (hasta 1912×800), y el modelo nunca ve el
  detalle original.
- **No se usó transferencia de aprendizaje.** Con 19.000 imágenes una CNN desde cero es suficiente;
  con menos datos, una red preentrenada en ImageNet sería el primer recurso.
- **Una sola semilla por experimento.** Las diferencias menores a un punto porcentual podrían ser
  ruido de inicialización.
- **El test tiene solo 890 imágenes:** cada imagen vale 0,11 puntos de accuracy.
- El test resulta **más fácil** que la validación (97,87% contra 91,13% en el modelo final),
  exactamente como advierte el enunciado. Conviene no confundir ambas cifras: la de validación mide
  el desempeño durante el desarrollo, la de test es la que responde al objetivo.
- **El experimento de regularización no respalda el uso de BatchNorm** en el modelo final. Se
  mantiene por ser parte de la arquitectura base y por su comportamiento con el volumen completo de
  datos, y esa tensión está escrita en el cuaderno en vez de omitirse.
