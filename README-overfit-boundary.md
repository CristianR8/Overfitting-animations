# CVAIL · Frontera de decisión

Intro de ManimGX con puntos verdes y dorados y una frontera de decisión que pasa de sencilla a irregular para separar ejemplos mezclados. Se conservan la paleta, los logos y los títulos de CVAIL.

## Secuencia

- **0–3,3 s:** los puntos aparecen con una inclinación suave que termina frontal. Se dibuja una frontera sencilla.
- **3,6–6,9 s:** la frontera incorpora recodos para separar los ejemplos atípicos de ambas clases. Los puntos permanecen en su lugar.
- **7,1–9,6 s:** aparecen cinco muestras huecas que siguen una separación más sencilla y quedan en desacuerdo con los recodos del modelo.
- **9,6–12,2 s:** el gráfico se desvanece y el fondo verde, el título y los logos aparecen mediante un fundido. Todo permanece en posiciones fijas durante esta transición.
- **12,2–15,5 s:** cierre con «CVAIL Talk Series», «Overfitting with», la secuencia α β γ δ Σ y los logos originales.

Sin ejes, etiquetas ni música. Los colores son verde `#243F34`, dorado `#CCA064` y blanco. Las imágenes utilizadas son los logos originales.

## Renderizar

```bash
cd /home/cristiancrr/CVAIL-animation
./render_overfit.sh
./render_overfit.sh --preview
```

`./render.sh` también genera esta versión. La salida es `renders/manimgx_overfit/CVAIL-Overfit-Intro.mp4`: 1920 × 1080, 60 fps y 15,5 segundos.

## Archivos

- `manimgx_overfit/scene.py`: escena `CVAILOverfitIntro`, geometría, datos y animación.
- `manimgx_overfit/config.json`: colores, textos y tiempos del ajuste y los fundidos.
- `manimgx_overfit/finalize.py`: exportación H.264 a frecuencia constante, manteniendo el cierre quieto.
- `manimgx_overfit/assets/`: logos UIS y CVAIL originales.

La frontera se representa como `x = b(y)` mediante bases gaussianas. La reducción de su anchura y de la regularización produce recodos locales que separan las muestras de entrenamiento. Las dos clases están ligeramente escalonadas y sus puntos se mantienen fijos una vez que llegan. Los puntos huecos ilustran el desacuerdo de la frontera sobreajustada con una separación simple; son ejemplos seleccionados para visualizar el concepto, no un conjunto de evaluación estadística.

## Instalación en otro equipo

```bash
uv venv --python 3.13 .venv-manimgx
uv pip install --python .venv-manimgx/bin/python -r manimgx_overfit/requirements.txt
```

Se necesita acceso gráfico compatible con ManimGX. PyAV proporciona el codificador H.264; el script utiliza el entorno del proyecto o el Python del sistema si allí está disponible. No requiere Blender, LaTeX ni el ejecutable de FFmpeg.

En la carpeta de trabajo original se conservan también la propuesta del iris en `manimgx_overfit_iris/` y las animaciones de ondas en `manimgx/`.
