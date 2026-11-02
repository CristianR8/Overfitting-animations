# CVAIL · Overfit / Iris

Intro nueva hecha en ManimGX. Conserva verde `#243F34`, dorado `#CCA064`, blanco, los logos UIS/CVAIL y los títulos «CVAIL Talk Series» y «Overfitting with».

## La animación

- **0–3,6 s:** partículas doradas viajan por el espacio y se ordenan. La cámara se aleja suavemente mientras aparece una curva verde sencilla.
- **3,6–6,7 s:** la curva gana complejidad y memoriza las posiciones de los puntos, incluido su ruido.
- **7–8,7 s:** llegan puntos huecos nuevos. Sus separaciones respecto de la curva muestran la pérdida de generalización, sin ejes, etiquetas ni explicación en pantalla.
- **8,7–12,1 s:** la curva se pliega en un iris, los puntos forman su borde y una apertura verde llena el encuadre.
- **12,1–16,5 s:** título grande, símbolos α β γ δ Σ y logos originales sobre verde. El cierre se mantiene quieto para una entrada limpia a la charla.

Sin música. Los brillos, las partículas, el trazo y la apertura son geometría renderizada por ManimGX, sin imágenes generadas ni material de archivo. Las únicas imágenes son los logos originales.

## Renderizar

```bash
cd /home/cristiancrr/CVAIL-animation
./render_overfit.sh
./render_overfit.sh --preview
```

`./render.sh` también genera la intro nueva. La animación anterior se conserva en `manimgx/` y se puede renderizar con `./render_manimgx.sh`.

Archivos de la nueva versión:

- `manimgx_overfit/scene.py`: escena `CVAILOverfitIntro`, geometría y animación.
- `manimgx_overfit/finalize.py`: exportación del cierre a 60 fps constantes.
- `manimgx_overfit/config.json`: colores, títulos, duración y tiempos de las transiciones.
- `manimgx_overfit/assets/`: logos originales.
- `renders/manimgx_overfit/CVAIL-Overfit-Intro.mp4`: video final, 1920 × 1080, 60 fps, 16,5 s.

El script usa ManimGX para todas las imágenes y `finalize.py` (PyAV/libx264) para conservar el cierre estático a frecuencia constante y preparar el MP4 para reproducción inmediata. No aplica filtros ni composición a las imágenes.

## El ajuste

Se usan 15 muestras de una tendencia suave más ruido fijo. La curva se calcula con bases gaussianas: disminuye el ancho de las bases y la regularización hasta interpolar los datos de entrenamiento. Los seis puntos nuevos proceden de la tendencia sin ruido. Son puntos huecos para diferenciarlos de las muestras memorizadas. Los datos y la función están en `scene.py`; no hay movimiento aleatorio entre renders.

## Instalación

El entorno `.venv-manimgx` ya está instalado en este equipo. En otro equipo:

```bash
uv venv --python 3.13 .venv-manimgx
uv pip install --python .venv-manimgx/bin/python -r manimgx_overfit/requirements.txt
```

Se necesita acceso gráfico compatible con ManimGX. PyAV proporciona el codificador H.264; el script utiliza el entorno del proyecto o el Python del sistema si allí está disponible. No requiere Blender, LaTeX ni el ejecutable de FFmpeg.
