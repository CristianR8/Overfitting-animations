# CVAIL · Curva de overfitting

Versión principal: la curva verde tridimensional que se ajusta a los puntos dorados, como en la captura seleccionada. Se conserva la entrada original, el ajuste progresivo y los puntos huecos de comprobación. El cierre usa el fundido suave al título y los logos, con todos los elementos quietos.

## Generar el video

```bash
cd /home/cristiancrr/CVAIL-animation
./render_overfit.sh
```

`./render.sh` genera la misma versión. `./render_overfit.sh --preview` genera una prueba 720p/30 fps.

**Video principal:** `renders/manimgx_overfit/CVAIL-Overfit-Regression.mp4`.

**Código principal:** `manimgx_overfit/scene.py`, escena `CVAILOverfitIntro`.

- `manimgx_overfit/config.json`: colores, textos, duración y tiempos de los fundidos.
- `manimgx_overfit/finalize.py`: exportación H.264 a 60 fps constantes, con cierre estático.
- `manimgx_overfit/assets/`: logos originales.

La salida final es 1920 × 1080, 60 fps y 15,5 segundos. No contiene música. El modelo usa bases gaussianas progresivamente más estrechas y menor regularización: termina memorizando el ruido de los puntos dorados mientras se aleja de las muestras huecas.

## Instalar en otro equipo

```bash
uv venv --python 3.13 .venv-manimgx
uv pip install --python .venv-manimgx/bin/python -r manimgx_overfit/requirements.txt
```

Se necesita acceso gráfico compatible con ManimGX. El script utiliza PyAV del entorno del proyecto o del Python del sistema para codificar el MP4.

En el proyecto original se conservan las propuestas anteriores en `manimgx_overfit_boundary/`, `manimgx_overfit_iris/` y `manimgx/`.
