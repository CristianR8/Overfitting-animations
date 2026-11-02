# CVAIL · ManimGX · 3D → 2D

Las dos señales oscilan juntas: comparten la misma frecuencia y la misma fase, tanto en altura como en profundidad. Mantienen verde y dorado fijos. Las flechas avanzan de izquierda a derecha por la tangente y van dibujando las señales detrás de ellas. Tras la órbita y la separación, siguen construyendo las curvas frontales y salen por el borde derecho; desaparecen fuera de pantalla y la gráfica permanece sin flechas.

La cámara parte de una vista frontal, se inclina y realiza una órbita completa de 360° alrededor de las señales mientras estas siguen viajando. Después vuelve suavemente a la vista frontal y las curvas pierden su profundidad para terminar en 2D. El encuadre sigue las señales durante el giro y queda fijo al llegar a la vista frontal para que las puntas salgan hacia la derecha. El tramo inicial de las curvas finales mantiene una separación normal de 0,55 unidades hasta la bifurcación.

El cierre centra el bloque de texto en la zona blanca y amplía «CVAIL Talk Series». Muestra los distintivos UIS, el logo de cámara adjuntado y la secuencia α β γ δ ε ζ η θ Σ bajo el título. Los segmentos dorados siguen avanzando en 2D. Duración: 15 s. Sin ejes, explicación ni música.

## Ejecutar

ManimGX 0.1.1 está instalado en el entorno aislado `.venv-manimgx` del proyecto.

```bash
cd /home/cristiancrr/CVAIL-animation
.venv-manimgx/bin/manimgx inspect manimgx/scene.py CVAILIntro -t 0,1.5,4,6.5,9.8,14.6 -o renders/manimgx/storyboard.png
.venv-manimgx/bin/manimgx render manimgx/scene.py CVAILIntro -r 1920x1080 --fps 60 --crf 18 -o renders/manimgx/CVAIL-ManimGX.mp4
```

El comando principal `./render.sh` genera esta versión. Usa `./render.sh --preview` para 720p a 30 fps.

## Personalizar

Edita `manimgx/config.json`: `wave_amplitude`, `depth_amplitude`, `wave_cycles` y `wave_speed` controlan el vuelo; `dash_flow_speed` controla el avance de los segmentos al cierre. `front_hold` y `opening_seconds` controlan el arranque frontal; `orbit_start` y `orbit_seconds` controlan la vuelta completa. `signal_start` controla el origen del trazo; `draw_keyframes` controla el avance de las puntas mediante fotogramas de tiempo, posición y velocidad. `arrow_hide_start` oculta las puntas una vez que ya han salido de pantalla. `greek_start`, `greek_interval` y `greek_letters` controlan los símbolos finales. `series`, `title`, `guest` y los colores controlan la identidad. Los logos están en `manimgx/assets/`; se conserva la transparencia original del logo adjuntado. Los distintivos UIS se extrajeron del video de referencia.

Las líneas iniciales son mallas de tubos, no una curva plana girada. La cámara y todos los vértices se calculan a partir del tiempo, de forma independiente de los fps. Al terminar la transición, todos los vértices de las líneas tienen Z=0 y color plano.

## Instalar en otro equipo

```bash
uv venv --python 3.13 .venv-manimgx
uv pip install --python .venv-manimgx/bin/python manimgx==0.1.1
```

ManimGX incluye su codificador de MP4 y sus fuentes. En Linux necesita acceso a Vulkan o al renderizador de software incluido. No requiere Blender, Manim CE ni LaTeX.

Documentación: https://manimgx.academa.ai/
