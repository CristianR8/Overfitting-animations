# CVAIL · Intro 3D, versión 2

La composición de referencia llevada a 3D: una curva verde continua, una curva dorada discontinua, separación de las curvas, bloque verde superior izquierdo y título al cierre. Sin explicación, tarjetas, ejes, leyendas ni textos adicionales. Un único movimiento de cámara. Duración: 9 segundos.

```bash
cd /home/cristiancrr/CVAIL-animation
python intro.py --preset full
```

Genera `renders/intro-v2/CVAIL-intro-3D.mp4` y la escena editable `CVAIL-intro-3D.blend`, con fuentes empaquetadas. Requiere Python con `bpy`, NumPy y FFmpeg. También se puede ejecutar con `blender -b --python intro.py -- --preset full`.

Edita `intro.json` para el título, el nombre opcional del invitado y los colores. Para una variante usa una carpeta nueva, por ejemplo `--output renders/invitado-01`; los PNG existentes se reutilizan para continuar renders interrumpidos.

Presets: `draft` (640×360), `preview` (1280×720), `full` (1920×1080), `4k` (3840×2160). `--stills` genera cinco imágenes de revisión; `--build-only` guarda solo la escena. La escena está configurada para MP4 H.264 desde Render Animation. La pausa final reutiliza el mismo fotograma, sin alterar los 24 fps del archivo.

Si está presente, `animate.py` y sus renders corresponden a la primera propuesta y se conservan por separado. Esta versión se ejecuta con `intro.py`.
