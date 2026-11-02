# CVAIL · Overfitting in motion

Intro 3D de 18 segundos, verde bosque y dorado, con escena editable de Blender y exportación MP4 H.264. La animación está en inglés como la referencia; todos los textos se cambian en `config.json`.

## Ejecutar en este equipo

Este equipo dispone de `/usr/bin/python` con `bpy` 5.1.2, NumPy y FFmpeg. No necesitas instalar un complemento de Blender.

```bash
cd /home/cristiancrr/CVAIL-animation
python animate.py --preset preview
```

Genera `renders/CVAIL-overfitting.blend`, los fotogramas PNG y `renders/CVAIL-overfitting.mp4` (1280 × 720, 24 fps). Usa EEVEE por defecto para texto nítido y un render ágil; `--engine cycles` activa Cycles CPU. La secuencia PNG permite continuar un render interrumpido: repite el mismo comando. No cambies la configuración, el motor o el preset al reanudar en la misma carpeta; utiliza otro `--output` para una variante. Los tramos completamente estáticos reutilizan su imagen sin modificar la duración ni la frecuencia del video.

```bash
# Cinco imágenes de revisión.
python animate.py --preset draft --stills --output renders/stills

# Solo construir la escena editable.
python animate.py --build-only

# Render final Full HD, 32 muestras por píxel.
python animate.py --preset full --output renders/full-hd

# Render 4K, 64 muestras por píxel.
python animate.py --preset 4k --output renders/4k
```

También funciona dentro de una instalación completa de Blender con NumPy incluido:

```bash
blender --background --python animate.py -- --preset full --output renders/full-hd
```

## Personalizar

Edita `config.json`: `guest` para el nombre del invitado (vacío muestra puntos suspensivos), `title`, `series`, `tagline`, `chapters` y colores hexadecimales. Vuelve a ejecutar con una carpeta de salida nueva. El guion está temporizado a 18 segundos: aumentar `seconds` prolonga el cierre; no cambia los tiempos de los capítulos.

No se incluyen logos oficiales extraídos del video ni música. El cierre reserva espacio para el invitado. Los textos y curvas son objetos nativos editables, los puntos son mallas y la superficie usa shape keys. Los marcadores de la línea de tiempo identifican los cuatro actos. Las fuentes se empaquetan en el archivo Blender; se usa Noto Sans cuando está disponible, con la fuente interna de Blender como alternativa.

El archivo `.blend` también está configurado para exportar directamente MP4 H.264 desde **Render → Render Animation**. El script utiliza PNG intermedios para poder reanudar el trabajo.

## Cómo se representa el overfitting

1. Datos sintéticos de regresión con ruido independiente.
2. Una regresión cuadrática aprende una tendencia suave.
3. Una interpolación de residuos con funciones de base radial gaussianas ajusta casi exactamente los datos de entrenamiento.
4. Aparecen observaciones de validación del mismo proceso generador, nunca usadas para entrenar ni modificar el modelo. Las líneas doradas son residuos de predicción.
5. Las curvas finales y `metrics.json` usan el MSE calculado en esos mismos datos, al aumentar la contribución del interpolador. Representan este ejemplo de regresión, no una ley universal sobre redes neuronales o el número de épocas.

## Librerías

- **bpy**: API oficial de Blender para escena, materiales, iluminación, textos, shape keys y fotogramas clave.
- **NumPy**: regresión, interpolación y geometría de la superficie.
- **FFmpeg**: une los PNG en un MP4 compatible con Google Slides y reproductores comunes; detecta `libx264` o `libopenh264` automáticamente.
- **fontTools** (opcional): convierte Noto Sans variable en fuentes estáticas para Blender.

Geometry Nodes es una buena herramienta nativa para ampliar el estilo con partículas o geometría procedural. Esta versión usa `bpy` y keyframes para que el `.blend` no dependa de complementos ni de handlers Python al abrirse.

Documentación: https://docs.blender.org/api/current/ · https://docs.blender.org/manual/en/latest/render/output/animation.html
