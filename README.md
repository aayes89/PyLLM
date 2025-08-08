# PyLLM
Genera y entrena un LLM desde cero con tus propios datos.

El código implementa un LLM utilizando las siguientes dependencias:
- <b>os, math, random, pathlib</b>     - para funciones de sistema y aritmética
- <b>torch</b>                         - para entrenamiento con CPU o GPU (cuda)
- <b>sentencepiece</b>                 - para tokenización

# Estructura del proyecto
- <b>run.sh</b>: script batch para inicializar el proceso de ejecución.
- <b>main.py</b>: script python que genera y entrena el LLM.
- <b>generate_text.py</b>: script python que permite interactuar con el modelo una vez generado, sin necesidad de entrenarlo.
- <b>data</b>: directorio creado por <b>run.sh</b> donde se encuentra el corpus a entrenar el LLM.
- <b>corpus.txt</b>: archivo de texto plano codificado en UTF-8 con el que entrena el LLM. (puede ser generado automáticamente por run.sh o puedo cargarlo con datos propios)
- <b>venv</b>: directorio creado por <b>run.sh</b> para el entorno de virtual de python.

# Selección de hardware
Si va a utilizar un entrenamiento con GPU, descomentar la línea:<br>
<b># device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')</b><br><br>
Comentar la siguiente en caso de utilizar sólo CPU:<br>
<b>device = torch.device('cpu')</b>

# Dataset de lenguaje
<p>
El dataset devuelve pares: <b>(input_ids, labels)</b> donde <code>input_ids[i] = token t_i</code> y <code>labels[i] = token t_{i+1}</code> <br>
Se recorta en bloques de `block_size`.<br>
Se añade un token EOS (id = vocab-1) al final de cada texto.
</p>

# Modelo Transformer “desde cero por etapas” 
- Las Proyecciones lineales para Q, K, V y salida final.
- Escalamiento dot‑product attention.
- Concatenamiento de cabezas y proyección final.
- Self‑attention + residual.
- FFN + residual.
- Tensor: <code> input_ids : (B, T)  dtype=torch.long        returns logits : (B, T, vocab)</code>

# Entrenamiento / Evaluación
# Generación de texto

# Ficheros generados

El script genera los siguientes archivos:<br>
- <b>llm_es.pth</b>: el modelo generado. 
- <b>bpe_sp.model</b>: contiene información con la que opera el modelo generado.
- <b>bpe_sp.vocab</b>: contiene información sobre los vocabularios con los que opera el modelo.
