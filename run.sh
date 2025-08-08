#!/bin/bash

# 1. Crear entorno virtual
echo "1. Creando entorno virtual\n"
python3 -m venv venv
source venv/bin/activate
# 1.1 Actualizar pip
#echo "Actualizando PIP"
#pip install --upgrade pip

# 2. Instalar dependencias
#echo "2. Instalando dependencias\n"
#pip install torch sentencepiece numpy
#echo "For CUDA\n"
#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 3. Crear directorio de datos y un corpus de ejemplo
echo "Creando directorio de datos y un corpus de ejemplo\n"
mkdir -p data
#echo -e "El futuro de la inteligencia artificial es prometedor.\nLa tecnología avanza rápidamente." > data/corpus.txt

# 4. Entrenar un modelo SentencePiece (BPE)
echo "Entrenando un modelo SentencePiece (BPE)\n"
python -c "
import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input=data/corpus.txt --model_prefix=bpe_sp --vocab_size=30000 --model_type=bpe --max_sentence_length=5000')
"

# 5. Ejecutar el código principal
echo "Ejecutando código principal\n"
python main.py
echo "Modelo generado!\n"
echo "Desactivando entorno virtual\n"
# 6. Desactivar entorno virtual
deactivate
