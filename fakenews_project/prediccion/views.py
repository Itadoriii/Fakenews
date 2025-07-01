import os
import joblib
import numpy as np
import torch
from tensorflow.keras.models import load_model
from transformers import DistilBertTokenizer, DistilBertModel

from django.shortcuts import render
import pandas as pd
from . import heuristicas_module  # tu módulo heurístico

# Rutas absolutas para que Django lo encuentre
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'model')

# Cargar modelos y transformadores guardados
mlp_model = joblib.load(os.path.join(MODELS_DIR, 'modelo_fakenews_mlp.pkl'))
keras_model = load_model(os.path.join(MODELS_DIR, 'modelo_fakenews_keras.h5'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
encoder = joblib.load(os.path.join(MODELS_DIR, 'encoder.pkl'))

# Cargar tokenizer y modelo BERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
bert_model.eval()

def limpiar_texto(texto):
    import re
    return re.sub(r'http\S+|www\S+|@\w+|#\w+|\s+', ' ', texto.lower()).strip()

def predecir(request):
    if request.method == 'POST':
        titulo = request.POST.get('titulo', '')
        contenido = request.POST.get('contenido', '')
        fecha = request.POST.get('fecha', '')
        autor = request.POST.get('autor', '')

        # Crear DataFrame con la entrada
        df = pd.DataFrame([{
            'titulo': titulo,
            'contenido': contenido,
            'fecha': fecha,
            'autor': autor
        }])

        # Embedding BERT
        texto_limpio = limpiar_texto(contenido)
        inputs = tokenizer(texto_limpio, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            X_bert = outputs.last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)

        # Ingeniería de features con tu función original
        df_meta, _ = heuristicas_module.feature_engineering(df)

        columnas_num = ['longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
                       'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido']

        # Limpiar valores infinitos y nulos
        df_meta[columnas_num] = df_meta[columnas_num].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Escalar solo columnas numéricas que conoce el scaler
        df_meta[columnas_num] = scaler.transform(df_meta[columnas_num])

        # Codificar autor con encoder guardado
        autor_encoded = encoder.transform(df_meta[['autor']])
        df_autor = pd.DataFrame(autor_encoded.toarray() if hasattr(autor_encoded, "toarray") else autor_encoded,
                                columns=encoder.get_feature_names_out(['autor']),
                                index=df_meta.index)

        # Concatenar columnas numéricas y codificadas
        df_meta_final = pd.concat([df_meta[columnas_num], df_autor], axis=1)
        X_meta = df_meta_final.to_numpy()

        # Embeddings heurísticos
        df_diccionario = pd.read_csv(os.path.join(BASE_DIR, 'diccionario.csv'))
        X_heur = heuristicas_module.generar_embedding_heuristico(df.copy(), df_diccionario).reshape(1, -1)

        # Concatenar todas las características
        X_final = np.concatenate([X_bert, X_meta, X_heur], axis=1)
        X_final = np.concatenate([X_bert, X_meta, X_heur], axis=1)

        # Validar tamaño y corregir si hace falta (rellenar con ceros)
        expected_features = keras_model.input_shape[1]
        actual_features = X_final.shape[1]

        if actual_features < expected_features:
            diff = expected_features - actual_features
            print(f"Rellenando {diff} columnas faltantes con ceros")
            X_final = np.hstack([X_final, np.zeros((X_final.shape[0], diff))])
        elif actual_features > expected_features:
            print(f"Recortando {actual_features - expected_features} columnas de más")
            X_final = X_final[:, :expected_features]

        # Predicciones
        pred_keras = keras_model.predict(X_final)[0][0]
        pred_mlp = mlp_model.predict_proba(X_final)[0][1]

        # Renderizar en la misma página y pasar predicciones
        return render(request, 'index.html', {
            'titulo': titulo,
            'contenido': contenido,
            'fecha': fecha,
            'autor': autor,
            'pred_keras': round(float(pred_keras), 2),
            'pred_mlp': round(float(pred_mlp), 2)
        })

    else:
        return render(request, 'index.html')

def index(request):
    return render(request, 'index.html')
