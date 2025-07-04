from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tensorflow.keras.models import load_model

# --- ⚙️ Carga ÚNICA de recursos pesados ---
# Cargar modelo Keras entrenado
model = load_model('best_model.h5')

# Cargar transformadores entrenados
scaler = joblib.load('scaler_metadata.pkl')
scaler_bert = joblib.load('scaler_bert.pkl')
scaler_heuristic = joblib.load('scaler_heuristic.pkl')
ohe = joblib.load('ohe.pkl')

# Tokenizer y modelo BERT (en CPU)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model_bert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

# Diccionario de emociones
df_diccionario = pd.read_csv('diccionario.csv')

# --- 🔧 FUNCIONES AUXILIARES --- (pon las mismas que usas en tu script)

import re
import string
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

def construir_diccionario_emocional(df_diccionario):
    emociones = df_diccionario.columns[1:]
    diccionario_emocional = {}
    for _, fila in df_diccionario.iterrows():
        palabra = fila['palabra']
        emociones_asociadas = [emocion for emocion in emociones if fila[emocion] == 1]
        if emociones_asociadas:
            diccionario_emocional[palabra] = emociones_asociadas
    return diccionario_emocional

def calcular_emotividad(texto, diccionario):
    palabras = re.findall(r'\b\w+\b', str(texto).lower())
    total = len(palabras)
    emocionales = sum(1 for palabra in palabras if palabra in diccionario)
    return emocionales / total if total > 0 else 0

def emocion_predominante(texto, diccionario):
    palabras = re.findall(r'\b\w+\b', str(texto).lower())
    emociones = []
    for palabra in palabras:
        if palabra in diccionario:
            emociones.extend(diccionario[palabra])
    if emociones:
        return Counter(emociones).most_common(1)[0][0]
    else:
        return 'ninguna'

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

verbos_modales = ['debería', 'podría', 'tendría', 'habría', 'puede', 'pueden', 'podría', 'debe', 'deben', 'sería']
terminos_generalizadores = ['siempre', 'nunca', 'todos', 'nadie', 'jamás', 'ninguno', 'cualquiera']
verbos_opinion = ['creo', 'considero', 'opino', 'pienso', 'me parece', 'supongo', 'siento', 'estimo']
lexico_polarizado = ['horrible', 'excelente', 'terrible', 'maravilloso', 'desastroso', 'perfecto', 'abominable', 'magnífico']

def limpiar_texto_heuristica(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    for c in string.punctuation:
        texto = texto.replace(c, "")
    return texto

def frecuencia_heuristica(texto, palabras_clave):
    texto_limpio = limpiar_texto_heuristica(texto)
    palabras = texto_limpio.split()
    return sum(1 for palabra in palabras if palabra in palabras_clave)

def diversidad_lexica(texto):
    texto_limpio = limpiar_texto_heuristica(texto)
    palabras = [p for p in texto_limpio.split() if p.isalpha() and p not in stop_words]
    if len(palabras) == 0:
        return 0
    return len(set(palabras)) / len(palabras)

import spacy
nlp = spacy.load("es_core_news_sm", disable=["ner", "tagger"])

def aplicar_patrones_sintacticos(df_heuristica):
    resultados = []
    for doc in nlp.pipe(df_heuristica['contenido'], batch_size=32):
        oraciones = list(doc.sents)
        total_palabras = len([token.text for token in doc if token.is_alpha])
        total_palabras_mayus = len([token.text for token in doc if token.is_upper])
        total_exclamaciones = doc.text.count('!')
        frases_cortas = sum(1 for s in oraciones if len(s.text.split()) <= 7)
        porcentaje_mayus = total_palabras_mayus / total_palabras if total_palabras > 0 else 0
        frases_cortas_porcentaje = frases_cortas / len(oraciones) if oraciones else 0
        resultados.append({
            "porcentaje_mayusculas": porcentaje_mayus,
            "exclamaciones_totales": total_exclamaciones,
            "frases_cortas": frases_cortas,
            "frases_cortas_%": frases_cortas_porcentaje
        })
    return pd.DataFrame(resultados, index=df_heuristica.index)

def consistencia_jaccard(titulo, contenido):
    def limpiar(texto):
        texto = texto.lower()
        for c in string.punctuation:
            texto = texto.replace(c, "")
        return texto

    titulo = limpiar(titulo)
    contenido = limpiar(contenido)
    set_titulo = set([p for p in titulo.split() if p not in stop_words])
    set_contenido = set([p for p in contenido.split() if p not in stop_words])
    if len(set_titulo) == 0 or len(set_contenido) == 0:
        return 0
    interseccion = set_titulo.intersection(set_contenido)
    union = set_titulo.union(set_contenido)
    return len(interseccion) / len(union)

def generar_embedding_heuristico(df, df_diccionario):
    diccionario_emocional = construir_diccionario_emocional(df_diccionario)
    df['score_emotividad'] = df['contenido'].apply(lambda x: calcular_emotividad(x, diccionario_emocional))
    df['incertidumbre'] = df['contenido'].apply(lambda x: frecuencia_heuristica(x, verbos_modales + terminos_generalizadores))
    df['subjetividad'] = df['contenido'].apply(lambda x: frecuencia_heuristica(x, verbos_opinion + lexico_polarizado))
    df['diversidad_lexica'] = df['contenido'].apply(diversidad_lexica)
    patrones = aplicar_patrones_sintacticos(df)
    df = pd.concat([df, patrones], axis=1)
    df['consistencia_titulo_cuerpo'] = df.apply(lambda row: consistencia_jaccard(row['titulo'], row['contenido']), axis=1)
    features = ['score_emotividad', 'incertidumbre', 'subjetividad', 'diversidad_lexica',
                'porcentaje_mayusculas', 'exclamaciones_totales', 'frases_cortas',
                'frases_cortas_%', 'consistencia_titulo_cuerpo']
    return df[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()

def feature_engineering(df):
    df['longitud'] = df['contenido'].apply(lambda x: len(str(x).split()))
    df['longitud_titulo'] = df['titulo'].apply(lambda x: len(str(x).split()))
    df['exclamaciones'] = df['contenido'].apply(lambda x: str(x).count('!'))
    df['interrogaciones'] = df['contenido'].apply(lambda x: str(x).count('?'))
    sensacionalistas = ['increíble', 'urgente', 'impactante', 'viral', 'escándalo']
    df['palabras_sensacionalistas'] = df['contenido'].apply(lambda x: sum(x.lower().count(p) for p in sensacionalistas))
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df['mes'] = df['fecha'].dt.month.fillna(0)
    df['dia_semana_num'] = df['fecha'].dt.dayofweek.fillna(0).astype(int)
    df['noticias_autor'] = 1  # Para 1 noticia ingresada
    df['relacion_titulo_contenido'] = df.apply(lambda r: r['longitud_titulo'] / r['longitud'] if r['longitud'] > 0 else 0, axis=1)
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana_num'] / 7)
    df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana_num'] / 7)
    columnas_num = ['longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
                    'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido']
    df[columnas_num] = df[columnas_num].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df, scaler

# --- Vista raíz ---
def index(request):
    return render(request, 'index.html')

# --- Vista de predicción ---
def predecir(request):
    if request.method == 'POST':
        titulo = request.POST['titulo']
        contenido = request.POST['contenido']
        fecha = request.POST['fecha']
        autor = request.POST['autor']

        df_noticia = pd.DataFrame([{
            'titulo': titulo,
            'contenido': contenido,
            'fecha': fecha,
            'autor': autor
        }])

        # Heurístico
        X_heuristico = generar_embedding_heuristico(df_noticia.copy(), df_diccionario)
        X_heuristico = scaler_heuristic.transform(X_heuristico)

        # Metadata
        df_meta, _ = feature_engineering(df_noticia.copy())
        col_num = ['longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
                   'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido']
        X_num = df_meta[col_num].values
        X_num_scaled = scaler.transform(X_num)
        X_ciclicas = df_meta[['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos']].values
        X_autor = ohe.transform(df_meta[['autor']])
        X_metadata = np.concatenate([X_num_scaled, X_ciclicas, X_autor], axis=1)

        # BERT
        df_noticia['contenido_limpio'] = df_noticia['contenido'].apply(limpiar_texto)
        inputs = tokenizer(
            df_noticia['contenido_limpio'].iloc[0],
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model_bert(**inputs)
            X_bert = outputs.last_hidden_state[:, 0, :].squeeze().numpy().reshape(1, -1)
        X_bert = scaler_bert.transform(X_bert)

        # Final
        X_final = np.concatenate([X_bert, X_metadata, X_heuristico], axis=1)
        required_dim = model.input_shape[1]
        if X_final.shape[1] < required_dim:
            padding = np.zeros((1, required_dim - X_final.shape[1]))
            X_final = np.concatenate([X_final, padding], axis=1)
        elif X_final.shape[1] > required_dim:
            X_final = X_final[:, :required_dim]

        probabilidad = float(model.predict(X_final, verbose=0)[0][0])
        clasificacion = "Verdadera" if probabilidad >= 0.5 else "Falsa"

        contexto = {
            'pred_keras': f'{clasificacion} ({probabilidad:.2%})',
            'pred_mlp': 'MLP OFF'
        }
        return render(request, 'index.html', contexto)

    return render(request, 'index.html')
