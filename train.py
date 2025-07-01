
# Procesamiento de datos
import pandas as pd
import numpy as np
import re

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento y modelado clásico
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Deep Learning con Keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout

# Transformers y Deep Learning con PyTorch
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Barra de progreso
from tqdm import tqdm

df = pd.read_csv('Fakenews.csv')
df_diccionario = pd.read_csv('diccionario.csv')

df.info()
df['label'] = df['label'].apply(lambda x: 1 if str(x).strip().lower() == 'verdadera' else 0)
df_metadata = df.copy()
df_heuristica = df.copy()

from collections import Counter
import re

def construir_diccionario_emocional(df_diccionario):
    emociones = df_diccionario.columns[1:]  # Todas menos 'palabra'
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

def aplicar_emotividad(df_heuristica, df_diccionario):
    diccionario_emocional = construir_diccionario_emocional(df_diccionario)
    df_heuristica['score_emotividad'] = df_heuristica['contenido'].apply(
        lambda x: calcular_emotividad(x, diccionario_emocional)
    )
    df_heuristica['emocion_predominante'] = df_heuristica['contenido'].apply(
        lambda x: emocion_predominante(x, diccionario_emocional)
    )
    return df_heuristica[['score_emotividad']]

import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

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

def aplicar_heuristicas_linguisticas(df_heuristica):
    df_heuristica['incertidumbre'] = df_heuristica['contenido'].apply(
        lambda x: frecuencia_heuristica(x, verbos_modales + terminos_generalizadores)
    )
    df_heuristica['subjetividad'] = df_heuristica['contenido'].apply(
        lambda x: frecuencia_heuristica(x, verbos_opinion + lexico_polarizado)
    )
    df_heuristica['diversidad_lexica'] = df_heuristica['contenido'].apply(diversidad_lexica)
    return df_heuristica[['incertidumbre', 'subjetividad', 'diversidad_lexica']]

import spacy
import pandas as pd

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
    df_patrones = pd.DataFrame(resultados, index=df_heuristica.index)
    return df_patrones

import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

def limpiar_texto_jaccard(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    for c in string.punctuation:
        texto = texto.replace(c, "")
    return texto

def consistencia_jaccard(titulo, contenido):
    titulo = limpiar_texto_jaccard(titulo)
    contenido = limpiar_texto_jaccard(contenido)
    set_titulo = set([p for p in titulo.split() if p not in stop_words])
    set_contenido = set([p for p in contenido.split() if p not in stop_words])
    if len(set_titulo) == 0 or len(set_contenido) == 0:
        return 0
    interseccion = set_titulo.intersection(set_contenido)
    union = set_titulo.union(set_contenido)
    return len(interseccion) / len(union)

def aplicar_relacion_titulo_cuerpo(df_heuristica):
    df_heuristica['consistencia_titulo_cuerpo'] = df_heuristica.apply(
        lambda row: consistencia_jaccard(row['titulo'], row['contenido']), axis=1)
    return df_heuristica[['consistencia_titulo_cuerpo']]

def generar_embedding_heuristico(df, df_diccionario):
    emotividad = aplicar_emotividad(df.copy(), df_diccionario)
    heuristicas = aplicar_heuristicas_linguisticas(df.copy())
    patrones = aplicar_patrones_sintacticos(df.copy())
    relacion = aplicar_relacion_titulo_cuerpo(df.copy())

    heuristico_df = pd.concat([emotividad, heuristicas, patrones, relacion], axis=1)
    heuristico_df = heuristico_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return heuristico_df.to_numpy()

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def feature_engineering(df_metadata):
    # 1. Longitud del contenido
    df_metadata['longitud'] = df_metadata['contenido'].apply(lambda x: len(str(x).split()))

    # 2. Longitud del título
    df_metadata['longitud_titulo'] = df_metadata['titulo'].apply(lambda x: len(str(x).split()))

    # 3. Número de signos de exclamación e interrogación
    df_metadata['exclamaciones'] = df_metadata['contenido'].apply(lambda x: str(x).count('!'))
    df_metadata['interrogaciones'] = df_metadata['contenido'].apply(lambda x: str(x).count('?'))

    # 4. Palabras sensacionalistas
    sensacionalistas = ['increíble', 'urgente', 'impactante', 'viral', 'escándalo']
    df_metadata['palabras_sensacionalistas'] = df_metadata['contenido'].apply(
        lambda x: sum([x.lower().count(palabra) for palabra in sensacionalistas])
    )

    # 5. Día de la semana y mes de publicación
    df_metadata['fecha'] = pd.to_datetime(df_metadata['fecha'], format='%d-%m-%Y', errors='coerce')
    df_metadata['dia_semana'] = df_metadata['fecha'].dt.day_name()
    df_metadata['mes'] = df_metadata['fecha'].dt.month

    # 6. Frecuencia del autor
    autor_counts = df_metadata['autor'].value_counts().to_dict()
    df_metadata['noticias_autor'] = df_metadata['autor'].map(autor_counts)

    # 7. Relación longitud título / contenido evitando división por cero
    df_metadata['relacion_titulo_contenido'] = df_metadata.apply(
        lambda row: row['longitud_titulo'] / row['longitud'] if row['longitud'] > 0 else 0,
        axis=1
    )

    # 8. Agrupar autores poco frecuentes
    autor_freq = df_metadata['autor'].value_counts()
    autores_principales = autor_freq[autor_freq > 20].index
    df_metadata['autor'] = df_metadata['autor'].apply(lambda x: x if x in autores_principales else 'Otro')

    # 9. Codificación cíclica del mes (cuidado con NaN)
    df_metadata['mes_sin'] = np.sin(2 * np.pi * df_metadata['mes'].fillna(0) / 12)
    df_metadata['mes_cos'] = np.cos(2 * np.pi * df_metadata['mes'].fillna(0) / 12)

    # 10. Codificación cíclica del día de la semana
    dias = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df_metadata['dia_semana_num'] = df_metadata['dia_semana'].map(dias).fillna(0).astype(int)
    df_metadata['dia_sin'] = np.sin(2 * np.pi * df_metadata['dia_semana_num'] / 7)
    df_metadata['dia_cos'] = np.cos(2 * np.pi * df_metadata['dia_semana_num'] / 7)

    # 11. Escalar columnas numéricas
    columnas_numericas = [
        'longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
        'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido'
    ]

    # Reemplazar NaN e infinitos por 0 para evitar error en scaler
    df_metadata[columnas_numericas] = df_metadata[columnas_numericas].replace([np.inf, -np.inf], np.nan)
    df_metadata[columnas_numericas] = df_metadata[columnas_numericas].fillna(0)

    escalar = StandardScaler()
    df_metadata[columnas_numericas] = escalar.fit_transform(df_metadata[columnas_numericas])

    return df_metadata, escalar

def generar_embeddings(df_metadata):
    # Imputar valor nulo
    df_metadata['relacion_titulo_contenido'] = df_metadata['relacion_titulo_contenido'].fillna(
        df_metadata['relacion_titulo_contenido'].mean())

    # Agrupar autores poco frecuentes
    autor_freq = df_metadata['autor'].value_counts()
    autores_principales = autor_freq[autor_freq > 20].index
    df_metadata['autor'] = df_metadata['autor'].apply(lambda x: x if x in autores_principales else 'Otro')

    # Codificación cíclica de mes y día
    df_metadata['mes_sin'] = np.sin(2 * np.pi * df_metadata['mes'] / 12)
    df_metadata['mes_cos'] = np.cos(2 * np.pi * df_metadata['mes'] / 12)
    dias = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df_metadata['dia_semana_num'] = df_metadata['dia_semana'].map(dias)
    df_metadata['dia_sin'] = np.sin(2 * np.pi * df_metadata['dia_semana_num'] / 7)
    df_metadata['dia_cos'] = np.cos(2 * np.pi * df_metadata['dia_semana_num'] / 7)

    # Escalar columnas numéricas
    col_num = [
        'longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
        'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido'
    ]
    scaler = StandardScaler()
    df_metadata[col_num] = scaler.fit_transform(df_metadata[col_num])

    # One-Hot Encoding del autor
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    autor_encoded = ohe.fit_transform(df_metadata[['autor']])
    df_autor = pd.DataFrame(autor_encoded, columns=ohe.get_feature_names_out(['autor']), index=df_metadata.index)

    # Unir todo en un solo DataFrame de embeddings
    df_metadata_embed = pd.concat([
        df_metadata[col_num + ['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos']],
        df_autor
    ], axis=1)

    return df_metadata_embed,scaler, ohe

# Limpieza básica
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

df['contenido_limpio'] = df['contenido'].astype(str).apply(limpiar_texto)

def generar_embedding_bert(df, tokenizer, model, columna_texto='contenido_limpio'):
    model.eval()
    embeddings = []

    with torch.no_grad():
        for texto in tqdm(df[columna_texto], desc="Generando embeddings BERT"):
            inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(cls_embedding)

    return np.array(embeddings)

X_heuristico = generar_embedding_heuristico(df_heuristica, df_diccionario)

# Aplicar ingeniería de características
df_metadata = feature_engineering(df_metadata)

# Embedding metadatos
X_metadata = generar_embeddings(df_metadata)

#Cargar modelo BERT multilingüe
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

# Embedding texto
X_bert = generar_embedding_bert(df, tokenizer, model)

#argar modelo BERT multilingüe
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

# Embedding texto
X_bert = generar_embedding_bert(df, tokenizer, model)

X_final = np.concatenate([X_bert, X_metadata, X_heuristico], axis=1)
y = df['label'].values  # Asumiendo que tu columna objetivo se llama 'label'

#Separar datos
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Arquitectura del modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_final.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compilar
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión en test: {acc:.4f}")

# 5. Separar en train/test
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# 6. Entrenar MLP
mlp = MLPClassifier(hidden_layer_sizes=(256, 64), activation='relu', solver='adam',
                    max_iter=300, random_state=42, early_stopping=True, verbose=True)
mlp.fit(X_train, y_train)



import joblib


# 7. Evaluar
y_pred = mlp.predict(X_test)
print("\n--- RESULTADOS ---")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Guarda modelos
model.save("modelo_fakenews_keras.h5")
joblib.dump(mlp, "modelo_fakenews_mlp.pkl")

# Ajusta cómo llamas a tus funciones
# Corrección clave
df_metadata, escalar = feature_engineering(df_metadata)
X_metadata, scaler, ohe = generar_embeddings(df_metadata)

# Ahora los tienes:
joblib.dump(escalar, "scaler.pkl")
joblib.dump(ohe, "encoder.pkl")


# Opcional: guarda embeddings pesados
np.save("X_bert.npy", X_bert)
