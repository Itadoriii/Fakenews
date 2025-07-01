import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder

# Asegúrate de que ya hiciste nltk.download('stopwords') antes de usar esto
stop_words = set(stopwords.words('spanish'))
nlp = spacy.load("es_core_news_sm", disable=["ner", "tagger"])

verbos_modales = ['debería', 'podría', 'tendría', 'habría', 'puede', 'pueden', 'debe', 'deben', 'sería']
terminos_generalizadores = ['siempre', 'nunca', 'todos', 'nadie', 'jamás', 'ninguno', 'cualquiera']
verbos_opinion = ['creo', 'considero', 'opino', 'pienso', 'me parece', 'supongo', 'siento', 'estimo']
lexico_polarizado = ['horrible', 'excelente', 'terrible', 'maravilloso', 'desastroso', 'perfecto', 'abominable', 'magnífico']

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

def aplicar_emotividad(df, df_diccionario):
    diccionario_emocional = construir_diccionario_emocional(df_diccionario)
    df['score_emotividad'] = df['contenido'].apply(lambda x: calcular_emotividad(x, diccionario_emocional))
    return df[['score_emotividad']]

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
    return len(set(palabras)) / len(palabras) if palabras else 0

def aplicar_heuristicas_linguisticas(df):
    df['incertidumbre'] = df['contenido'].apply(lambda x: frecuencia_heuristica(x, verbos_modales + terminos_generalizadores))
    df['subjetividad'] = df['contenido'].apply(lambda x: frecuencia_heuristica(x, verbos_opinion + lexico_polarizado))
    df['diversidad_lexica'] = df['contenido'].apply(diversidad_lexica)
    return df[['incertidumbre', 'subjetividad', 'diversidad_lexica']]

def aplicar_patrones_sintacticos(df):
    resultados = []
    for doc in nlp.pipe(df['contenido'], batch_size=32):
        oraciones = list(doc.sents)
        total_palabras = len([token.text for token in doc if token.is_alpha])
        total_palabras_mayus = len([token.text for token in doc if token.is_upper])
        total_exclamaciones = doc.text.count('!')
        frases_cortas = sum(1 for s in oraciones if len(s.text.split()) <= 7)
        porcentaje_mayus = total_palabras_mayus / total_palabras if total_palabras else 0
        frases_cortas_porcentaje = frases_cortas / len(oraciones) if oraciones else 0
        resultados.append({
            "porcentaje_mayusculas": porcentaje_mayus,
            "exclamaciones_totales": total_exclamaciones,
            "frases_cortas": frases_cortas,
            "frases_cortas_%": frases_cortas_porcentaje
        })
    return pd.DataFrame(resultados, index=df.index)

def consistencia_jaccard(titulo, contenido):
    titulo = limpiar_texto_heuristica(titulo)
    contenido = limpiar_texto_heuristica(contenido)
    set_titulo = set([p for p in titulo.split() if p not in stop_words])
    set_contenido = set([p for p in contenido.split() if p not in stop_words])
    return len(set_titulo & set_contenido) / len(set_titulo | set_contenido) if set_titulo and set_contenido else 0

def aplicar_relacion_titulo_cuerpo(df):
    df['consistencia_titulo_cuerpo'] = df.apply(lambda row: consistencia_jaccard(row['titulo'], row['contenido']), axis=1)
    return df[['consistencia_titulo_cuerpo']]

def generar_embedding_heuristico(df, df_diccionario):
    emotividad = aplicar_emotividad(df.copy(), df_diccionario)
    heuristicas = aplicar_heuristicas_linguisticas(df.copy())
    patrones = aplicar_patrones_sintacticos(df.copy())
    relacion = aplicar_relacion_titulo_cuerpo(df.copy())
    heuristico_df = pd.concat([emotividad, heuristicas, patrones, relacion], axis=1)
    return heuristico_df.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()

from sklearn.preprocessing import StandardScaler

def feature_engineering(df):
    df['longitud'] = df['contenido'].apply(lambda x: len(str(x).split()))
    df['longitud_titulo'] = df['titulo'].apply(lambda x: len(str(x).split()))
    df['exclamaciones'] = df['contenido'].apply(lambda x: str(x).count('!'))
    df['interrogaciones'] = df['contenido'].apply(lambda x: str(x).count('?'))
    sensacionalistas = ['increíble', 'urgente', 'impactante', 'viral', 'escándalo']
    df['palabras_sensacionalistas'] = df['contenido'].apply(lambda x: sum(x.lower().count(p) for p in sensacionalistas))
    df['fecha'] = pd.to_datetime(df['fecha'], format='%d-%m-%Y', errors='coerce')
    df['dia_semana'] = df['fecha'].dt.day_name()
    df['mes'] = df['fecha'].dt.month
    df['noticias_autor'] = df['autor'].map(df['autor'].value_counts())
    df['relacion_titulo_contenido'] = df.apply(lambda r: r['longitud_titulo'] / r['longitud'] if r['longitud'] > 0 else 0, axis=1)
    autor_freq = df['autor'].value_counts()
    df['autor'] = df['autor'].apply(lambda x: x if x in autor_freq[autor_freq > 20].index else 'Otro')
    df['mes_sin'] = np.sin(2 * np.pi * df['mes'].fillna(0) / 12)
    df['mes_cos'] = np.cos(2 * np.pi * df['mes'].fillna(0) / 12)
    dias = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['dia_semana_num'] = df['dia_semana'].map(dias).fillna(0).astype(int)
    df['dia_sin'] = np.sin(2 * np.pi * df['dia_semana_num'] / 7)
    df['dia_cos'] = np.cos(2 * np.pi * df['dia_semana_num'] / 7)
    columnas_num = ['longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
                    'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido']
    df[columnas_num] = df[columnas_num].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    df[columnas_num] = scaler.fit_transform(df[columnas_num])
    return df, scaler
 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def generar_embeddings(df):
    col_num = ['longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
               'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido']
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    autor_encoded = ohe.fit_transform(df[['autor']])
    df_autor = pd.DataFrame(autor_encoded, columns=ohe.get_feature_names_out(['autor']), index=df.index)
    df_embed = pd.concat([df[col_num + ['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos']], df_autor], axis=1)
    return df_embed.to_numpy(), ohe
