import pandas as pd
import numpy as np
import re
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import joblib
from tensorflow.keras.models import load_model
from collections import Counter
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from django.shortcuts import render

import string
import nltk
from nltk.corpus import stopwords
import spacy
# --- Pega aquí las funciones que definiste para:
# construir_diccionario_emocional, calcular_emotividad, emocion_predominante,
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
# limpiar_texto, feature_engineering, generar_embedding_heuristico
import string
import nltk
from nltk.corpus import stopwords


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
# (Ya las tienes definidas en tu código, solo hay que usarlas.)
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

def generar_embeddings(df, ohe=None):
    col_num = ['longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
               'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido']

    if ohe is None:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        autor_encoded = ohe.fit_transform(df[['autor']])
    else:
        autor_encoded = ohe.transform(df[['autor']])

    df_autor = pd.DataFrame(autor_encoded, columns=ohe.get_feature_names_out(['autor']), index=df.index)
    df_embed = pd.concat([df[col_num + ['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos']], df_autor], axis=1)
    return df_embed.to_numpy(), ohe
# Limpieza básica
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, textos):
        self.textos = textos

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        return self.textos[idx]

def generar_embedding_bert(df, tokenizer, model, columna_texto='contenido_limpio', batch_size=32):
    device = torch.device('cpu')  # Fuerza uso de CPU
    model.to(device)
    model.eval()

    dataset = TextDataset(df[columna_texto].tolist())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generando embeddings BERT (CPU)"):
            encoded = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token

            embeddings.append(cls_embeddings.cpu())

    return torch.cat(embeddings, dim=0).numpy()

# Carga modelo y recursos
model = load_model('mejor_modelo_val_loss.h5')
scaler_metadata = joblib.load('scaler_metadata.pkl')
scaler_bert = joblib.load('scaler_bert.pkl')
scaler_heuristic = joblib.load('scaler_heuristic.pkl')
ohe_autor = joblib.load('encoder_autor.pkl')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model_bert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')

df_diccionario = pd.read_csv('diccionario.csv')

def predecir_noticia(noticia):
    df_noticia = pd.DataFrame(noticia)

    # 1) Embeddings heurísticos
    X_heuristico = generar_embedding_heuristico(df_noticia.copy(), df_diccionario)
    X_heuristico = scaler_heuristic.transform(X_heuristico)

    # 2) Metadata (feature engineering + escalado)
    df_meta, _ = feature_engineering(df_noticia.copy())

    col_num = ['longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
               'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido']
    X_num = df_meta[col_num].values
    X_num_scaled = scaler_metadata.transform(X_num)

    X_ciclicas = df_meta[['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos']].values
    X_autor = ohe_autor.transform(df_meta[['autor']])

    X_metadata = np.concatenate([X_num_scaled, X_ciclicas, X_autor], axis=1)

    # 3) Embedding BERT
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

    # 4) Concatenar todo y ajustar dimensiones
    X_final = np.concatenate([X_bert, X_metadata, X_heuristico], axis=1)
    required_dim = model.input_shape[1]

    if X_final.shape[1] < required_dim:
        padding = np.zeros((1, required_dim - X_final.shape[1]))
        X_final = np.concatenate([X_final, padding], axis=1)
    elif X_final.shape[1] > required_dim:
        X_final = X_final[:, :required_dim]

    # 5) Predecir
    prob = float(model.predict(X_final, verbose=0)[0][0])
    etiqueta = "Verdadera" if prob >= 0.5 else "Falsa"

    # 6) Mostrar resultados y heurísticas explicativas
    print(f"\nProbabilidad de ser verdadera: {prob:.2%}")
    print(f"Clasificación: {etiqueta}")

    titulo = df_noticia['titulo'].iloc[0]
    contenido = df_noticia['contenido'].iloc[0]
    autor = df_noticia['autor'].iloc[0]

    from collections import Counter

    coincidencias = len(set(re.findall(r'\b\w+\b', titulo.lower())).intersection(
                       set(re.findall(r'\b\w+\b', contenido.lower()))))
    print(f"- Coincidencias título-contenido: {coincidencias} palabras clave")

    if autor.lower() == "desconocido":
        print("- ⚠️ Autor desconocido (reduce credibilidad)")

    longitud = len(contenido.split())
    if longitud < 150:
        print(f"- Contenido corto ({longitud} palabras)")
    elif longitud > 800:
        print(f"- Contenido muy largo ({longitud} palabras)")
    else:
        print(f"- Longitud adecuada ({longitud} palabras)")

    exclamaciones = contenido.count('!')
    if exclamaciones > 3:
        print(f"- ⚠️ {exclamaciones} signos de exclamación (posible sensacionalismo)")

    dic_emocional = construir_diccionario_emocional(df_diccionario)
    emotividad = calcular_emotividad(contenido, dic_emocional)
    emocion = emocion_predominante(contenido, dic_emocional)
    print(f"- Score de emotividad: {emotividad:.2f} (emoción predominante: {emocion})")

# --- Ejemplo de uso ---
noticia_ejemplo = {
    'titulo': ['Expulsan a una joven de 16 años de un instituto en Cataluña por hablar español… ¡en una clase de lengua española!'],
    'autor': ['alertadigital'],
    'fecha': ['05-04-2019'],
    'contenido': ["""
Expulsan a una joven de 16 años de un instituto en Cataluña por hablar español… ¡en una clase de lengua española!

Una madre ha denunciado ante el Defensor del Pueblo catalán «Sindic de Greuges» la expulsión de su hija de 16 años de un centro de enseñanzas medias de la provincia de Barcelona por hablar en español.

Según afirma la madre de la joven expulsada, la clase era de español; aún así, la profesora la estaba impartiendo en catalán y exigió a los alumnos que hablaran en esa lengua. La respuesta de la estudiante fue rotunda: “No voy a hablar en catalán porque estamos en clase de español”.

La reacción de la dirección del centro no se hizo esperar: la joven de 16 años ha sido expulsada hasta después de Semana Santa.

Los hechos, que se produjeron ayer jueves 4 de abril, prueba, en opinión de la denunciante, la persecución ideológica en los centros educativos catalanes.

“No se puede consentir que porque nuestra lengua no sea la catalana, nos miren y nos traten como gentuza y ni siquiera quieran atendernos”, denuncia la madre de la alumna expulsada. Y añade: “No podemos permitir que en el siglo XXI te impongan el idioma que debes hablar, cuando el idioma más hablado en Cataluña, mal que le pese, es el español”.
    """]
}


# ⚙️ Carga una sola vez
model = load_model('mejor_modelo_val_loss.h5')
scaler_metadata = joblib.load('scaler_metadata.pkl')
scaler_bert = joblib.load('scaler_bert.pkl')
scaler_heuristic = joblib.load('scaler_heuristic.pkl')
ohe_autor = joblib.load('encoder_autor.pkl')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model_bert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
df_diccionario = pd.read_csv('diccionario.csv')
def predict_noticia_view(request):
    if request.method == "POST":
        titulo = request.POST.get('titulo')
        autor = request.POST.get('autor')
        fecha = request.POST.get('fecha')
        contenido = request.POST.get('contenido')

        df_noticia = pd.DataFrame([{
            'titulo': titulo,
            'autor': autor,
            'fecha': fecha,
            'contenido': contenido
        }])

        # === Igual que antes ===
        X_heuristico = generar_embedding_heuristico(df_noticia.copy(), df_diccionario)
        X_heuristico = scaler_heuristic.transform(X_heuristico)

        df_meta, _ = feature_engineering(df_noticia.copy())
        col_num = ['longitud', 'longitud_titulo', 'exclamaciones', 'interrogaciones',
                   'palabras_sensacionalistas', 'noticias_autor', 'relacion_titulo_contenido']
        X_num = df_meta[col_num].values
        X_num_scaled = scaler_metadata.transform(X_num)
        X_ciclicas = df_meta[['mes_sin', 'mes_cos', 'dia_sin', 'dia_cos']].values
        X_autor = ohe_autor.transform(df_meta[['autor']])
        X_metadata = np.concatenate([X_num_scaled, X_ciclicas, X_autor], axis=1)

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

        X_final = np.concatenate([X_bert, X_metadata, X_heuristico], axis=1)
        required_dim = model.input_shape[1]

        if X_final.shape[1] < required_dim:
            padding = np.zeros((1, required_dim - X_final.shape[1]))
            X_final = np.concatenate([X_final, padding], axis=1)
        elif X_final.shape[1] > required_dim:
            X_final = X_final[:, :required_dim]

        prob = float(model.predict(X_final, verbose=0)[0][0])
        etiqueta = "Verdadera" if prob >= 0.5 else "Falsa"

        # === Calcula explicativos ===
        coincidencias = len(set(re.findall(r'\b\w+\b', titulo.lower())).intersection(
                             set(re.findall(r'\b\w+\b', contenido.lower()))))
        longitud = len(contenido.split())

        dic_emocional = construir_diccionario_emocional(df_diccionario)
        emotividad = calcular_emotividad(contenido, dic_emocional)
        emocion = emocion_predominante(contenido, dic_emocional)

        # === Contexto para el template ===
        contexto = {
            'titulo': titulo,
            'autor': autor,
            'fecha': fecha,
            'contenido': contenido,
            'pred_keras': f"{etiqueta} ({prob*100:.2f}%)",
            'coincidencias': coincidencias,
            'longitud': longitud,
            'score_emotividad': f"{emotividad:.2f}",
            'emocion': emocion,
        }

        return render(request, 'index.html', contexto)

    else:
        return render(request, 'index.html')