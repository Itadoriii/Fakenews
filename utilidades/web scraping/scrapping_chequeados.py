import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

# URLs base
base_url = "https://chequeado.com/category/s5-articulos/c34-el-explicador/"
page_url = "https://chequeado.com/category/s5-articulos/c34-el-explicador/page/{}/"

# Ruta donde guardar el CSV
output_path = r"C:\Users\fabia\OneDrive\Escritorio\ScrappingTesis"
os.makedirs(output_path, exist_ok=True)

# Lista de datos
data = []

# Recorrer páginas
for page in range(1, 4):
    print(f"Procesando página {page}...")
    res = requests.get(page_url.format(page))
    soup = BeautifulSoup(res.text, 'html.parser')

    articles = soup.find_all('article')

    for art in articles:
        try:
            title_tag = art.find('h2')
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            link = title_tag.find('a')['href']
            resumen = art.find('p').get_text(strip=True)
            fecha = art.find('time').get_text(strip=True)

            # === Entrar a la noticia para sacar autor ===
            res_noticia = requests.get(link)
            soup_noticia = BeautifulSoup(res_noticia.text, 'html.parser')

            autor_tag = soup_noticia.find('a', class_='author vcard')  # alternativa
            if not autor_tag:
                autor_tag = soup_noticia.find('span', class_='autor')

            autor = autor_tag.get_text(strip=True) if autor_tag else 'No encontrado'

            # Guardar
            data.append({
                'titulo': title,
                'resumen': resumen,
                'fecha': fecha,
                'url': link,
                'autor': autor
            })

            time.sleep(1)

        except Exception as e:
            print(f"Error procesando artículo: {e}")
            continue

# Guardar en CSV
df = pd.DataFrame(data)
df.to_csv(os.path.join(output_path, "chequeado_dataset.csv"), index=False, encoding='utf-8-sig')

print(f"Scraping completado. Se guardaron {len(df)} artículos en {output_path}")
