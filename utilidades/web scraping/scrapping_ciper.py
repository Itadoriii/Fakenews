import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# User-Agent para que el servidor crea que somos un navegador real
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
}

noticias = []
num_paginas = 73  # Total de páginas a scrapear

for pagina in range(2, num_paginas + 1):
    url = f'https://www.ciperchile.cl/category/actualidad/page/{pagina}/'
    print(f'Scrapeando {url}')
    
    # AQUI SE AGREGA EL HEADERS
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Buscar los artículos correctamente
    articulos = soup.find_all('div', class_='col-md-4 col-lg-3 mb-3')

    if not articulos:
        print(f'No se encontraron artículos en la página {pagina}')
        continue

    for articulo in articulos:
        try:
            enlace = articulo.find('a')['href']
        except:
            continue

        print(f'Scrapeando noticia: {enlace}')
        
        # AQUI TAMBIEN SE AGREGA EL HEADERS
        noticia_response = requests.get(enlace, headers=headers)
        noticia_soup = BeautifulSoup(noticia_response.content, 'html.parser')
        
        try:
            titulo = noticia_soup.find('h1', class_='article-big-text__title').text.strip()
        except:
            titulo = ''
        
        try:
            fecha = noticia_soup.find('p', class_='article-big-text__date').text.strip()
        except:
            fecha = ''
        
        try:
            autor = noticia_soup.find('a', class_='text-underline').text.strip()
        except:
            autor = ''
        
        try:
            parrafos = noticia_soup.find_all('p', class_='texto-nota')
            contenido = ' '.join([p.text.strip() for p in parrafos])
        except:
            contenido = ''
        
        noticias.append({
            'titulo': titulo,
            'fecha': fecha,
            'autor': autor,
            'contenido': contenido,
            'url': enlace
        })
        
        time.sleep(1)  # Espera para no saturar el servidor

# Guardar en CSV
df = pd.DataFrame(noticias)
df.to_csv('noticias_ciperchile1.csv', index=False, encoding='utf-8-sig')

# Guardar en Excel (sin encoding)
df.to_excel('noticias_ciperchile_excel1.xlsx', index=False)

print('Scraping completado. Archivos guardados como noticias_ciperchile.csv y noticias_ciperchile_excel.xlsx.')

