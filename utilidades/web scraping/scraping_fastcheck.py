from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configura Selenium
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
driver = webdriver.Chrome(options=options)

BASE_URL = "https://www.fastcheck.cl/category/chequeo/"
MAX_THREADS = 10  # Puedes subirlo hasta 20-30 si tu red es buena

def obtener_links_totales():
    """Recorre todas las páginas y junta todos los links."""
    all_links = set()
    driver.get(BASE_URL)
    time.sleep(2)

    page = 1
    while True:
        print(f"📄 Revisando página {page}")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        anchors = soup.select("a[href^='https://www.fastcheck.cl/20']")
        page_links = {a['href'] for a in anchors}
        all_links.update(page_links)

        try:
            next_button = driver.find_element(By.LINK_TEXT, 'Siguiente')
            next_button.click()
            page += 1
            time.sleep(2)
        except:
            print("🚩 No se encontró el botón 'Siguiente'. Fin de las páginas.")
            break

    driver.quit()
    print(f"✅ Total de noticias encontradas: {len(all_links)}")
    return list(all_links)

def obtener_veredicto(titulo):
    if '#falso' in titulo.lower():
        return 'FALSO'
    elif '#engañoso' in titulo.lower():
        return 'ENGAÑOSO'
    elif '#real' in titulo.lower():
        return 'VERDADERO'
    else:
        return 'NO ENCONTRADO'

def scrape_noticia(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        titulo_tag = soup.select_one('h1')
        titulo = titulo_tag.text.strip() if titulo_tag else 'No encontrado'

        autor_tag = soup.select_one('.elementor-author-box__name')
        autor = autor_tag.text.strip() if autor_tag else 'No encontrado'

        fecha_tag = soup.select_one('time, .date')
        fecha = fecha_tag.text.strip() if fecha_tag else 'No encontrada'

        veredicto = obtener_veredicto(titulo)

        contenido = ' '.join(p.text.strip() for p in soup.select('article p'))

        print(f"✅ Scrapeado: {titulo}")

        return {
            'titulo': titulo,
            'autor': autor,
            'fecha': fecha,
            'veredicto': veredicto,
            'contenido': contenido,
            'url': url
        }
    except Exception as e:
        print(f"⚠️ Error en {url}: {e}")
        return None

def main():
    links = obtener_links_totales()
    noticias = []

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        future_to_url = {executor.submit(scrape_noticia, url): url for url in links}
        for future in as_completed(future_to_url):
            data = future.result()
            if data:
                noticias.append(data)

    df = pd.DataFrame(noticias)
    df.to_csv('fastcheck_masivo.csv', index=False, encoding='utf-8')
    print("\n✅ Scraping completado. Datos guardados en 'fastcheck_masivo.csv'.")

if __name__ == "__main__":
    main()
