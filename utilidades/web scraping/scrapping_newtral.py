import requests
from bs4 import BeautifulSoup
import time
import json
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import csv
import pandas as pd
from urllib.parse import unquote


class NewtralScraper:
    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')

        self.driver = None
        self.articles_data = []

        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]

    def init_driver(self):
        try:
            self.driver = webdriver.Chrome(options=self.chrome_options)
            return True
        except Exception as e:
            print(f"Error al inicializar el driver: {e}")
            return False

    def load_all_articles(self, url):
        if not self.driver:
            if not self.init_driver():
                return []

        print("Cargando página principal...")
        self.driver.get(url)
        time.sleep(3)

        articles_links = set()
        load_more_clicks = 0

        while True:
            try:
                load_more_btn = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "vog-newtral-es-verification-list-load-more-btn"))
                )
                self.driver.execute_script("arguments[0].click();", load_more_btn)
                load_more_clicks += 1
                print(f"✅ Clic #{load_more_clicks} en 'Cargar más'")
                time.sleep(2)

            except TimeoutException:
                print("🚫 No se encontró el botón 'Cargar más' (fin del scroll)")
                break
            except Exception as e:
                print(f"⚠️ Error al hacer clic en 'Cargar más': {e}")
                break

        print("🔍 Extrayendo enlaces de artículos...")
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        article_links = soup.find_all('a', class_='card-title-link')

        for link in article_links:
            href = link.get('href')
            if href and href.startswith('https://www.newtral.es/'):
                articles_links.add(href)

        print(f"📌 Total de artículos encontrados: {len(articles_links)}")
        print(f"🖱️ Total de clics en 'Cargar más': {load_more_clicks}")
        return list(articles_links)

    def clean_text(self, text):
        if not text:
            return ""

        text = unquote(text)
        replacements = {
            "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
            "Ã±": "ñ", "Ã¼": "ü", "Ã‰": "É", "Ã‘": "Ñ", "â€œ": '"',
            "â€": '"', "â€“": "-", "â€¢": "-", "â€¦": "..."
        }

        for wrong, right in replacements.items():
            text = text.replace(wrong, right)

        text = ' '.join(text.split())
        return text.strip()

    def scrape_article_details(self, article_url):
        """Scrape con reintentos manuales y user-agent rotatorio"""
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                headers = {'User-Agent': random.choice(self.user_agents)}
                response = requests.get(article_url, headers=headers, timeout=10)
                response.encoding = 'utf-8'
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                article_data = {
                    'titulo': '',
                    'autor': '',
                    'fecha': '',
                    'contenido': '',
                    'veracidad': '',
                    'url': article_url
                }

                title_elem = soup.find('h1', class_='post-title-1')
                if title_elem:
                    article_data['titulo'] = self.clean_text(title_elem.get_text())

                author_elem = soup.find('a', class_='author-link')
                if author_elem:
                    article_data['autor'] = self.clean_text(author_elem.get_text())

                date_elem = soup.find('div', class_='post-date')
                if date_elem:
                    article_data['fecha'] = self.clean_text(date_elem.get_text())

                content_elem = soup.find('section', class_='section-post-content')
                if content_elem:
                    paragraphs = content_elem.find_all(['p', 'li', 'h2', 'h3', 'h4'])
                    content_text = ' '.join([self.clean_text(p.get_text()) for p in paragraphs])
                    article_data['contenido'] = content_text

                verification_elem = soup.find('div', class_='post-category-1')
                if verification_elem:
                    text = verification_elem.get_text()
                    article_data['veracidad'] = self.clean_text(text.split('|')[-1].strip())

                return article_data

            except Exception as e:
                print(f"⚠️ Intento {attempt} fallido para {article_url}: {e}")
                if attempt < max_retries:
                    wait_time = random.uniform(3, 6)
                    print(f"⏳ Reintentando en {wait_time:.1f} segundos...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Fallo definitivo al procesar {article_url}")
                    return None

    def scrape_all(self, base_url):
        print("Iniciando scraping de Newtral...")
        article_links = self.load_all_articles(base_url)

        if not article_links:
            print("No se encontraron artículos")
            return []

        print(f"Procesando {len(article_links)} artículos...")

        for i, link in enumerate(article_links, 1):
            print(f"Procesando artículo {i}/{len(article_links)}: {link}")
            article_data = self.scrape_article_details(link)
            if article_data:
                self.articles_data.append(article_data)

            # Guardado intermedio cada 50 artículos
            if i % 200 == 0:
                self.save_to_json(f'backup_{i}.json')
                print(f"💾 Backup guardado: backup_{i}.json")

            # Pausa aleatoria para simular navegación humana
            time.sleep(random.uniform(1, 3))

        return self.articles_data

    def save_to_json(self, filename='newtral_articles.json'):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.articles_data, f, ensure_ascii=False, indent=2)
        print(f"Datos guardados en {filename}")

    def save_to_csv(self, filename='newtral_articles.csv'):
        if not self.articles_data:
            print("No hay datos para guardar")
            return

        fieldnames = ['titulo', 'autor', 'fecha', 'contenido', 'veracidad', 'url']

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for article in self.articles_data:
                writer.writerow(article)
        print(f"Datos guardados en {filename}")

    def save_to_excel(self, filename='newtral_articles.xlsx'):
        if not self.articles_data:
            print("No hay datos para guardar")
            return

        df = pd.DataFrame(self.articles_data)
        df = df[['titulo', 'autor', 'fecha', 'veracidad', 'contenido', 'url']]

        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Artículos')

        workbook = writer.book
        worksheet = writer.sheets['Artículos']

        text_format = workbook.add_format({'text_wrap': True})

        worksheet.set_column('A:A', 50, text_format)
        worksheet.set_column('B:B', 25)
        worksheet.set_column('C:C', 20)
        worksheet.set_column('D:D', 15)
        worksheet.set_column('E:E', 80, text_format)
        worksheet.set_column('F:F', 50)

        writer.close()
        print(f"Datos guardados en {filename}")

    def close(self):
        if self.driver:
            self.driver.quit()


def main():
    url = "https://www.newtral.es/zona-verificacion/todo/"
    scraper = NewtralScraper()

    try:
        articles = scraper.scrape_all(url)
        print(f"\nScraping completado. Total de artículos procesados: {len(articles)}")

        scraper.save_to_json()
        scraper.save_to_csv()
        scraper.save_to_excel()

        if articles:
            print("\nPrimeros 3 artículos:")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n{i}. {article['titulo']}")
                print(f"   Autor: {article['autor']}")
                print(f"   Fecha: {article['fecha']}")
                print(f"   Veracidad: {article['veracidad']}")
                print(f"   URL: {article['url']}")
                print(f"   Contenido (inicio): {article['contenido'][:100]}...")

    except KeyboardInterrupt:
        print("\nScraping interrumpido por el usuario")
    finally:
        scraper.close()


if __name__ == "__main__":
    main()