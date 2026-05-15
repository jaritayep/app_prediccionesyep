"""
scraper_oddschecker_tabs.py
==========
Estrategia Anti-Bloqueo:
1. Tiempos de espera mucho más largos y aleatorios (simulando lectura humana).
2. Abre cada partido en una pestaña nueva, extrae la data, cierra la pestaña 
   y descansa antes de abrir el siguiente partido.
"""

import json
import logging
import time
import random
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# --- 1. TU BASE DE DATOS ---
PARTIDOS_DB = [
    ("Premier League", "Aston Villa", "Liverpool"),
    # Agrega aquí los de mañana, por ejemplo:
    # ("Premier League", "Manchester City", "West Ham"),
    # ("La Liga", "Real Madrid", "Real Betis")
]

RUTAS_LIGAS = {
    "Premier League": "inglaterra/premier-league",
    "La Liga": "espana/la-liga",
    "Bundesliga": "alemania/bundesliga",
    "Serie A": "italia/serie-a",
    "Ligue 1": "francia/ligue-1"
}

OUTPUT_DIR = Path("odds_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def get_driver():
    opts = Options()
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    # Este comando es clave para quitar la bandera de "robot" de Selenium
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option('useAutomationExtension', False)
    opts.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
    
    driver = webdriver.Chrome(options=opts)
    # Inyectamos un script extra para engañar a los detectores de bots
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
      "source": """
        Object.defineProperty(navigator, 'webdriver', {
          get: () => undefined
        })
      """
    })
    return driver

def format_url(liga, local, visita):
    ruta_liga = RUTAS_LIGAS.get(liga, "")
    local_url = local.lower().replace(" ", "-")
    visita_url = visita.lower().replace(" ", "-")
    return f"https://www.oddschecker.com/es/futbol/{ruta_liga}/{local_url}-v-{visita_url}"

def pausa_humana(min_seg, max_seg):
    """Pausa aleatoria para no parecer un robot exacto"""
    tiempo = random.uniform(min_seg, max_seg)
    time.sleep(tiempo)

def aceptar_cookies(driver):
    try:
        botones_cookies = driver.find_elements(By.XPATH, "//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'aceptar') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept')]")
        if botones_cookies:
            driver.execute_script("arguments[0].click();", botones_cookies[0])
            pausa_humana(1.5, 2.5)
    except:
        pass

def apretar_mostrar_mas(driver):
    try:
        botones_mas = driver.find_elements(By.XPATH, "//*[contains(text(), 'Mostrar más')]")
        if botones_mas:
            log.info(f"      Desplegando {len(botones_mas)} botones de 'Mostrar más'...")
            for btn in botones_mas:
                driver.execute_script("arguments[0].click();", btn)
                pausa_humana(0.8, 1.5) # Pausa entre cada clic de "Mostrar más"
    except Exception as e:
        pass

def extraer_datos_html(html):
    """Aspiradora extrema: extrae todo el texto que parezca una cuota o un mercado"""
    soup = BeautifulSoup(html, "html.parser")
    resultados = []
    
    # Buscamos en etiquetas comunes donde Oddschecker guarda la información
    # (divs, secciones, listas y botones)
    elementos = soup.find_all(["div", "section", "li", "button"])
    
    for el in elementos:
        # Convertimos las clases y data-testids a texto para buscar palabras clave
        clases_str = " ".join(el.get("class", [])).lower()
        testid_str = el.get("data-testid", "").lower()
        
        # Si el "contenedor" huele a mercado de apuestas...
        if any(clave in clases_str or clave in testid_str for clave in ["market", "odd", "bet", "selection", "participant"]):
            
            # Extraemos el texto separando los elementos con " | "
            texto_limpio = el.get_text(separator=" | ", strip=True)
            
            # Solo lo guardamos si tiene más de 10 caracteres y contiene algún número (una cuota)
            if texto_limpio and len(texto_limpio) > 10 and any(char.isdigit() for char in texto_limpio):
                # Filtramos basura como el menú de navegación
                if "Inicia sesión" not in texto_limpio and "Regístrate" not in texto_limpio:
                    resultados.append(texto_limpio)
    
    # Si la búsqueda inteligente no sacó nada, aspiramos todos los botones que tengan números
    if not resultados:
        botones = soup.find_all("button")
        for btn in botones:
            txt = btn.get_text(separator=" ", strip=True)
            if txt and any(char.isdigit() for char in txt):
                resultados.append(f"[Botón]: {txt}")
                
    # Eliminamos duplicados (ya que un div dentro de otro div captura lo mismo)
    resultados_unicos = []
    for r in resultados:
        if r not in resultados_unicos:
            resultados_unicos.append(r)
            
    # Retornamos la data cruda. Si esto funciona, verás los mercados y las cuotas en el JSON.
    return {"datos_crudos": resultados_unicos[:60]}

def scrape_partido(driver, url, local, visita):
    log.info(f"📍 Cargando URL: {local} vs {visita}")
    driver.get(url)
    
    # Pausa inicial larga simulando que un humano lee la página al entrar
    pausa_humana(5.0, 7.0) 
    aceptar_cookies(driver)
    
    datos_partido = {"home": local, "away": visita, "url": url, "markets": {}}
    
    # Simulamos un scroll humano (bajamos un poco)
    driver.execute_script("window.scrollBy(0, 300);")
    pausa_humana(1.0, 2.0)
    
    log.info("  👉 Apretando 'Mercados de ganador'")
    try:
        btn_ganador = driver.find_element(By.XPATH, "//span[contains(text(), 'Mercados de ganador')] | //button[contains(text(), 'Mercados de ganador')] | //a[contains(text(), 'Mercados de ganador')]")
        driver.execute_script("arguments[0].click();", btn_ganador)
        pausa_humana(3.0, 5.0) # Tiempo para "leer" las cuotas del ganador
        datos_partido["markets"]["Ganador"] = extraer_datos_html(driver.page_source)
    except:
        log.warning("    ❌ No se encontró 'Mercados de ganador'.")

    log.info("  👉 Apretando 'Apuestas de estadísticas'")
    try:
        btn_estadisticas = driver.find_element(By.XPATH, "//span[contains(text(), 'Apuestas de estadísticas')] | //button[contains(text(), 'Apuestas de estadísticas')] | //a[contains(text(), 'Apuestas de estadísticas')]")
        driver.execute_script("arguments[0].click();", btn_estadisticas)
        pausa_humana(4.0, 6.0) # Esperamos que cambie la tabla de stats
        
        apretar_mostrar_mas(driver)
        datos_partido["markets"]["Estadisticas"] = extraer_datos_html(driver.page_source)
    except:
        log.warning("    ❌ No se encontró 'Apuestas de estadísticas'.")
        
    return datos_partido

def main():
    driver = get_driver()
    resultados = []
    
    try:
        log.info("Iniciando Scraper Oddschecker (Modo Cauteloso)...")
        
        # 1. Abrimos una pestaña "Base" (un sitio neutral para no levantar sospechas)
        driver.get("https://www.google.com")
        pausa_humana(2.0, 3.0)
        
        for liga, local, visita in PARTIDOS_DB:
            url = format_url(liga, local, visita)
            
            log.info(f"Abriendo pestaña nueva para {local}...")
            # Abrimos pestaña en blanco
            driver.execute_script("window.open('');")
            # Saltamos a la última pestaña abierta (la nueva)
            driver.switch_to.window(driver.window_handles[-1])
            
            # Scrapeamos
            datos = scrape_partido(driver, url, local, visita)
            resultados.append(datos)
            
            log.info("Cerrando pestaña y descansando...")
            # Cerramos esa pestaña específica
            driver.close()
            
            # Volvemos a la pestaña base (Google)
            driver.switch_to.window(driver.window_handles[0])
            
            # Pausa muy larga entre partidos (clave para evitar el ban)
            pausa_humana(6.0, 10.0) 
            
    finally:
        driver.quit()
        
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "latest.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    log.info("🎯 Proceso terminado con seguridad. Datos guardados en 'latest.json'.")

if __name__ == "__main__":
    main()