"""
OddsPortal Advanced Scraper — Playwright
======================================================
Navega a las URLs de los partidos, extrae cuotas de cierre (1X2, AH, O/U)
y exporta en formato ancho.
"""

import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import re

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN Y URLs (Mundial 2026 - 17 al 23 de Junio)
# ─────────────────────────────────────────────────────────────────────────────

# Debes colocar las URLs directas de los partidos de OddsPortal que quieres raspar.
# Encontrar las URLs en la página de resultados es el primer paso.
MATCH_URLS = [
    "https://www.oddsportal.com/football/world/world-cup/norway-senegal-xxxxxx/",
    # Agrega aquí el resto de las URLs de los partidos del 17 al 23...
]

OUTPUT_CSV = "worldcup_2026_oddsportal.csv"

# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA DE EXTRACCIÓN
# ─────────────────────────────────────────────────────────────────────────────

async def extract_market_data(page, match_data, tab_name, market_prefix):
    """Hace clic en la pestaña del mercado y extrae las cuotas."""
    try:
        # Hacer clic en la pestaña del mercado (ej. "Asian Handicap", "Over/Under")
        await page.get_by_text(tab_name, exact=True).click(timeout=3000)
        # Esperar a que la tabla de cuotas se actualice
        await page.wait_for_selector('div.table-container', timeout=3000)
        
        # OddsPortal agrupa las cuotas por líneas (ej. +0.5, 2.5). Iteramos sobre ellas.
        rows = await page.locator('div.table-container > div.border-black-borders').all()
        
        for row in rows:
            text_content = await row.inner_text()
            lines = text_content.split('\n')
            
            if len(lines) >= 3:
                line_val = lines[0].strip() # Ej: "+0.5" o "2.5"
                # Limpiamos el valor de la línea para usarlo en el nombre de la columna
                clean_line = re.sub(r'[^\d\.\+\-]', '', line_val)
                
                odd_1 = lines[1].strip()
                odd_2 = lines[2].strip()
                
                # Asignamos al diccionario en formato ancho
                if "hdp" in market_prefix:
                    match_data[f"{market_prefix}_home_{clean_line}"] = odd_1
                    match_data[f"{market_prefix}_away_{clean_line}"] = odd_2
                elif "goles" in market_prefix:
                    match_data[f"{market_prefix}_{clean_line}_over"] = odd_1
                    match_data[f"{market_prefix}_{clean_line}_under"] = odd_2
                    
    except Exception as e:
        print(f"  ⚠ No se encontró el mercado '{tab_name}' o hubo un error: {e}")

async def scrape_match(browser, url):
    """Navega a un partido y extrae todos sus mercados."""
    print(f"Navegando a: {url}")
    page = await browser.new_page()
    
    match_data = {}
    
    try:
        # Ir a la página y esperar a que cargue el contenido principal
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_selector('h1', timeout=10000)
        
        # 1. Extraer Nombres de Equipos
        header_text = await page.locator('h1').inner_text()
        teams = header_text.split(' - ')
        match_data['liga'] = "FIFA - World Cup"
        match_data['home'] = teams[0].strip() if len(teams) > 0 else "Unknown"
        match_data['away'] = teams[1].strip() if len(teams) > 1 else "Unknown"
        
        print(f"  ⚽ Procesando: {match_data['home']} vs {match_data['away']}")

        # 2. Extraer 1X2 (Mercado por defecto al cargar la página)
        try:
            await page.wait_for_selector('div.table-container', timeout=5000)
            rows = await page.locator('div.table-container > div.border-black-borders').first.inner_text()
            odds = rows.split('\n')
            if len(odds) >= 3:
                match_data['1x2_home'] = odds[-3].strip()
                match_data['1x2_draw'] = odds[-2].strip()
                match_data['1x2_away'] = odds[-1].strip()
        except:
            print("  ⚠ No se pudo extraer 1X2.")

        # 3. Navegar y extraer Hándicap Asiático
        await extract_market_data(page, match_data, "Asian Handicap", "hdp")
        
        # 4. Navegar y extraer Over/Under (Goles)
        await extract_market_data(page, match_data, "Over/Under", "goles")

    except Exception as e:
        print(f"Error procesando {url}: {e}")
    finally:
        await page.close()
        
    return match_data

async def main():
    async with async_playwright() as p:
        # Usamos headles=False para depurar y ver si Cloudflare nos bloquea
        browser = await p.chromium.launch(headless=False) 
        
        all_matches = []
        for url in MATCH_URLS:
            data = await scrape_match(browser, url)
            if data:
                all_matches.append(data)
                
        await browser.close()
        
        if all_matches:
            df = pd.DataFrame(all_matches)
            # Ordenar columnas base al principio
            base_cols = ['liga', 'home', 'away']
            market_cols = sorted([c for c in df.columns if c not in base_cols])
            df = df[base_cols + market_cols]
            
            df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
            print(f"\n✅ Scraping completado. Guardado en {OUTPUT_CSV}")
        else:
            print("\n⚠ No se extrajeron datos.")

if __name__ == "__main__":
    asyncio.run(main())