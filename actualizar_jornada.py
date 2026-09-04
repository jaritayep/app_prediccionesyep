import requests
import sqlite3
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
import os

API_KEY = "c81aa18fa4974dda90812a83f1aec599"
LIGAS = ['PL', 'PD', 'BL1', 'SA', 'FL1', "CL"]  # Ligas principales
DB_NAME = "database_partidos.db"

# Estados que cuentan como "partido programado, aún no jugado".
# football-data.org pasa un partido de SCHEDULED a TIMED apenas confirma
# la hora exacta -> eso suele pasar justo en la semana previa, que es
# la ventana que estamos pidiendo. Filtrar solo por SCHEDULED en la URL
# hace perder justo los partidos más próximos.
ESTADOS_VALIDOS = {'SCHEDULED', 'TIMED'}


def actualizar_partidos_semana():
    headers = {'X-Auth-Token': API_KEY}
    all_matches = []

    # Usamos UTC para comparar contra utcDate (evita desfases de huso horario
    # entre tu hora local y la hora en que la API entrega los partidos).
    hoy = datetime.now(timezone.utc)
    proxima_semana = hoy + timedelta(days=7)

    print(f"🚀 Buscando partidos entre {hoy.date()} y {proxima_semana.date()}...")

    for liga in LIGAS:
        # Ya no filtramos status en la URL: lo hacemos nosotros después,
        # aceptando tanto SCHEDULED como TIMED.
        url = f"https://api.football-data.org/v4/competitions/{liga}/matches"

        partidos_liga = 0
        intentos = 0

        while intentos < 3:
            try:
                response = requests.get(url, headers=headers)
            except Exception as e:
                print(f"❌ Fallo de conexión en {liga}: {e}")
                break

            if response.status_code == 200:
                data = response.json()
                matches = data.get('matches', [])

                for m in matches:
                    if m.get('status') not in ESTADOS_VALIDOS:
                        continue
                    if not m.get('utcDate'):
                        continue

                    fecha_partido = datetime.strptime(
                        m['utcDate'], "%Y-%m-%dT%H:%M:%SZ"
                    ).replace(tzinfo=timezone.utc)

                    if hoy <= fecha_partido <= proxima_semana:
                        all_matches.append({
                            'League': liga,
                            'Date': m['utcDate'],
                            'Local': m['homeTeam']['shortName'],
                            'Visita': m['awayTeam']['shortName']
                        })
                        partidos_liga += 1

                print(f"✅ {liga}: {partidos_liga} partidos válidos.")
                break  # respuesta ok, no reintentamos

            elif response.status_code == 429:
                # Límite de la cuenta gratuita (10 req/min): esperamos y reintentamos
                intentos += 1
                espera = 15 * intentos
                print(f"⏳ Rate limit en {liga} (intento {intentos}/3), esperando {espera}s...")
                time.sleep(espera)
                continue

            else:
                # Ahora sí mostramos el motivo real (antes solo se veía el código)
                print(f"⚠️ Error en {liga}: {response.status_code} - {response.text[:200]}")
                break

        # Pausa obligatoria para no pasarse del límite de la cuenta gratuita
        time.sleep(10)

    if all_matches:
        conn = sqlite3.connect(DB_NAME)
        df_jornada = pd.DataFrame(all_matches)

        # Esta es la tabla que "visualizaciones.py" busca para limpiar el desorden
        df_jornada.to_sql('tabla_predicciones_limpia', conn, if_exists='replace', index=False)
        conn.close()
        print(f"\n🔥 ¡LISTO! {len(all_matches)} partidos cargados en la base de datos.")
    else:
        print("\nℹ️ No se encontraron partidos programados para los próximos días.")


if __name__ == "__main__":
    actualizar_partidos_semana()