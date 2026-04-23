import sqlite3

conn = sqlite3.connect('database_partidos.db')
cursor = conn.cursor()

# Mapa de traducciones manuales para limpiar la base de datos actual
mapeo = {
    "Nott'm Forest": "Nottingham Forest",
    "Ath Bilbao": "Athletic Club",
    "Athletic Bilbao": "Athletic Club",
    "Atl Madrid": "Atletico Madrid",
    "Barca": "Barcelona",
    "Barça": "Barcelona",
    "Paris SG": "PSG",
    "M'gladbach": "Borussia Monchengladbach"
}

for sucio, limpio in mapeo.items():
    # Limpiar en el historial
    cursor.execute("UPDATE historial_multiliga_ml SET HomeTeam = ? WHERE HomeTeam = ?", (limpio, sucio))
    cursor.execute("UPDATE historial_multiliga_ml SET AwayTeam = ? WHERE AwayTeam = ?", (limpio, sucio))
    # Limpiar en las predicciones por si acaso
    cursor.execute("UPDATE tabla_predicciones_limpia SET Local = ? WHERE Local = ?", (limpio, sucio))
    cursor.execute("UPDATE tabla_predicciones_limpia SET Visita = ? WHERE Visita = ?", (limpio, sucio))

conn.commit()
conn.close()
print("✅ Nombres estandarizados en toda la base de datos.")