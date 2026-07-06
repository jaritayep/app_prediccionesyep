"""
Convierte Date/Time de fixture_mundial (asumidos en UTC) a hora de Chile (America/Santiago).
Hace un backup del .db antes de tocar nada.
"""

import sqlite3
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

DB_PATH = "database_partidos.db"
BACKUP_PATH = "database_partidos_backup.db"

UTC = ZoneInfo("UTC")
CHILE = ZoneInfo("America/Santiago")

def main():
    shutil.copy(DB_PATH, BACKUP_PATH)
    print(f"Backup creado en {BACKUP_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT fixture_id, Date, Time FROM fixture_mundial")
    filas = cur.fetchall()

    actualizados = 0
    for fixture_id, fecha, hora in filas:
        # Parsear fecha+hora como UTC
        dt_utc = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        dt_chile = dt_utc.astimezone(CHILE)

        nueva_fecha = dt_chile.strftime("%Y-%m-%d")
        nueva_hora = dt_chile.strftime("%H:%M")

        cur.execute(
            "UPDATE fixture_mundial SET Date = ?, Time = ? WHERE fixture_id = ?",
            (nueva_fecha, nueva_hora, fixture_id),
        )
        actualizados += 1

    conn.commit()
    conn.close()
    print(f"Listo: {actualizados} partidos actualizados a hora de Chile.")

if __name__ == "__main__":
    main()