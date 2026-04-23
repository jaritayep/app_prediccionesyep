import sqlite3

# Conectar a tu base de datos
conn = sqlite3.connect('database_partidos.db')
cursor = conn.cursor()

# Ejecutar la orden de aniquilación de partidos sin tiros
cursor.execute("""
    DELETE FROM historial_multiliga_ml 
    WHERE HST IS NULL OR HST = 0 OR HS IS NULL OR HS = 0
""")

# Ver cuántos borramos
filas_borradas = cursor.rowcount
conn.commit()
conn.close()

print(f"🧹 Limpieza exitosa: Se eliminaron {filas_borradas} partidos 'zombies' sin estadísticas.")