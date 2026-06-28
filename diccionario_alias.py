# diccionario_alias.py
#
# Canonical names always match the exact spelling stored in the database.
# Verified against: historial_multiliga_ml, historial_selecciones_ml, fixture_mundial.

ALIAS_GLOBAL = {
    # ── SELECCIONES (MUNDIAL / historial_selecciones_ml) ──────────────────────
    # Canonical = name as stored in historial_selecciones_ml / fixture_mundial
    "USA": "United States",
    "US": "United States",
    "Estados Unidos": "United States",
    "South Korea": "South Korea",          # DB stores "South Korea", NOT "Korea Republic"
    "Korea Republic": "South Korea",       # external sources may use this form
    "Corea del Sur": "South Korea",
    "Czechia": "Czech Republic",
    "Chequia": "Czech Republic",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
    "Congo DR": "DR Congo",
    "RD Congo": "DR Congo",
    "Cape Verde Islands": "Cape Verde",    # fixture_mundial uses "Cape Verde Islands"
    "Cabo Verde": "Cape Verde",
    "UAE": "United Arab Emirates",
    "EAU": "United Arab Emirates",
    "Turkiye": "Turkey",
    "Türkiye": "Turkey",
    "Republic of Ireland": "Republic of Ireland",   # DB canonical (not "Ireland")
    "Ivory Coast": "Ivory Coast",                   # DB canonical (not "Côte d'Ivoire")
    "Curaçao": "Curaçao",                           # DB canonical (accent included)
    "Curazao": "Curaçao",


    # ── CLUBES — PREMIER LEAGUE ────────────────────────────────────────────────
    # Canonical = name as stored in historial_multiliga_ml
    "Man United": "Man United",            # DB stores "Man United"
    "Manchester United": "Man United",
    "Man Utd": "Man United",
    "Man City": "Man City",                # DB stores "Man City"
    "Manchester City": "Man City",
    "Spurs": "Tottenham",                  # DB stores "Tottenham"
    "Tottenham Hotspur": "Tottenham",
    "Wolves": "Wolves",                    # DB stores "Wolves"
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Newcastle": "Newcastle",              # DB stores "Newcastle"
    "Newcastle United": "Newcastle",
    "Brighton": "Brighton",                # DB stores "Brighton"
    "Brighton & Hove Albion": "Brighton",
    "Nott'm Forest": "Nottingham Forest",  # both forms exist in DB; canonical = full name
    "Nottm Forest": "Nottingham Forest",
    "Sheffield Utd": "Sheffield United",   # DB stores "Sheffield United"
    "Luton": "Luton",                      # DB stores "Luton"
    "Luton Town": "Luton",
    "Bournemouth": "Bournemouth",          # DB stores "Bournemouth"
    "AFC Bournemouth": "Bournemouth",
    "West Brom": "West Brom",              # DB stores "West Brom"
    "West Bromwich Albion": "West Brom",
    "West Ham": "West Ham",                # DB stores "West Ham"
    "West Ham United": "West Ham",
    "Ipswich": "Ipswich",                  # DB stores "Ipswich"
    "Ipswich Town": "Ipswich",

    # ── CLUBES — LA LIGA ──────────────────────────────────────────────────────
    "Atlético Madrid": "Ath Madrid",       # DB stores "Ath Madrid"
    "Atletico Madrid": "Ath Madrid",
    "Atlético de Madrid": "Ath Madrid",
    "Atletico de Madrid": "Ath Madrid",
    "Athletic Club": "Athletic Club",      # DB stores "Athletic Club" (Bilbao)
    "Athletic Bilbao": "Ath Bilbao",       # DB stores "Ath Bilbao"
    "Real Betis": "Betis",                 # DB stores "Betis"
    "Real Sociedad": "Sociedad",           # DB stores "Sociedad"
    "Celta Vigo": "Celta",                 # DB stores "Celta"
    "Espanyol": "Espanol",                 # DB stores "Espanol"
    "Rayo Vallecano": "Vallecano",         # DB stores "Vallecano"
    "Real Valladolid": "Valladolid",       # DB stores "Valladolid"
    "UD Las Palmas": "Las Palmas",         # DB stores "Las Palmas"
    "CD Leganes": "Leganes",               # DB stores "Leganes"
    "Real Oviedo": "Oviedo",               # DB stores "Oviedo"

    # ── CLUBES — SERIE A ──────────────────────────────────────────────────────
    "Inter": "Inter",                      # DB stores "Inter"
    "Inter Milan": "Inter",
    "Internazionale": "Inter",
    "AC Milan": "Milan",                   # DB stores "Milan"
    "Hellas Verona": "Verona",             # DB stores "Verona"
    "US Salernitana": "Salernitana",       # DB stores "Salernitana"

    # ── CLUBES — BUNDESLIGA ───────────────────────────────────────────────────
"Borussia Monchengladbach": "M'gladbach",  # El nombre a la derecha DEBE ser el de la BD
    "Borussia M'gladbach": "M'gladbach",
    "Monchengladbach": "M'gladbach",
    "Gladbach": "M'gladbach",
    "B. Monchengladbach": "M'gladbach",
    "BMG": "M'gladbach",
    "Mönchengladbach": "Borussia Monchengladbach",
    "Borussia Mönchengladbach": "Borussia Monchengladbach",
    "BMG": "Borussia Monchengladbach",
    "FC Bayern": "Bayern Munich",          # DB stores "Bayern Munich"
    "Bayern": "Bayern Munich",
    "FC Bayern München": "Bayern Munich",
    "Dortmund": "Dortmund",               # DB stores "Dortmund"
    "Borussia Dortmund": "Dortmund",
    "BVB": "Dortmund",
    "Köln": "FC Koln",                    # DB stores "FC Koln"
    "FC Köln": "FC Koln",
    "Cologne": "FC Koln",
    "Hoffenheim": "Hoffenheim",           # DB stores "Hoffenheim"
    "TSG Hoffenheim": "Hoffenheim",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "Leverkusen": "Leverkusen",           # DB stores "Leverkusen"
    "Bayer Leverkusen": "Leverkusen",
    "Ein Frankfurt": "Ein Frankfurt",     # DB stores "Ein Frankfurt"
    "Eintracht Frankfurt": "Ein Frankfurt",
    "RB Leipzig": "RB Leipzig",           # DB stores "RB Leipzig"
    "Leipzig": "RB Leipzig",
    "Greuther Fürth": "Greuther Furth",   # DB stores "Greuther Furth"
    "SpVgg Greuther Fürth": "Greuther Furth",
    "Hertha BSC": "Hertha",               # DB stores "Hertha"
    "Hertha Berlin": "Hertha",
    "St. Pauli": "St Pauli",              # DB stores "St Pauli"
    "FC St. Pauli": "St Pauli",
    "Arminia Bielefeld": "Bielefeld",     # DB stores "Bielefeld"
    "VfL Wolfsburg": "Wolfsburg",         # DB stores "Wolfsburg"
    "SV Werder Bremen": "Werder Bremen",  # DB stores "Werder Bremen"
    "Mainz 05": "Mainz",                  # DB stores "Mainz"
    "FSV Mainz": "Mainz",
    "Union Berlin": "Union Berlin",       # DB stores "Union Berlin"
    "1. FC Union Berlin": "Union Berlin",
    "VfB Stuttgart": "Stuttgart",         # DB stores "Stuttgart"
    "SC Freiburg": "Freiburg",            # DB stores "Freiburg"
    "FC Augsburg": "Augsburg",            # DB stores "Augsburg"
    "Darmstadt 98": "Darmstadt",          # DB stores "Darmstadt"
    "SV Darmstadt": "Darmstadt",
    "Holstein Kiel": "Holstein Kiel",     # DB stores "Holstein Kiel"
    "Heidenheim": "Heidenheim",           # DB stores "Heidenheim"
    "1. FC Heidenheim": "Heidenheim",
    "Hamburger SV": "Hamburg",            # DB stores "Hamburg"

    # ── CLUBES — LIGUE 1 ──────────────────────────────────────────────────────
    "PSG": "PSG",                          # DB stores "PSG"
    "Paris Saint-Germain": "PSG",
    "Paris SG": "PSG",
    "Paris Saint Germain": "PSG",
    "AS Monaco": "Monaco",                 # DB stores "Monaco"
    "Olympique Marseille": "Marseille",    # DB stores "Marseille"
    "Olympique Lyonnais": "Lyon",          # DB stores "Lyon"
    "OL": "Lyon",
    "AS Saint-Etienne": "St Etienne",     # DB stores "St Etienne"
    "Saint-Etienne": "St Etienne",
    "Stade Rennais": "Rennes",             # DB stores "Rennes"
    "OGC Nice": "Nice",                    # DB stores "Nice"
    "LOSC Lille": "Lille",                 # DB stores "Lille"
    "RC Lens": "Lens",                     # DB stores "Lens"
    "RC Strasbourg": "Strasbourg",         # DB stores "Strasbourg"
    "Stade Brestois": "Brest",             # DB stores "Brest"
    "Stade de Reims": "Reims",             # DB stores "Reims"
    "Montpellier HSC": "Montpellier",      # DB stores "Montpellier"
    "FC Nantes": "Nantes",                 # DB stores "Nantes"
    "Le Havre AC": "Le Havre",             # DB stores "Le Havre"
    "FC Lorient": "Lorient",               # DB stores "Lorient"
    "Toulouse FC": "Toulouse",             # DB stores "Toulouse"
    "Angers SCO": "Angers",               # DB stores "Angers"
    "Clermont Foot": "Clermont",           # DB stores "Clermont"
    "Paris FC": "Paris FC",                # DB stores "Paris FC"
}


def normalizar_nombre(nombre):
    """
    Recibe un nombre de equipo (cualquier variante conocida),
    y devuelve el nombre canónico tal como está almacenado en la base de datos.
    La comparación ignora mayúsculas/minúsculas.
    """
    nombre_str = str(nombre).strip()
    nombre_lower = nombre_str.lower()

    for alias, canonical in ALIAS_GLOBAL.items():
        if alias.lower() == nombre_lower:
            return canonical

    return nombre_str