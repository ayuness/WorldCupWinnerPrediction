import csv
from collections import defaultdict

# --- 1. Leer todos los equipos y sus status desde los archivos de eliminatorias ---
elim_files = {
    'conmebol': 'Data/Eliminatorias/eliminatorias_conmebol.csv',
    'uefa':     'Data/Eliminatorias/eliminatorias_uefa.csv',
    'afc':      'Data/Eliminatorias/eliminatorias_afc.csv',
    'caf':      'Data/Eliminatorias/eliminatorias_caf.csv',
    'concacaf': 'Data/Eliminatorias/eliminatorias_concacaf.csv',
    'ofc':      'Data/Eliminatorias/eliminatorias_ofc.csv',
}

team_status = {}  # country_code -> status (última fase)
for conf, path in elim_files.items():
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['country_code']
            status = row['status']
            team_status[code] = status  # sobreescribe con la última fase

all_codes = set(team_status.keys())
print(f"Total equipos en eliminatorias: {len(all_codes)}")

# --- 2. Mapeo nombre completo → country_code ---
# SRB y RUS no están en eliminatorias, entonces Yugoslavia/Soviet Union se ignoran
name_to_code = {
    'Algeria':                'ALG',
    'Argentina':              'ARG',
    'Australia':              'AUS',
    'Austria':                'AUT',
    'Belgium':                'BEL',
    'Bolivia':                'BOL',
    'Bosnia and Herzegovina': 'BIH',
    'Brazil':                 'BRA',
    'Canada':                 'CAN',
    "Côte d'Ivoire":          'CIV',
    'Colombia':               'COL',
    'Croatia':                'CRO',
    'Czech Republic':         'CZE',
    'Czechoslovakia':         'CZE',
    'Denmark':                'DEN',
    'Ecuador':                'ECU',
    'Egypt':                  'EGY',
    'England':                'ENG',
    'Spain':                  'ESP',
    'France':                 'FRA',
    'Germany':                'GER',
    'West Germany':           'GER',
    'Ghana':                  'GHA',
    'Haiti':                  'HAI',
    'Republic of Ireland':    'IRL',
    'IR Iran':                'IRN',
    'Iraq':                   'IRQ',
    'Italy':                  'ITA',
    'Jamaica':                'JAM',
    'Japan':                  'JPN',
    'Korea Republic':         'KOR',
    'Saudi Arabia':           'KSA',
    'Morocco':                'MAR',
    'Mexico':                 'MEX',
    'Netherlands':            'NED',
    'Nigeria':                'NGA',
    'Northern Ireland':       'NIR',
    'Norway':                 'NOR',
    'New Zealand':            'NZL',
    'Panama':                 'PAN',
    'Paraguay':               'PAR',
    'Poland':                 'POL',
    'Portugal':               'POR',
    'Qatar':                  'QAT',
    'Romania':                'ROU',
    'South Africa':           'RSA',
    'Scotland':               'SCO',
    'Senegal':                'SEN',
    'Switzerland':            'SUI',
    'Slovakia':               'SVK',
    'Sweden':                 'SWE',
    'Tunisia':                'TUN',
    'Türkiye':                'TUR',
    'Ukraine':                'UKR',
    'Uruguay':                'URU',
    'United States':          'USA',
    'Wales':                  'WAL',
    # Equipos históricos con sucesor en eliminatorias
    'FR Yugoslavia':          None,  # SRB no está en eliminatorias
    'Yugoslavia':             None,  # SRB no está en eliminatorias
    'Serbia':                 None,  # SRB no está en eliminatorias
    'Serbia and Montenegro':  None,  # SRB no está en eliminatorias
    'Soviet Union':           None,  # RUS no está en eliminatorias
    'Russia':                 None,  # RUS no está en eliminatorias
}

# --- 3. Inicializar estadísticas para todos los equipos ---
stats = {}
for code in all_codes:
    stats[code] = {'pj': 0, 'g': 0, 'e': 0, 'p': 0, 'gf': 0, 'gc': 0}

# --- 4. Procesar partidos ---
with open('Data/WorldCupHistory/matches_1930_2022.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        home_team = row['home_team']
        away_team = row['away_team']
        home_score = int(row['home_score'])
        away_score = int(row['away_score'])

        home_code = name_to_code.get(home_team)
        away_code = name_to_code.get(away_team)

        # Procesar equipo local
        if home_code and home_code in all_codes:
            s = stats[home_code]
            s['pj'] += 1
            s['gf'] += home_score
            s['gc'] += away_score
            if home_score > away_score:
                s['g'] += 1
            elif home_score == away_score:
                s['e'] += 1
            else:
                s['p'] += 1

        # Procesar equipo visitante
        if away_code and away_code in all_codes:
            s = stats[away_code]
            s['pj'] += 1
            s['gf'] += away_score
            s['gc'] += home_score
            if away_score > home_score:
                s['g'] += 1
            elif away_score == home_score:
                s['e'] += 1
            else:
                s['p'] += 1

# --- 5. Construir filas finales ---
rows = []
for code in all_codes:
    s = stats[code]
    pj = s['pj']
    g  = s['g']
    e  = s['e']
    p  = s['p']
    gf = s['gf']
    gc = s['gc']
    dg = gf - gc

    gf_per_game = round(gf / pj, 3) if pj > 0 else 0.0
    gc_per_game = round(gc / pj, 3) if pj > 0 else 0.0
    win_pct     = round(g  / pj, 3) if pj > 0 else 0.0

    rows.append({
        'country_code': code,
        'status':       team_status[code],
        'pj':           pj,
        'g':            g,
        'e':            e,
        'p':            p,
        'gf':           gf,
        'gc':           gc,
        'dg':           dg,
        'gf_per_game':  gf_per_game,
        'gc_per_game':  gc_per_game,
        'win_pct':      win_pct,
    })

# --- 6. Ordenar: win_pct desc, gf_per_game desc ---
rows.sort(key=lambda r: (-r['win_pct'], -r['gf_per_game']))

# --- 7. Escribir CSV ---
fieldnames = [
    'country_code', 'status', 'pj', 'g', 'e', 'p',
    'gf', 'gc', 'dg', 'gf_per_game', 'gc_per_game', 'win_pct'
]

output_path = 'Data/historial_mundialista.csv'
with open(output_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV generado: {output_path}")
print(f"Total filas: {len(rows)}")
print("\nPrimeras 10 filas:")
for r in rows[:10]:
    print(r)
print("\nEquipos sin historial mundialista:")
for r in rows:
    if r['pj'] == 0:
        print(r['country_code'], r['status'])
