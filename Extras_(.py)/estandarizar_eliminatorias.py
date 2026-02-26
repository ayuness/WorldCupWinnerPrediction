import csv
import os

BASE = 'Data/Eliminatorias'
STANDARD_COLS = ['country_code', 'status', 'pj', 'g', 'e', 'p', 'pts', 'gf', 'gc', 'dg',
                 'gf_per_game', 'gc_per_game', 'win_pct']

def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=STANDARD_COLS)
        writer.writeheader()
        writer.writerows(rows)

def round3(val):
    return round(val, 3)

# ──────────────────────────────────────────────
# UEFA, CAF, OFC — solo agregar win_pct
# ──────────────────────────────────────────────
for name in ['eliminatorias_uefa.csv', 'eliminatorias_caf.csv', 'eliminatorias_ofc.csv']:
    path = os.path.join(BASE, name)
    rows = []
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pj = int(row['pj'])
            g  = int(row['g'])
            win_pct = round3(g / pj) if pj > 0 else 0.0
            rows.append({
                'country_code': row['country_code'],
                'status':       row['status'],
                'pj':           pj,
                'g':            g,
                'e':            int(row['e']),
                'p':            int(row['p']),
                'pts':          int(row['pts']),
                'gf':           int(row['gf']),
                'gc':           int(row['gc']),
                'dg':           int(row['dg']),
                'gf_per_game':  float(row['gf_per_game']),
                'gc_per_game':  float(row['gc_per_game']),
                'win_pct':      win_pct,
            })
    write_csv(path, rows)
    print(f"[OK] {name} — {len(rows)} filas")

# ──────────────────────────────────────────────
# CONMEBOL — eliminar columnas extra, reordenar
# ──────────────────────────────────────────────
path = os.path.join(BASE, 'eliminatorias_conmebol.csv')
rows = []
with open(path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pj = int(row['pj'])
        g  = int(row['g'])
        win_pct = round3(g / pj) if pj > 0 else 0.0
        rows.append({
            'country_code': row['country_code'],
            'status':       row['status'],
            'pj':           pj,
            'g':            g,
            'e':            int(row['e']),
            'p':            int(row['p']),
            'pts':          int(row['pts']),
            'gf':           int(row['gf']),
            'gc':           int(row['gc']),
            'dg':           int(row['dg']),
            'gf_per_game':  float(row['gf_per_game']),
            'gc_per_game':  float(row['gc_per_game']),
            'win_pct':      win_pct,
        })
write_csv(path, rows)
print(f"[OK] eliminatorias_conmebol.csv — {len(rows)} filas")

# ──────────────────────────────────────────────
# AFC — agrupar por equipo, sumar fases
# ──────────────────────────────────────────────
PHASE_ORDER = {'2da_ronda': 1, '3ra_fase': 2, '4ta_ronda': 3}
path = os.path.join(BASE, 'eliminatorias_afc.csv')

# Detectar si el archivo ya fue convertido (no tiene columna 'phase')
with open(path, encoding='utf-8') as f:
    first_cols = csv.DictReader(f).fieldnames

if 'phase' not in first_cols:
    print(f"[SKIP] eliminatorias_afc.csv — ya tiene formato estándar")
else:
    # Acumular por equipo
    acc = {}  # country_code -> dict
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row['country_code']
            phase = row['phase']
            if code not in acc:
                acc[code] = {
                    'status': row['status'],
                    'last_phase_order': PHASE_ORDER.get(phase, 0),
                    'pj': 0, 'g': 0, 'e': 0, 'p': 0,
                    'pts': 0, 'gf': 0, 'gc': 0,
                }
            # Actualizar status si esta es una fase posterior
            phase_order = PHASE_ORDER.get(phase, 0)
            if phase_order >= acc[code]['last_phase_order']:
                acc[code]['status'] = row['status']
                acc[code]['last_phase_order'] = phase_order
            # Sumar estadísticas
            acc[code]['pj']  += int(row['pj'])
            acc[code]['g']   += int(row['g'])
            acc[code]['e']   += int(row['e'])
            acc[code]['p']   += int(row['p'])
            acc[code]['pts'] += int(row['pts'])
            acc[code]['gf']  += int(row['gf'])
            acc[code]['gc']  += int(row['gc'])

    rows = []
    for code, d in acc.items():
        pj = d['pj']
        g  = d['g']
        gf = d['gf']
        gc = d['gc']
        rows.append({
            'country_code': code,
            'status':       d['status'],
            'pj':           pj,
            'g':            g,
            'e':            d['e'],
            'p':            d['p'],
            'pts':          d['pts'],
            'gf':           gf,
            'gc':           gc,
            'dg':           gf - gc,
            'gf_per_game':  round3(gf / pj) if pj > 0 else 0.0,
            'gc_per_game':  round3(gc / pj) if pj > 0 else 0.0,
            'win_pct':      round3(g  / pj) if pj > 0 else 0.0,
        })
    write_csv(path, rows)
    print(f"[OK] eliminatorias_afc.csv — {len(rows)} filas (agrupadas desde 21 filas)")

# ──────────────────────────────────────────────
# CONCACAF — anfitriones con 0s, agregar win_pct
# ──────────────────────────────────────────────
path = os.path.join(BASE, 'eliminatorias_concacaf.csv')
rows = []
with open(path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convertir vacíos/NaN a 0 para numéricos
        def to_int(v):
            return int(float(v)) if v.strip() not in ('', 'nan', 'NaN') else 0
        def to_float(v):
            return float(v) if v.strip() not in ('', 'nan', 'NaN') else 0.0

        pj  = to_int(row['pj'])
        g   = to_int(row['g'])
        e   = to_int(row['e'])
        p   = to_int(row['p'])
        pts = to_int(row['pts'])
        gf  = to_int(row['gf'])
        gc  = to_int(row['gc'])
        dg  = gf - gc

        gf_per_game = to_float(row['gf_per_game'])
        gc_per_game = to_float(row['gc_per_game'])
        win_pct = round3(g / pj) if pj > 0 else 0.0

        rows.append({
            'country_code': row['country_code'],
            'status':       row['status'],
            'pj':           pj,
            'g':            g,
            'e':            e,
            'p':            p,
            'pts':          pts,
            'gf':           gf,
            'gc':           gc,
            'dg':           dg,
            'gf_per_game':  gf_per_game,
            'gc_per_game':  gc_per_game,
            'win_pct':      win_pct,
        })
write_csv(path, rows)
print(f"[OK] eliminatorias_concacaf.csv — {len(rows)} filas")

# ──────────────────────────────────────────────
# VALIDACIÓN — imprimir cabeceras y primeras filas
# ──────────────────────────────────────────────
print()
print("=" * 60)
print("VALIDACIÓN")
print("=" * 60)
for name in [
    'eliminatorias_afc.csv', 'eliminatorias_conmebol.csv',
    'eliminatorias_concacaf.csv', 'eliminatorias_uefa.csv',
    'eliminatorias_caf.csv', 'eliminatorias_ofc.csv',
]:
    path = os.path.join(BASE, name)
    with open(path, encoding='utf-8') as f:
        rows_val = list(csv.DictReader(f))
    cols = list(rows_val[0].keys()) if rows_val else []
    print(f"\n{name}")
    print(f"  Columnas ({len(cols)}): {cols}")
    print(f"  Coincide con estándar: {cols == STANDARD_COLS}")
    print(f"  Filas: {len(rows_val)}")
    for r in rows_val[:3]:
        print(f"    {list(r.values())}")
