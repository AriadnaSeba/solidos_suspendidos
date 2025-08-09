import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# mapa S2 (nm)
S2_BAND_WAVELENGTHS = {
    'B01': 443, 'B02': 490, 'B03': 560, 'B04': 665, 'B05': 705,
    'B06': 740, 'B07': 783, 'B08': 842, 'B8A': 865, 'B09': 945,
    'B10': 1375, 'B11': 1610, 'B12': 2190
}

def descripcion_a_nm(desc):
    if desc is None:
        return None
    s = str(desc).upper()
    m = re.search(r"(\d{3,4})", s)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    m2 = re.match(r"^B(\d{1,2}A?)$", s)
    if m2:
        key = 'B' + m2.group(1).upper()
        variants = [key, key.zfill(3), key.replace('B8','B08')]
        for v in variants:
            if v in S2_BAND_WAVELENGTHS:
                return S2_BAND_WAVELENGTHS[v]
    return None

def elegir_bandas(ruta, prefer_b02=True):
    """Devuelve diccionario sel={'R':(idx,nm),'G':...,'B':(...)} (índices 1-based)."""
    with rasterio.open(ruta) as src:
        bandas_desc = [d if d is not None else '' for d in src.descriptions]
    band_nm_list = [descripcion_a_nm(d) for d in bandas_desc]

    def banda_mas_cercana(target_nm):
        best_idx, best_diff = None, float("inf")
        for i, nm in enumerate(band_nm_list):
            if nm is None:
                continue
            diff = abs(nm - target_nm)
            if diff < best_diff:
                best_diff = diff
                best_idx = i + 1
        return best_idx, (band_nm_list[best_idx-1] if best_idx is not None else None)

    targets = {'R':665,'G':560,'B':443}
    sel = {}
    for k,t in targets.items():
        sel[k] = banda_mas_cercana(t)

    # preferir B02 si existe (azul más natural)
    desc_upper = [d.upper() for d in bandas_desc]
    if prefer_b02 and any(d == 'B02' for d in desc_upper):
        sel['B'] = (desc_upper.index('B02') + 1, S2_BAND_WAVELENGTHS.get('B02'))

    return sel, bandas_desc, band_nm_list

def read_band(ruta, idx):
    if idx is None:
        return None
    with rasterio.open(ruta) as src:
        try:
            return src.read(idx).astype(float), src.nodata
        except Exception:
            return None, src.nodata

def handle_nodata_and_scaling(band, src_nodata=None, scale_divisor=10000.0, scale_threshold=1.5):
    if band is None:
        return None, False
    b = band.copy()
    if src_nodata is not None:
        b[b == src_nodata] = np.nan
    b[b < 0] = np.nan
    try:
        p98 = np.nanpercentile(b, 98)
    except:
        p98 = np.nan
    scaled = False
    if not np.isnan(p98) and p98 > scale_threshold:
        b = b / scale_divisor
        scaled = True
    return b, scaled

def normalizar(b, ampl=6.0, p_low=1, p_high=99):
    if b is None:
        return None
    b = b * ampl
    p_lo, p_hi = np.nanpercentile(b, p_low), np.nanpercentile(b, p_high)
    if np.isclose(p_hi, p_lo):
        arr = np.clip(b, p_lo - 1e-6, p_lo + 1e-6)
        return (arr - (p_lo - 1e-6)) / (2e-6)
    b_clip = np.clip(b, p_lo, p_hi)
    return (b_clip - p_lo) / (p_hi - p_lo)

def make_rgb(ruta,
             prefer_b02=True,
             scale_divisor=10000.0,
             scale_threshold=1.5,
             ampl=6.0,
             p_low=1, p_high=99,
             gamma=1.1):
    """Construye el arreglo RGB normalizado (valores en [0,1]) y devuelve metadata."""
    sel, bandas_desc, band_nm_list = elegir_bandas(ruta, prefer_b02=prefer_b02)

    # leer bandas
    r_band, nod = read_band(ruta, sel['R'][0])
    g_band, _ = read_band(ruta, sel['G'][0])
    b_band, _ = read_band(ruta, sel['B'][0])

    # nodata/scale
    r_band, s_r = handle_nodata_and_scaling(r_band, nod, scale_divisor, scale_threshold)
    g_band, s_g = handle_nodata_and_scaling(g_band, nod, scale_divisor, scale_threshold)
    b_band, s_b = handle_nodata_and_scaling(b_band, nod, scale_divisor, scale_threshold)

    # normalizar
    r_n = normalizar(r_band, ampl=ampl, p_low=p_low, p_high=p_high)
    g_n = normalizar(g_band, ampl=ampl, p_low=p_low, p_high=p_high)
    b_n = normalizar(b_band, ampl=ampl, p_low=p_low, p_high=p_high)

    # forma y relleno
    shape = None
    for x in (r_n, g_n, b_n):
        if x is not None:
            shape = x.shape
            break
    if shape is None:
        raise RuntimeError("No se pudo leer ninguna banda válida para RGB.")

    def fill(x):
        return np.zeros(shape) if x is None else np.nan_to_num(x, nan=0.0)

    rgb = np.dstack((fill(r_n), fill(g_n), fill(b_n)))
    rgb = np.clip(rgb ** (1.0 / gamma), 0, 1)

    info = {
        'sel': sel,
        'bandas_desc': bandas_desc,
        'band_nm_list': band_nm_list,
        'scaled': (s_r, s_g, s_b)
    }
    return rgb, info

def save_rgb(rgb, out_png, dpi=300):
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.imsave(out_png, np.clip(rgb,0,1), dpi=dpi)
    return out_png

if __name__ == "__main__":
    # ejemplo de uso rápido
    import sys
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    rgb, info = make_rgb(ruta)
    print("Bandas elegidas:", info['sel'])
    plt.figure(figsize=(8,6))
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()