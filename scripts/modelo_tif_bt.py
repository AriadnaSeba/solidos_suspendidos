import os
import numpy as np
import rasterio

# Función para procesar un archivo .tif y obtener índice de agua, máscara y sol_sus
#   • índice de agua (B03 y B11)
#   • máscara binaria de agua (1=agua, 0=tierra, 255=nodata)
#   • sol_sus según la fórmula: alpha * (B04/B05)^exp_b04_b05 * (B02)^exp_b02
def procesar_tif(ruta_tif,
                 umbral=0.0,
                 alpha=86.402,
                 exp_b04_b05=-0.371,
                 exp_b02=-0.091,
                 nodata_out=-9999.0):
    # 1) Abrir ráster y validar que tenga al menos 11 bandas
    with rasterio.open(ruta_tif) as src:
        if src.count < 11:
            raise ValueError(f"Se esperaban ≥11 bandas, pero el archivo tiene {src.count}")
        perfil = src.profile.copy()  # Guardar metadatos (CRS, transform, etc.)
        # Leer solo las bandas necesarias de golpe:
        #   B03 y B11 para el índice de agua,
        #   B02, B04, B05 para sol_sus
        # IMPORTANTE: read indices as they appear en tu producto (aquí se asume 1-based)
        b03, b11, b02, b04, b05 = src.read([3, 11, 2, 4, 5]).astype('float32')

    # 2) Calcular índice de agua (normalizado), evitando división por cero
    indice = (b03 - b11) / (b03 + b11 + 1e-6)
    print(f"Índice de agua → min: {np.nanmin(indice):.3f}, max: {np.nanmax(indice):.3f}")

    # 3) Crear máscara binaria para agua: 1=agua, 0=tierra, 255=nodata
    mascara = np.full(indice.shape, 255, dtype='uint8')
    mascara[indice > umbral] = 1
    mascara[indice <= umbral] = 0
    pct_agua = 100 * np.mean(mascara == 1)
    print(f"% de píxeles clasificados como agua: {pct_agua:.2f}%")

    # 4) Calcular sol_sus con la fórmula empírica, con protecciones numéricas
    # Epsilon para evitar ceros/valores negativos que provoquen potencias inválidas
    eps = 1e-6

    # Calculamos ratio y protegemos para que sea > 0
    ratio = np.divide(b04, b05 + eps)        # b05 + eps para evitar división por cero
    # Si ratio no es positivo (p. ej. por valores negativos en bandas), lo reemplazamos por eps
    ratio_safe = np.where(np.isfinite(ratio) & (ratio > 0), ratio, eps)

    # Para B02: necesitamos base positiva para potencias fraccionarias
    b02_safe = np.where(np.isfinite(b02) & (b02 > 0), b02, eps)

    with np.errstate(divide='ignore', invalid='ignore'):
        sol_sus = alpha * (ratio_safe ** exp_b04_b05) * (b02_safe ** exp_b02)

    # 5) Aplicar máscara de agua: fuera de agua → nodata (nodata_out)
    parametro = np.where(mascara == 1, sol_sus, nodata_out).astype('float32')

    # 6) Reemplazar inf/nan resultantes dentro de la zona de agua por nodata
    bad = ~np.isfinite(parametro)
    parametro[bad] = nodata_out

    return indice.astype('float32'), mascara, parametro, perfil


if __name__ == '__main__':
    # === CONFIGURACIÓN ===
    carpeta = './recortes_acolite'
    archivo = os.path.join(carpeta, '2025-06-09.tif')  # Archivo de entrada (ajusta nombre/ruta)
    umbral = 0.0  # Umbral de ejemplo para detectar agua en el índice
    guardar_archivos = False  # Poner True para exportar resultados a disco
    nodata_out = -9999.0

    # === PROCESAMIENTO ===
    try:
        indice, mascara, parametro, perfil = procesar_tif(
            archivo,
            umbral=umbral,
            alpha=86.402,
            exp_b04_b05=-0.371,
            exp_b02=-0.091,
            nodata_out=nodata_out
        )
    except Exception as e:
        raise SystemExit(f"Error durante procesamiento: {e}")

    if guardar_archivos:
        # Guardar índice de agua
        pf = perfil.copy()
        pf.update(dtype=rasterio.float32, count=1, nodata=None)
        out_ind = os.path.join(carpeta, 'indice_agua.tif')
        with rasterio.open(out_ind, 'w', **pf) as dst:
            dst.write(indice.astype(np.float32), 1)

        # Guardar máscara de agua (uint8, nodata=255)
        pf = perfil.copy()
        pf.update(dtype=rasterio.uint8, count=1, nodata=255)
        out_msk = os.path.join(carpeta, 'mascara_agua.tif')
        with rasterio.open(out_msk, 'w', **pf) as dst:
            dst.write(mascara.astype(np.uint8), 1)

        # Guardar sol_sus
        pf = perfil.copy()
        pf.update(dtype=rasterio.float32, count=1, nodata=nodata_out)
        out_sol = os.path.join(carpeta, 'sol_sus.tif')
        with rasterio.open(out_sol, 'w', **pf) as dst:
            dst.write(parametro.astype(np.float32), 1)

        print('Archivos guardados:')
        print(f' - Índice de agua: {out_ind}')
        print(f' - Máscara de agua: {out_msk}')
        print(f' - sol_sus: {out_sol}')
    else:
        print("Procesamiento completo. No se guardaron archivos.")
