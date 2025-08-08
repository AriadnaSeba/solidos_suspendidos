import os
import numpy as np
import rasterio

# Función para procesar un archivo .tif y obtener índice de agua, máscara y sol_sus
#   • índice de agua (B03 y B11)
#   • máscara binaria de agua
#   • sol_sus según la fórmula: alpha * (B05/B04)^exp_b05_b04 * (B02)^exp_b02
def procesar_tif(ruta_tif,
                 umbral=0.0,
                 alpha=86.402,
                 exp_b05_b04=-0.371,
                 exp_b02=-0.091):
    # 1) Abrir ráster y validar que tenga al menos 11 bandas
    with rasterio.open(ruta_tif) as src:
        if src.count < 11:
            raise ValueError(f"Se esperaban ≥11 bandas, pero el archivo tiene {src.count}")
        perfil = src.profile.copy()  # Guardar metadatos (CRS, transform, etc.)
        # Leer solo las bandas necesarias de golpe:
        #   B03 y B11 para el índice de agua,
        #   B02, B04, B05 para sol_sus
        b03, b11, b02, b04, b05 = src.read([3, 11, 2, 4, 5]).astype('float32')

    # 2) Calcular índice de agua (normalizado)
    indice = (b03 - b11) / (b03 + b11 + 1e-6)  # +1e-6 para evitar división por cero
    print(f"Índice de agua → min: {np.nanmin(indice):.3f}, max: {np.nanmax(indice):.3f}")

    # 3) Crear máscara binaria para agua: 1=agua, 0=tierra, 255=nodata
    mascara = np.full(indice.shape, 255, dtype='uint8')
    mascara[indice > umbral] = 1
    mascara[indice <= umbral] = 0
    pct_agua = 100 * np.mean(mascara == 1)
    print(f"% de píxeles clasificados como agua: {pct_agua:.2f}%")

    # 4) Calcular sol_sus con la fórmula empírica
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = b05 / b04  # B05/B04
        sol_sus = alpha * (ratio**exp_b05_b04) * (b02**exp_b02)

    # 5) Aplicar máscara de agua: fuera de agua → nodata (-9999.0)
    parametro = np.where(mascara == 1, sol_sus, -9999.0).astype('float32')

    return indice.astype('float32'), mascara, parametro, perfil


if __name__ == '__main__':
    # === CONFIGURACIÓN ===
    carpeta = './recortes_acolite'
    archivo = os.path.join(carpeta, '2025-06-09.tif')  # Archivo de entrada
    umbral = 0.33  # Umbral de ejemplo para agua
    guardar_archivos = False  # Cambia a True para exportar resultados

    # === PROCESAMIENTO ===
    indice, mascara, parametro, perfil = procesar_tif(archivo, umbral=umbral)

    if guardar_archivos:
        # Guardar índice de agua
        pf = perfil.copy()
        pf.update(dtype=rasterio.float32, count=1, nodata=None)
        out_ind = os.path.join(carpeta, 'indice_agua.tif')
        with rasterio.open(out_ind, 'w', **pf) as dst:
            dst.write(indice, 1)

        # Guardar máscara de agua
        pf = perfil.copy()
        pf.update(dtype=rasterio.uint8, count=1, nodata=255)
        out_msk = os.path.join(carpeta, 'mascara_agua.tif')
        with rasterio.open(out_msk, 'w', **pf) as dst:
            dst.write(mascara, 1)

        # Guardar sol_sus
        pf = perfil.copy()
        pf.update(dtype=rasterio.float32, count=1, nodata=-9999.0)
        out_sol = os.path.join(carpeta, 'sol_sus.tif')
        with rasterio.open(out_sol, 'w', **pf) as dst:
            dst.write(parametro, 1)

        print('Archivos guardados:')
        print(f' - Índice de agua: {out_ind}')
        print(f' - Máscara de agua: {out_msk}')
        print(f' - sol_sus: {out_sol}')
    else:
        print("Procesamiento completo. No se guardaron archivos.")
