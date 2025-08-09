import os
import numpy as np
import rasterio

# Función para procesar un archivo .tif y obtener índice de agua, máscara y sólidos suspendidos
def procesar_tif(ruta_tif, umbral=0.0, alpha=86.402, exp_b04_b05=-0.424, exp_b04=-0.116):
    # Abrir ráster y validar que tenga al menos 11 bandas
    with rasterio.open(ruta_tif) as src:
        if src.count < 11:
            raise ValueError(f"Se esperaban ≥11 bandas, pero el archivo tiene {src.count}")  # Validación de bandas
        perfil = src.profile.copy()  # Guardar metadatos (CRS, transform, etc.)
        # Leer solo las bandas necesarias: B03, B04, B05 y B11
        b03, b04, b05, b11 = src.read([3, 4, 5, 11]).astype('float32')

    # Calcular índice de agua tipo ND (Normalizado) para identificar zonas de agua
    indice = (b03 - b11) / (b03 + b11 + 1e-6)  # Se añade 1e-6 para evitar división por cero
    print(f"Índice de agua → min: {np.nanmin(indice):.3f}, max: {np.nanmax(indice):.3f}")  # Estadísticas básicas

    # Crear máscara binaria: 1=agua, 0=no agua, 255=fuera de escena (nodata)
    mascara = np.full(indice.shape, 255, dtype='uint8')  # Inicializar todo como nodata
    mascara[indice > umbral] = 1  # Marcar como agua los píxeles por encima del umbral
    mascara[indice <= umbral] = 0  # Marcar como tierra el resto
    porcentaje_agua = 100 * np.mean(mascara == 1)  # Calcular porcentaje de agua
    print(f"% de píxeles clasificados como agua: {porcentaje_agua:.2f}%")

    # Calcular sólidos suspendidos usando bandas B04 y B05 en píxeles válidos
    valido = (b04 > 0) & (b05 > 0)  # Solo donde ambas bandas tienen valores positivos
    parametro = np.full(b04.shape, -9999.0, dtype='float32')  # Inicializar con nodata = -9999.0
    parametro[valido] = (
        alpha * (b04[valido] / b05[valido])**exp_b04_b05 * (b04[valido]**exp_b04)
    )  # Fórmula empírica para sólidos suspendidos

    # Aplicar máscara de agua al parámetro: fuera de agua → nodata
    parametro[mascara != 1] = -9999.0  # Asegurar que solo zona de agua conserve valores

    return indice.astype('float32'), mascara, parametro, perfil  # Devolver resultados y perfil

if __name__ == '__main__':
    # Configuración de rutas y parámetros
    carpeta = './recortes_acolite'
    archivo = os.path.join(carpeta, '2025-06-09.tif')  # Archivo de entrada
    umbral = 0.0           # Umbral inicial para clasificar agua
    guardar_archivos = False  # Cambia a True para escribir archivos de salida

    # Ejecutar procesamiento
    indice, mascara, parametro, perfil = procesar_tif(archivo, umbral=umbral)

    if guardar_archivos:
        # Guardar índice de agua como TIFF
        perfil_ind = perfil.copy()
        perfil_ind.update(dtype=rasterio.float32, count=1, nodata=None)
        out_ind = os.path.join(carpeta, 'indice_agua.tif')
        with rasterio.open(out_ind, 'w', **perfil_ind) as dst:
            dst.write(indice, 1)

        # Guardar máscara de agua como TIFF
        perfil_msk = perfil.copy()
        perfil_msk.update(dtype=rasterio.uint8, count=1, nodata=255)
        out_msk = os.path.join(carpeta, 'mascara_agua.tif')
        with rasterio.open(out_msk, 'w', **perfil_msk) as dst:
            dst.write(mascara, 1)

        # Guardar parámetro de sólidos suspendidos
        perfil_par = perfil.copy()
        perfil_par.update(dtype=rasterio.float32, count=1, nodata=-9999.0)
        out_par = os.path.join(carpeta, 'parametro_sol_sus.tif')
        with rasterio.open(out_par, 'w', **perfil_par) as dst:
            dst.write(parametro, 1)

        print('Archivos guardados:')
        print(' - Índice de agua:', out_ind)
        print(' - Máscara de agua:', out_msk)
        print(' - Parámetro sol_sus:', out_par)
    else:
        print("Procesamiento completo. No se guardaron archivos.")
