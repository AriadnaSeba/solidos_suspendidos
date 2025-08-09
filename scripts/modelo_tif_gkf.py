import os
import numpy as np
import rasterio

# Función para procesar un archivo .tif y obtener índice de agua y parámetro de sólidos suspendidos
# basado en nueva fórmula y bandas (B02, B07, B8A)
def procesar_tif(ruta_tif, umbral=0.0,
                alpha=61.7089, exp_ratio=-0.8370, exp_b8a3=-0.0044):
    # 1. Abrir ráster y validar que tenga al menos 11 bandas
    with rasterio.open(ruta_tif) as src:
        if src.count < 11:
            raise ValueError(f"Se esperaban ≥11 bandas, pero el archivo tiene {src.count}")  # Validación de bandas
        perfil = src.profile.copy()  # Guardar metadatos (CRS, transform, etc.)
        # Leer solo las bandas necesarias: B03 para índice, B11 para índice, y luego B02, B07, B8A/B08 para sólidos
        b03 = src.read(3).astype('float32')  # Banda verde (B03)
        b11 = src.read(11).astype('float32')  # Banda SWIR (B11)
        # Para el cálculo de sólidos: necesitamos B02, B07, y B8A (o B08 si no existe B8A)
        bandas = {f"B{str(i).zfill(2)}": src.read(i).astype('float32')
                  for i in range(1, src.count + 1)}

    # 2. Calcular índice de agua tipo ND normalizado
    #    índice = (B03 - B11) / (B03 + B11)
    indice = (b03 - b11) / (b03 + b11 + 1e-6)  # Evitar división por cero
    print(f"Índice de agua → min: {np.nanmin(indice):.3f}, max: {np.nanmax(indice):.3f}")  # Estadísticas básicas

    # 3. Crear máscara binaria para agua: 1=agua, 0=no agua, 255=nodata (fuera escena)
    mascara = np.full(indice.shape, 255, dtype='uint8')  # Inicializar como nodata
    mascara[indice > umbral] = 1  # Agua
    mascara[indice <= umbral] = 0  # Tierra
    porcentaje_agua = 100 * np.mean(mascara == 1)
    print(f"% de píxeles clasificados como agua: {porcentaje_agua:.2f}%")

    # 4. Calcular parámetro de sólidos suspendidos usando la nueva fórmula:
    #    param = alpha * (B02/B07)**exp_ratio * (B8A**3)**exp_b8a3
    b02 = bandas.get('B02')
    b07 = bandas.get('B07')
    # Seleccionar B8A; si no existe, usar B08
    b8a = bandas.get('B8A', bandas.get('B08'))

    # Validar que las bandas existan en el archivo
    if b02 is None or b07 is None or b8a is None:
        raise KeyError("Faltan bandas B02, B07 o B8A/B08 en el archivo")

    # Convertir a float32
    b02 = b02.astype('float32')
    b07 = b07.astype('float32')
    b8a = b8a.astype('float32')

    # Cálculo robusto evitando errores numéricos
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = b02 / b07  # División de bandas
        cubo_b8a = b8a ** 3  # B8A elevado al cubo
        # Aplicar fórmula empírica
        parametro = alpha * (ratio ** exp_ratio) * (cubo_b8a ** exp_b8a3)

    # 5. Aplicar máscara de agua: fuera de agua → nodata (-9999.0)
    parametro = np.where(mascara == 1, parametro, -9999.0).astype('float32')

    return indice.astype('float32'), mascara, parametro, perfil  # Devolver resultados y perfil

if __name__ == '__main__':
    # Configuración de rutas y parámetros
    carpeta = './recortes_acolite'
    archivo = os.path.join(carpeta, '2025-06-09.tif')  # Archivo de entrada
    umbral = 0.0            # Umbral para clasificar agua
    guardar_archivos = False  # Cambia a True para escribir archivos de salida

    # Ejecutar procesamiento
    indice, mascara, parametro, perfil = procesar_tif(archivo, umbral=umbral)

    if guardar_archivos:
        # Guardar índice de agua
        perfil_ind = perfil.copy()
        perfil_ind.update(dtype=rasterio.float32, count=1, nodata=None)
        out_ind = os.path.join(carpeta, 'indice_agua.tif')
        with rasterio.open(out_ind, 'w', **perfil_ind) as dst:
            dst.write(indice, 1)

        # Guardar máscara de agua
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
