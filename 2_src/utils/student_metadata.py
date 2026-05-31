"""
Carga metadatos de estudiantes desde CSV para mostrar en la etiqueta en vivo.

El CSV debe incluir una columna que coincida con la carpeta del dataset
(columna 'carpeta') o con el nombre de clase en labels_map (columna 'nombre').
"""

import csv
from pathlib import Path


REQUIRED_FIELDS = ("ncontrol", "nombre", "carrera", "semestre")
FOLDER_ALIASES = ("carpeta", "folder", "clase", "persona", "label")
NAME_ALIASES = ("nombre", "name")


def _normalize_header(name):
    return name.strip().lower().replace(" ", "_")


def _pick_column(headers, aliases):
    for alias in aliases:
        if alias in headers:
            return alias
    return None


def load_students_csv(csv_path, class_names=None):
    """
    Lee el CSV y devuelve un dict: nombre_de_clase -> metadatos.

    Args:
        csv_path: Ruta al archivo .csv
        class_names: Lista de nombres del labels_map (para validar coincidencias)

    Returns:
        dict[str, dict] con claves ncontrol, nombre, carrera, semestre
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")

    students = {}

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("El CSV está vacío o no tiene encabezados")

        headers = {_normalize_header(h): h for h in reader.fieldnames}
        norm_keys = set(headers.keys())

        missing = [f for f in REQUIRED_FIELDS if f not in norm_keys]
        if missing:
            raise ValueError(
                f"Faltan columnas en el CSV: {', '.join(missing)}. "
                f"Se requieren: NControl, Nombre, Carrera, Semestre"
            )

        folder_key = _pick_column(norm_keys, FOLDER_ALIASES)
        name_key = _pick_column(norm_keys, NAME_ALIASES)

        for row_num, row in enumerate(reader, start=2):
            normalized_row = {
                _normalize_header(k): (v or "").strip()
                for k, v in row.items()
            }

            record = {
                "ncontrol": normalized_row["ncontrol"],
                "nombre": normalized_row["nombre"],
                "carrera": normalized_row["carrera"],
                "semestre": normalized_row["semestre"],
            }

            if not all(record.values()):
                print(f"[WARN] Fila {row_num} ignorada: campos incompletos")
                continue

            class_key = None
            if folder_key and normalized_row.get(folder_key):
                class_key = normalized_row[folder_key]
            elif name_key:
                class_key = normalized_row[name_key]

            if not class_key:
                print(f"[WARN] Fila {row_num} ignorada: sin carpeta/nombre de clase")
                continue

            students[class_key] = record

    if class_names:
        for name in class_names:
            if name not in students:
                print(
                    f"[WARN] Sin datos en CSV para la clase '{name}'. "
                    "Se mostrará solo el nombre detectado."
                )

    print(f"[INFO] Metadatos cargados para {len(students)} persona(s) desde CSV")
    return students
