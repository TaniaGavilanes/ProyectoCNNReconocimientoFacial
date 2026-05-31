"""Prueba rápida: modelo + CSV + una imagen (sin webcam)."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "2_src"))

import cv2
from realtime.detectar_y_clasificar import RealTimeFaceRecognizer

MODEL = project_root / "3_models/mobilenet_transfer_best.h5"
CSV = project_root / "1_dataset/metadata/estudiantes.csv"
TEST_IMAGES = [
    project_root / "1_dataset/processed/Karen/Karen_0001.jpg",
    project_root / "1_dataset/splits/test/Karen/Karen_0002.jpg",
    project_root / "1_dataset/raw/Karen/Karen (1).jpeg",
]
OUTPUT = project_root / "5_results/prueba_actual.jpg"


def main():
    print("=" * 50)
    print("PRUEBA CON CONFIGURACIÓN ACTUAL")
    print("=" * 50)

    if not MODEL.exists():
        print(f"[FALLO] No hay modelo: {MODEL}")
        return 1
    if not CSV.exists():
        print(f"[FALLO] No hay CSV: {CSV}")
        return 1
    test_image = next((p for p in TEST_IMAGES if p.exists()), None)
    if test_image is None:
        print("[FALLO] No hay imágenes de prueba en processed/test/raw")
        return 1

    print("[1/3] Cargando reconocedor...")
    recognizer = RealTimeFaceRecognizer(
        str(MODEL),
        students_csv_path=str(CSV),
    )

    print(f"[2/3] Procesando: {test_image}")
    frame = cv2.imread(str(test_image))
    if frame is None:
        print("[FALLO] No se pudo leer la imagen")
        return 1

    result = recognizer.process_frame(frame)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), result)
    print(f"[3/3] Resultado guardado en: {OUTPUT}")

    faces = recognizer.face_detector.detect_faces(frame)
    if faces is None or len(faces) == 0:
        h, w = frame.shape[:2]
        faces = [(0, 0, w, h)]
        print("[INFO] Imagen ya recortada: se usa el frame completo para clasificar")
    else:
        faces = [tuple(map(int, f)) for f in faces]

    x, y, w, h = faces[0]
    face_img = recognizer.face_detector.extract_face(frame, (x, y, w, h), recognizer.target_size)
    name, conf, recognized = recognizer.predict(face_img)

    print("-" * 50)
    if name and recognized:
        info = recognizer.student_info.get(name, {})
        print(f"Clase detectada : {name}")
        print(f"Confianza       : {conf:.1%}")
        if info:
            print(f"NControl        : {info.get('ncontrol', '-')}")
            print(f"Nombre (CSV)    : {info.get('nombre', '-')}")
            print(f"Carrera         : {info.get('carrera', '-')}")
            print(f"Semestre        : {info.get('semestre', '-')}")
        print("[OK] Prueba exitosa")
        return 0

    print(f"[AVISO] Rostro visto pero confianza baja ({conf:.1%})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
