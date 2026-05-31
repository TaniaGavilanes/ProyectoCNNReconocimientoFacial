"""
Sistema de detección y clasificación facial en tiempo real.

Este módulo combina el detector de rostros con el modelo de
clasificación para identificar personas en video en vivo.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras
from keras.layers import BatchNormalization, DepthwiseConv2D

# Agregar paths del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "2_src"))

# Importar módulos del proyecto
try:
    from .detector_rostro import FaceDetector
    from ..utils.config import IMAGE_SIZE, METADATA_DIR, STUDENTS_CSV, PROJECT_ROOT
    from ..utils.student_metadata import load_students_csv
except ImportError:
    from realtime.detector_rostro import FaceDetector
    from utils.config import IMAGE_SIZE, METADATA_DIR, STUDENTS_CSV, PROJECT_ROOT
    from utils.student_metadata import load_students_csv


def _resolve_project_path(path):
    """Convierte rutas relativas a absolutas respecto a la raíz del proyecto."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _patch_legacy_layer_loaders():
    """Parchea capas incompatibles entre Keras 2 y Keras 3 al cargar .h5."""
    patches = []

    original_bn = BatchNormalization.from_config

    @classmethod
    def bn_from_config(cls, config):
        config = dict(config)
        axis = config.get("axis")
        if isinstance(axis, (list, tuple)) and len(axis) == 1:
            config["axis"] = axis[0]
        return original_bn(config)

    BatchNormalization.from_config = bn_from_config
    patches.append((BatchNormalization, "from_config", original_bn))

    original_dw = DepthwiseConv2D.from_config

    @classmethod
    def dw_from_config(cls, config):
        config = dict(config)
        config.pop("groups", None)
        return original_dw(config)

    DepthwiseConv2D.from_config = dw_from_config
    patches.append((DepthwiseConv2D, "from_config", original_dw))

    return patches


def _restore_layer_loaders(patches):
    for layer_cls, attr, original in patches:
        setattr(layer_cls, attr, original)


def _load_weights_into_rebuilt_model(model_path, num_classes):
    """Reconstruye la arquitectura y carga solo los pesos del .h5."""
    from models.transfer_learning import (
        create_mobilenetv2_model,
        create_efficientnet_model,
    )

    path_str = str(model_path).lower()
    if "efficientnet" in path_str:
        model = create_efficientnet_model(
            input_shape=IMAGE_SIZE + (3,),
            num_classes=num_classes,
            freeze_base=True,
            weights=None,
        )
    else:
        model = create_mobilenetv2_model(
            input_shape=IMAGE_SIZE + (3,),
            num_classes=num_classes,
            freeze_base=True,
            weights=None,
        )

    model.load_weights(str(model_path))
    return model


def load_keras_model(model_path, compile_model=False, num_classes=None):
    """Carga modelos .h5 guardados con Keras 2 en entornos con Keras 3."""
    patches = _patch_legacy_layer_loaders()
    try:
        keras.config.enable_unsafe_deserialization()
        return keras.models.load_model(str(model_path), compile=compile_model)
    except (ValueError, TypeError, OSError) as exc:
        if num_classes is None:
            raise exc
        print(
            "[WARN] No se pudo deserializar el modelo completo; "
            "cargando pesos en arquitectura reconstruida..."
        )
        return _load_weights_into_rebuilt_model(model_path, num_classes)
    finally:
        _restore_layer_loaders(patches)


class RealTimeFaceRecognizer:
    """
    Clasificador de rostros en tiempo real.
    
    Esta clase coordina todo el proceso:
    1. Detectar rostros en el frame
    2. Preprocesar cada rostro
    3. Clasificar con el modelo entrenado
    4. Dibujar resultados en el frame
    """
    
    def __init__(
        self,
        model_path,
        labels_map_path=None,
        detection_method='haar',
        students_csv_path=None,
    ):
        """
        Inicializa el reconocedor.
        
        Args:
            model_path: Ruta al archivo .h5 del modelo entrenado
            labels_map_path: Ruta al labels_map.json (opcional)
            detection_method: 'haar' o 'mtcnn'
            students_csv_path: CSV con NControl, Nombre, Carrera, Semestre
        """
        if labels_map_path is None:
            labels_map_path = METADATA_DIR / "labels_map.json"

        with open(labels_map_path, 'r', encoding='utf-8') as f:
            labels_map = json.load(f)

        self.class_names = [labels_map[str(i)] for i in range(len(labels_map))]
        print(f"[INFO] Clases cargadas: {self.class_names}")

        print(f"[INFO] Cargando modelo desde {model_path}...")
        self.model = load_keras_model(
            model_path, num_classes=len(labels_map)
        )
        print("[INFO] Modelo cargado exitosamente")

        self.student_info = {}
        if students_csv_path is None:
            students_csv_path = STUDENTS_CSV
        students_csv_path = _resolve_project_path(students_csv_path)
        if students_csv_path.exists():
            self.student_info = load_students_csv(
                students_csv_path, self.class_names
            )
            print(f"[INFO] CSV de estudiantes: {students_csv_path}")
        else:
            print(
                f"[WARN] No se encontró CSV ({students_csv_path}). "
                "La etiqueta no mostrará NControl/Carrera/Semestre."
            )
        
        # Inicializar el detector de rostros
        self.face_detector = FaceDetector(method=detection_method)
        
        # Configuración
        self.target_size = IMAGE_SIZE
        self.confidence_threshold = 0.35
        self.label_font_scale = 0.55
        self.label_font_thickness = 2
        self.label_line_spacing = 22
    
    def preprocess_face(self, face_img):
        """
        Preprocesa una imagen de rostro para el modelo.
        
        El modelo espera:
        - Imagen en RGB (no BGR)
        - Tamaño 160x160
        - Valores normalizados entre 0 y 1
        - Batch dimension (1, 160, 160, 3)
        
        Args:
            face_img: Imagen del rostro en BGR
            
        Returns:
            Array listo para model.predict()
        """
        # Convertir de BGR a RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar si es necesario
        if face_rgb.shape[:2] != self.target_size:
            face_rgb = cv2.resize(face_rgb, self.target_size, 
                                 interpolation=cv2.INTER_AREA)
        
        # Normalizar pixeles a [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Agregar dimensión de batch
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def predict(self, face_img):
        """
        Predice la identidad de un rostro.
        
        Args:
            face_img: Imagen del rostro en BGR
            
        Returns:
            Tupla (nombre_clase, confianza, reconocido)
        """
        # Preprocesar imagen
        face_batch = self.preprocess_face(face_img)
        
        # Obtener predicciones (probabilidades para cada clase)
        predictions = self.model.predict(face_batch, verbose=0)
        
        # Encontrar la clase con mayor probabilidad
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        class_name = self.class_names[class_idx]
        recognized = confidence >= self.confidence_threshold
        return class_name, confidence, recognized

    def _build_label_lines(self, name, confidence, recognized):
        """Arma las líneas de texto para la etiqueta en pantalla."""
        info = self.student_info.get(name) if name else None

        if info:
            prefix = "" if recognized else "Posible - "
            return [
                f"{prefix}NControl: {info['ncontrol']}",
                f"Nombre: {info['nombre']}",
                f"Carrera: {info['carrera']}",
                f"Semestre: {info['semestre']}",
                f"Confianza: {confidence:.0%}",
            ]

        if name:
            prefix = "" if recognized else "Posible - "
            return [f"{prefix}{name}", f"Confianza: {confidence:.0%}"]

        return [f"Desconocido ({confidence:.0%})"]

    def _draw_label(self, frame, x, y, w, h, lines, color):
        """Dibuja una etiqueta de varias líneas junto al rostro (arriba o abajo)."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame_h, frame_w = frame.shape[:2]
        max_text_width = max(frame_w - x - 8, 120)

        scale = self.label_font_scale
        thickness = self.label_font_thickness
        spacing = self.label_line_spacing

        while scale >= 0.35:
            sizes = [
                cv2.getTextSize(line, font, scale, thickness)[0] for line in lines
            ]
            if max(size[0] for size in sizes) <= max_text_width:
                break
            scale -= 0.05
            spacing = max(16, int(self.label_line_spacing * scale / self.label_font_scale))

        box_width = max(size[0] for size in sizes) + 12
        box_height = len(lines) * spacing + 8

        space_above = y
        if space_above >= box_height + 10:
            label_y = y - 8
            top_left = (x, label_y - box_height)
            bottom_right = (x + box_width, label_y)
        else:
            top_left = (x, y + h + 8)
            bottom_right = (x + box_width, y + h + 8 + box_height)
            if bottom_right[1] > frame_h:
                top_left = (x, max(0, y - box_height - 8))
                bottom_right = (x + box_width, max(box_height, y - 8))

        bx2 = min(bottom_right[0], frame_w - 2)
        by2 = min(bottom_right[1], frame_h - 2)
        cv2.rectangle(frame, top_left, (bx2, by2), color, -1)

        for i, line in enumerate(lines):
            text_y = top_left[1] + (i + 1) * spacing
            cv2.putText(
                frame,
                line,
                (top_left[0] + 6, text_y),
                font,
                scale,
                (255, 255, 255),
                thickness,
            )
    
    def process_frame(self, frame):
        """
        Procesa un frame completo: detecta y clasifica todos los rostros.
        
        Args:
            frame: Frame de video en BGR
            
        Returns:
            Frame con bounding boxes y etiquetas dibujados
        """
        # Detectar todos los rostros
        faces = self.face_detector.detect_faces(frame)
        if faces is None or len(faces) == 0:
            return frame
        faces = [tuple(map(int, face)) for face in faces]

        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Extraer el rostro
            face_img = self.face_detector.extract_face(
                frame, (x, y, w, h), self.target_size
            )
            
            if face_img is None:
                continue
            
            name, confidence, recognized = self.predict(face_img)

            if recognized:
                color = (0, 255, 0)
            elif confidence >= 0.2:
                color = (0, 200, 255)
            else:
                color = (0, 0, 255)
            
            # Dibujar rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label_lines = self._build_label_lines(name, confidence, recognized)
            self._draw_label(frame, x, y, w, h, label_lines, color)
        
        return frame
