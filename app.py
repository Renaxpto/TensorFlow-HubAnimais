import os
import tempfile
from io import BytesIO

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image, ImageDraw

from coco_labels import COCO_LABELS, ANIMAL_CLASSES


st.set_page_config(
    page_title="Identificação de Animais com TensorFlow",
    layout="wide"
)

st.title("Identificação de Animais em Imagens e Vídeos")
st.write("Aplicação desenvolvida com TensorFlow Hub para detetar animais em imagens e vídeos.")


@st.cache_resource
def load_model():
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    return hub.load(model_url)


def get_detections_from_image_np(image_np, model, threshold=0.40):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    results = model(input_tensor)

    boxes = results["detection_boxes"][0].numpy()
    scores = results["detection_scores"][0].numpy()
    classes = results["detection_classes"][0].numpy().astype(int)

    detections = []

    height, width, _ = image_np.shape

    for i in range(len(scores)):
        score = scores[i]
        class_id = classes[i]
        label = COCO_LABELS.get(class_id, "unknown")

        if score >= threshold and label in ANIMAL_CLASSES:
            ymin, xmin, ymax, xmax = boxes[i]

            left = int(xmin * width)
            top = int(ymin * height)
            right = int(xmax * width)
            bottom = int(ymax * height)

            detections.append({
                "animal": label,
                "confidence": round(float(score), 2),
                "box": (left, top, right, bottom)
            })

    return detections


def draw_detections_pil(image, detections):
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)

    for detection in detections:
        left, top, right, bottom = detection["box"]
        label = detection["animal"]
        confidence = detection["confidence"]

        draw.rectangle(
            [(left, top), (right, bottom)],
            outline="red",
            width=4
        )

        text = f"{label} - {confidence:.2f}"
        draw.text(
            (left, max(top - 20, 0)),
            text,
            fill="red"
        )

    return output_image


def draw_detections_cv2(frame, detections):
    for detection in detections:
        left, top, right, bottom = detection["box"]
        label = detection["animal"]
        confidence = detection["confidence"]

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 0, 255),
            3
        )

        text = f"{label} - {confidence:.2f}"

        cv2.putText(
            frame,
            text,
            (left, max(top - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    return frame


def process_image(image, model, threshold):
    image_np = np.array(image.convert("RGB"))
    detections = get_detections_from_image_np(image_np, model, threshold)
    output_image = draw_detections_pil(image, detections)
    return output_image, detections


def process_video(video_file, model, threshold, frame_skip):
    os.makedirs("resultados", exist_ok=True)

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(video_file.read())
    temp_input.close()

    input_path = temp_input.name
    output_path = "resultados/video_processado.mp4"

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return None, "Erro ao abrir o vídeo."

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 24

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    last_detections = []
    animal_counter = {}

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_detections = get_detections_from_image_np(
                frame_rgb,
                model,
                threshold
            )

            for detection in last_detections:
                animal = detection["animal"]
                animal_counter[animal] = animal_counter.get(animal, 0) + 1

        frame = draw_detections_cv2(frame, last_detections)
        writer.write(frame)

        frame_count += 1

        if total_frames > 0:
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.write(f"A processar fotograma {frame_count} de {total_frames}")

    cap.release()
    writer.release()

    try:
        os.remove(input_path)
    except:
        pass

    return output_path, animal_counter


model = load_model()

with st.sidebar:
    st.header("Definições")

    tipo = st.radio(
        "Tipo de análise",
        ["Imagem", "Vídeo"]
    )

    threshold = st.slider(
        "Confiança mínima",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05
    )

    if tipo == "Vídeo":
        frame_skip = st.slider(
            "Analisar 1 fotograma a cada X fotogramas",
            min_value=1,
            max_value=20,
            value=5,
            step=1
        )

        st.caption("Valor maior = processamento mais rápido. Valor menor = maior precisão.")

    st.markdown("---")
    st.subheader("Animais que a aplicação identifica")

    animais_pt = {
        "bird": "Pássaro",
        "cat": "Gato",
        "dog": "Cão",
        "horse": "Cavalo",
        "sheep": "Ovelha",
        "cow": "Vaca",
        "elephant": "Elefante",
        "bear": "Urso",
        "zebra": "Zebra",
        "giraffe": "Girafa"
    }

    for animal_en, animal_pt in animais_pt.items():
        st.write(f"- {animal_pt} (`{animal_en}`)")

    st.markdown("---")
    st.caption("Projeto: Identificação de Animais com TensorFlow")


if tipo == "Imagem":
    uploaded_file = st.file_uploader(
        "Escolhe uma imagem",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file)

        with col1:
            st.subheader("Imagem original")
            st.image(image, use_container_width=True)

        result_image, detections = process_image(image, model, threshold)

        with col2:
            st.subheader("Imagem processada")
            st.image(result_image, use_container_width=True)

        buffer = BytesIO()
        result_image.save(buffer, format="PNG")
        buffer.seek(0)

        if detections:
            st.success(f"Foram detetados {len(detections)} animal/animais.")
            st.table([
                {
                    "animal": d["animal"],
                    "confidence": d["confidence"]
                }
                for d in detections
            ])
        else:
            st.warning("Nenhum animal foi detetado com a confiança escolhida.")

        st.download_button(
            label="Descarregar imagem processada",
            data=buffer,
            file_name="imagem_processada.png",
            mime="image/png"
        )


else:
    uploaded_video = st.file_uploader(
        "Escolhe um vídeo",
        type=["mp4", "mov", "avi"]
    )

    st.info("Para vídeos grandes, aconselha-se a usar frame skip 5 ou superior para acelerar.")

    if uploaded_video is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vídeo original")
            st.video(uploaded_video)

        with col2:
            st.subheader("Vídeo processado")
            video_placeholder = st.empty()

        if st.button("Processar vídeo"):
            with st.spinner("A processar vídeo..."):
                output_path, result = process_video(
                    uploaded_video,
                    model,
                    threshold,
                    frame_skip
                )

            if output_path is None:
                st.error(result)
            else:
                st.success("Vídeo processado com sucesso!")

                with col2:
                    video_placeholder.video(output_path)

                stats_col1, stats_col2 = st.columns([1, 1])

                with stats_col1:
                    if result:
                        st.subheader("Animais detetados")
                        st.table([
                            {
                                "animal": animal,
                                "deteções em fotogramas": count
                            }
                            for animal, count in result.items()
                        ])
                    else:
                        st.warning("Nenhum animal foi detetado no vídeo.")

                with stats_col2:
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Descarregar vídeo processado",
                            data=file,
                            file_name="video_processado.mp4",
                            mime="video/mp4"
                        )