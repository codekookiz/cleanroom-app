import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  # TensorFlow 내장 Keras 사용
from PIL import Image, ImageOps
import numpy as np

# 📌 모델과 라벨 파일을 캐싱하여 불필요한 로딩 방지
@st.cache_resource
def load_model_and_labels():
    model = load_model("model/keras_model.h5")  # compile=False 제거
    class_names = open("model/labels.txt", "r", encoding="UTF-8").readlines()
    return model, class_names

# 📌 이미지 예측 함수
def load_and_predict(image, model, class_names):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 🔹 TensorFlow 1.x 스타일 예외 처리
    try:
        prediction = model.predict(data)
    except tf.errors.InvalidArgumentError:
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.keras.backend.set_session(sess)
            prediction = model.predict(data)

    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

# 📌 메인 UI 함수
def main():
    st.set_page_config(page_title="방 청결 상태 예측", page_icon="🧹", layout="centered")

    st.markdown("<h1 style='text-align: center;'>🧼 방 청결 상태 분석</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>방의 사진을 업로드하면 깨끗한지 더러운지 분석합니다.</h4>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📸 이미지를 업로드하세요", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="📷 업로드한 이미지", use_container_width=True)

        model, class_names = load_model_and_labels()

        with st.spinner("🔍 분석 중... 잠시만 기다려 주세요!"):
            image = Image.open(uploaded_file)
            class_name, confidence_score = load_and_predict(image, model, class_names)

        class_name_cleaned = " ".join(class_name.split()[1:])
        styled_result = f"<h2 style='text-align: center; color: #4CAF50;'>✨ 이 방은 <b>{class_name_cleaned}</b> 방입니다! ✨</h2>"
        st.markdown(styled_result, unsafe_allow_html=True)

        st.info(f"📊 **분석 정확도: `{(confidence_score * 100):.2f}%`**")
    
    else:
        st.warning("📢 이미지를 업로드해 주세요.")

if __name__ == "__main__":
    main()