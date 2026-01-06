import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance

st.set_page_config(
    page_title="Fashion MNIST Classifier",
    layout="wide"
)

st.markdown(
    """
    <style>
    h1 { font-size: 36px; }
    h2 { font-size: 26px; }
    h3 { font-size: 20px; }

    p, li, span, div {
        font-size: 16px;
        line-height: 1.6;
    }

    .block-container {
        padding-left: 50px;
        padding-right: 50px;
        max-width: 1200px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

ALLOWED_FORMATS = ("PNG", "JPEG", "JPG")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Завантаження моделей
@st.cache_resource
def load_cnn():
    return tf.keras.models.load_model("cnn_model.h5")

@st.cache_resource
def load_vgg():
    return tf.keras.models.load_model("vgg16_model.keras")

# Заголовок та опис
st.title("Класифікація одягу (Fashion MNIST)")

st.markdown(
    """
Даний веб-застосунок призначений для класифікації зображень одягу за допомогою попередньо навченої згорткової нейронної мережі CNN
та моделі на основі VGG16, навчених на датасеті Fashion-MNIST. 

### **Як користуватися застосунком**

Оберіть модель у випадаючому списку: CNN або VGG16.
Завантажте зображення у форматі PNG / JPG / JPEG. Після завантаження зображення буде автоматично приведене до формату, необхідного для моделі, далі
буде виконана класифікація, результат якої відобразиться на екрані. 

### **Які зображення підходять найкраще**

Модель навчалась на датасеті Fashion-MNIST, тому найкращі результати отримуються для зображень, 
які:
- містять один предмет одягу 
- мають простий або однорідний фон
- показують предмет спереду 
- не містять зайвих деталей або тексту

Для реальних фотографій зі складним фоном точність може знижуватись, оскільки такі зображення відрізняються від навчальних даних. 

### **Графіки навчання моделі** 

На сторінці також відображаються графіки точності (Accuracy) та функції втрат (Loss) для обраної моделі. 
Ці графіки показують процес навчання моделі, відображають зміну метрик на тренувальній та валідаційній вибірках та 
дозволяють оцінити якість навчання. 

*Графіки не відносяться до окремого користувацького зображення, а характеризують загальну поведінку моделі під час навчання.* 
    """
)

st.divider()

# Вибір моделі
model_type = st.selectbox(
    "Оберіть модель:",
    ["CNN", "VGG16"]
)

model = load_cnn() if model_type == "CNN" else load_vgg()

# Завантаження зображення
uploaded_file = st.file_uploader(
    "Завантажте зображення (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"]
)

def preprocess_cnn(img: Image.Image):
    img = img.convert("L")
    img = ImageOps.invert(img) # Інверсія 
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.7) # Контраст
    # Центрування в квадраті
    w, h = img.size
    size = max(w, h)
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))
    img = canvas.resize((28, 28)) # Resize
    img = np.array(img).astype("float32") / 255.0
    img = img[..., np.newaxis]

    return np.expand_dims(img, axis=0)

def preprocess_vgg(img: Image.Image):
    img = img.convert("L")
    img = ImageOps.invert(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.7)

    w, h = img.size
    size = max(w, h)
    canvas = Image.new("L", (size, size), color=0)
    canvas.paste(img, ((size - w) // 2, (size - h) // 2))

    img = canvas.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0

    img = np.stack([img, img, img], axis=-1)

    return np.expand_dims(img, axis=0), img

if uploaded_file:
    col1, col2 = st.columns([1, 1.2])

    try:
        image = Image.open(uploaded_file)

        with col1:
            st.subheader("Вхідне зображення")
            st.image(image, width=280)

        if model_type == "CNN":
            x = preprocess_cnn(image)
        else:
            x, x_vis = preprocess_vgg(image)

        preds = model.predict(x)[0]
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

        with col2:
            st.subheader("Результат класифікації")
            st.write(f"**Передбачений клас:** {predicted_class}")
            st.write(f"**Впевненість моделі:** {confidence:.2f}%")

            if confidence < 60:
                st.info("Модель не впевнена у передбаченні для цього зображення.")

            st.subheader("Ймовірності класів")
            for name, p in zip(class_names, preds):
                st.write(name)
                st.progress(float(p))

        if model_type == "CNN":
            st.divider()
            st.subheader("Зображення після передобробки (CNN)")
            st.image(x[0, :, :, 0], width=180)
        else:
            st.divider()
            st.subheader("Зображення після передобробки (VGG16)")
            st.image(x_vis, width=180)

    except Exception as e:
        st.error("Помилка під час обробки зображення")
        st.caption(str(e))

# Графіки навчання
st.divider()
st.subheader("Графіки навчання моделі")

history = np.load(
    "history_cnn.npy" if model_type == "CNN" else "history_vgg.npy",
    allow_pickle=True
).item()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(history["accuracy"], label="Train")
ax[0].plot(history["val_accuracy"], label="Validation")
ax[0].set_title("Accuracy")
ax[0].legend()

ax[1].plot(history["loss"], label="Train")
ax[1].plot(history["val_loss"], label="Validation")
ax[1].set_title("Loss")
ax[1].legend()

st.pyplot(fig)

st.caption(
    "Графіки відображають процес навчання моделі та не залежать від конкретного користувацького зображення."
)