import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io
import colorsys
import random

st.set_page_config(page_title="🎀 핑크톤 이미지 편집기 20+ 필터", layout="centered")

st.title("🎀 핑크톤 이미지 편집기 20+ 필터 & 보정 💖")

uploaded_file = st.file_uploader("📤 이미지를 업로드하세요 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# ---------------- 유틸 함수 ----------------

def apply_sepia(img):
    img = img.convert("RGB")
    pixels = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            tr = int(0.393*r + 0.769*g + 0.189*b)
            tg = int(0.349*r + 0.686*g + 0.168*b)
            tb = int(0.272*r + 0.534*g + 0.131*b)
            pixels[x, y] = (min(255, tr), min(255, tg), min(255, tb))
    return img

def shift_hue(img, hue_shift):
    img = img.convert('RGB')
    pixels = img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            h, s, v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
            h = (h + hue_shift) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            pixels[x, y] = (int(r*255), int(g*255), int(b*255))
    return img

def add_noise(img, amount=0.05):
    img = img.convert("RGB")
    pixels = img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            nr = int(r + random.randint(-int(255*amount), int(255*amount)))
            ng = int(g + random.randint(-int(255*amount), int(255*amount)))
            nb = int(b + random.randint(-int(255*amount), int(255*amount)))
            pixels[x, y] = (max(0, min(255, nr)), max(0, min(255, ng)), max(0, min(255, nb)))
    return img

def gamma_correction(img, gamma=1.0):
    inv_gamma = 1.0 / gamma
    img = img.convert("RGB")
    pixels = img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            r = int((r / 255.0) ** inv_gamma * 255)
            g = int((g / 255.0) ** inv_gamma * 255)
            b = int((b / 255.0) ** inv_gamma * 255)
            pixels[x, y] = (r, g, b)
    return img

def posterize(img, bits=4):
    return ImageOps.posterize(img, bits)

def solarize(img, threshold=128):
    return ImageOps.solarize(img, threshold)

def color_balance(img, r_shift=0, g_shift=0, b_shift=0):
    img = img.convert("RGB")
    pixels = img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            r = max(0, min(255, r + r_shift))
            g = max(0, min(255, g + g_shift))
            b = max(0, min(255, b + b_shift))
            pixels[x, y] = (r, g, b)
    return img

def simple_color_temp(img, temp=0):
    img = img.convert("RGB")
    pixels = img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]
            r = int(r + temp*20)
            b = int(b - temp*20)
            r = max(0, min(255, r))
            b = max(0, min(255, b))
            pixels[x, y] = (r, g, b)
    return img

# ---------------- 메인 ----------------

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="✨ 원본 이미지", use_column_width=True)

    filter_option = st.selectbox("🎨 필터 선택", [
        "없음", "흑백", "세피아", "블러", "엠보스", "엣지 강화", "샤픈", "컨투어", "스무딩",
        "윤곽선", "디테일", "포스터화", "색상 반전", "솔라라이즈", "노이즈"
    ])

    # 보정
    sharpness_val = st.slider("🔍 선명도", 0.0, 3.0, 1.0, 0.1)
    brightness_val = st.slider("💡 밝기", 0.0, 3.0, 1.0, 0.1)
    contrast_val = st.slider("⚖️ 대비", 0.0, 3.0, 1.0, 0.1)
    saturation_val = st.slider("🌈 채도", 0.0, 3.0, 1.0, 0.1)
    hue_val = st.slider("🎨 색조", -0.5, 0.5, 0.0, 0.01)
    gamma_val = st.slider("🔆 감마 보정", 0.1, 3.0, 1.0, 0.05)
    invert_colors = st.checkbox("🌚 색상 반전")
    noise_amount = st.slider("✨ 노이즈", 0.0, 0.2, 0.0, 0.01)
    color_temp_val = st.slider("🔥 색온도", -5, 5, 0, 1)
    r_shift = st.slider("🔴 R 이동", -100, 100, 0, 1)
    g_shift = st.slider("🟢 G 이동", -100, 100, 0, 1)
    b_shift = st.slider("🔵 B 이동", -100, 100, 0, 1)

    filtered = image.copy()

    # 필터 적용
    if filter_option == "흑백":
        filtered = ImageOps.grayscale(filtered).convert("RGB")
    elif filter_option == "세피아":
        filtered = apply_sepia(filtered)
    elif filter_option == "블러":
        filtered = filtered.filter(ImageFilter.BLUR)
    elif filter_option == "엠보스":
        filtered = filtered.filter(ImageFilter.EMBOSS)
    elif filter_option == "엣지 강화":
        filtered = filtered.filter(ImageFilter.EDGE_ENHANCE)
    elif filter_option == "샤픈":
        filtered = filtered.filter(ImageFilter.SHARPEN)
    elif filter_option == "컨투어":
        filtered = filtered.filter(ImageFilter.CONTOUR)
    elif filter_option == "스무딩":
        filtered = filtered.filter(ImageFilter.SMOOTH)
    elif filter_option == "윤곽선":
        filtered = filtered.filter(ImageFilter.FIND_EDGES)
    elif filter_option == "디테일":
        filtered = filtered.filter(ImageFilter.DETAIL)
    elif filter_option == "포스터화":
        filtered = posterize(filtered, bits=4)
    elif filter_option == "색상 반전":
        filtered = ImageOps.invert(filtered)
    elif filter_option == "솔라라이즈":
        filtered = solarize(filtered, threshold=128)
    elif filter_option == "노이즈":
        filtered = add_noise(filtered, amount=0.1)

    # ----------- 보정 적용 (항상 RGB로 변환 후) -----------
    filtered = filtered.convert("RGB")
    filtered = ImageEnhance.Sharpness(filtered).enhance(sharpness_val)
    filtered = ImageEnhance.Brightness(filtered).enhance(brightness_val)
    filtered = ImageEnhance.Contrast(filtered).enhance(contrast_val)
    filtered = ImageEnhance.Color(filtered).enhance(saturation_val)
    if hue_val != 0.0:
        filtered = shift_hue(filtered, hue_val)
    filtered = gamma_correction(filtered, gamma_val)
    if invert_colors:
        filtered = ImageOps.invert(filtered)
    if noise_amount > 0.0:
        filtered = add_noise(filtered, noise_amount)
    if color_temp_val != 0:
        filtered = simple_color_temp(filtered, color_temp_val)
    if any([r_shift, g_shift, b_shift]):
        filtered = color_balance(filtered, r_shift, g_shift, b_shift)

    st.image(filtered, caption="💖 적용된 이미지", use_column_width=True)

    # 다운로드
    buf = io.BytesIO()
    filtered.save(buf, format="PNG")
    st.download_button("📥 이미지 다운로드", buf.getvalue(), "pink_edited_image.png", "image/png")
else:
    st.info("📌 이미지를 업로드 해주세요!")
