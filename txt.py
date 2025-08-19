import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import colorsys, random
import numpy as np

st.set_page_config(page_title="🎀 핑크톤 이미지 편집기", layout="centered")
st.title("🎀 핑크톤 이미지 편집기 30+ 기능 💖")

uploaded_file = st.file_uploader("📤 이미지를 업로드하세요 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# --- 필터 함수 정의 ---

def apply_sepia(img):
    img = img.convert("RGB")
    pixels = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            r, g, b = pixels[x, y]
            tr = int(0.393*r + 0.769*g + 0.189*b)
            tg = int(0.349*r + 0.686*g + 0.168*b)
            tb = int(0.272*r + 0.534*g + 0.131*b)
            pixels[x, y] = (min(255, tr), min(255, tg), min(255, tb))
    return img

def shift_hue(img, hue_shift):
    img = img.convert("RGB")
    pixels = img.load()
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            r, g, b = pixels[x, y]
            h, s, v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
            h = (h + hue_shift) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            pixels[x, y] = (int(r*255), int(g*255), int(b*255))
    return img

def add_noise(img, amount=0.05):
    img = img.convert("RGB")
    arr = np.array(img)
    noise = np.random.randint(-int(255*amount), int(255*amount)+1, arr.shape, dtype='int16')
    arr = np.clip(arr.astype('int16') + noise, 0, 255).astype('uint8')
    return Image.fromarray(arr)

def gamma_correction(img, gamma=1.0):
    img = img.convert("RGB")
    lut = [min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)]
    return img.point(lut * 3)

def posterize(img, bits=4):
    return ImageOps.posterize(img, bits)

def solarize(img, threshold=128):
    return ImageOps.solarize(img, threshold)

def color_balance(img, r_shift=0, g_shift=0, b_shift=0):
    img = img.convert("RGB")
    arr = np.array(img)
    arr = arr.astype('int16')
    arr[..., 0] = np.clip(arr[..., 0] + r_shift, 0, 255)
    arr[..., 1] = np.clip(arr[..., 1] + g_shift, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] + b_shift, 0, 255)
    return Image.fromarray(arr.astype('uint8'))

def simple_color_temp(img, temp=0):
    return color_balance(img, r_shift=temp*10, b_shift=-temp*10)

# --- 메인 앱 ---

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="✨ 원본 이미지", use_column_width=True)

    st.markdown("## 🎨 필터 설정")
    filter_option = st.selectbox("필터 선택", [
        "없음", "흑백", "세피아", "블러", "엠보스", "엣지 강화", "샤픈", "컨투어", "스무딩",
        "윤곽선", "디테일", "포스터화", "색상 반전", "솔라라이즈", "노이즈", "모션 블러"
    ])

    st.markdown("## 🎛️ 보정 기능")
    sharpness_val = st.slider("🔍 선명도", 0.0, 3.0, 1.0)
    brightness_val = st.slider("💡 밝기", 0.0, 3.0, 1.0)
    contrast_val = st.slider("⚖️ 대비", 0.0, 3.0, 1.0)
    saturation_val = st.slider("🌈 채도", 0.0, 3.0, 1.0)
    hue_val = st.slider("🎨 색조(Hue)", -0.5, 0.5, 0.0)
    gamma_val = st.slider("🔆 감마 보정", 0.1, 3.0, 1.0)
    invert_colors = st.checkbox("🌚 색상 반전")
    noise_amount = st.slider("✨ 노이즈 추가", 0.0, 0.2, 0.0)
    color_temp_val = st.slider("🔥 색온도", -5, 5, 0)
    r_shift = st.slider("🔴 R 이동", -100, 100, 0)
    g_shift = st.slider("🟢 G 이동", -100, 100, 0)
    b_shift = st.slider("🔵 B 이동", -100, 100, 0)

    st.markdown("## 🔄 이미지 변환")
    rotate_angle = st.selectbox("↪️ 회전", [0, 90, 180, 270])
    flip_horizontal = st.checkbox("↔️ 좌우 반전")
    flip_vertical = st.checkbox("↕️ 상하 반전")

    # --- 필터 적용 ---
    filtered = image.copy()

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
        filtered = posterize(filtered)
    elif filter_option == "색상 반전":
        filtered = ImageOps.invert(filtered)
    elif filter_option == "솔라라이즈":
        filtered = solarize(filtered)
    elif filter_option == "노이즈":
        filtered = add_noise(filtered, amount=noise_amount)
    elif filter_option == "모션 블러":
        filtered = filtered.filter(ImageFilter.GaussianBlur(radius=2))

    # --- 보정 ---
    filtered = ImageEnhance.Sharpness(filtered).enhance(sharpness_val)
    filtered = ImageEnhance.Brightness(filtered).enhance(brightness_val)
    filtered = ImageEnhance.Contrast(filtered).enhance(contrast_val)
    filtered = ImageEnhance.Color(filtered).enhance(saturation_val)
    filtered = shift_hue(filtered, hue_val)
    filtered = gamma_correction(filtered, gamma_val)
    if invert_colors:
        filtered = ImageOps.invert(filtered)
    filtered = simple_color_temp(filtered, temp=color_temp_val)
    filtered = color_balance(filtered, r_shift, g_shift, b_shift)

    # --- 이미지 변환 ---
    if rotate_angle:
        filtered = filtered.rotate(rotate_angle, expand=True)
    if flip_horizontal:
        filtered = ImageOps.mirror(filtered)
    if flip_vertical:
        filtered = ImageOps.flip(filtered)

    # --- 결과 출력 ---
    st.markdown("## 🖼️ 편집된 이미지")
    st.image(filtered, use_column_width=True)

    # 다운로드 기능
    img_byte_arr = io.BytesIO()
    filtered.save(img_byte_arr, format="PNG")
    st.download_button(
        label="💾 편집된 이미지 다운로드",
        data=img_byte_arr.getvalue(),
        file_name="edited_image.png",
        mime="image/png"
    )
