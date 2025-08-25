import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import colorsys, io

st.set_page_config(page_title="🎀 핑크톤 이미지 편집기", layout="centered")
st.title("🎀 핑크톤 이미지 편집기 40+ 기능 💖")
st.markdown("---")

uploaded_file = st.file_uploader("📤 이미지를 업로드하세요 (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

# --- 필터 함수 ---
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
    arr = np.array(img).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    hsv = np.vectorize(colorsys.rgb_to_hsv)(r, g, b)
    h, s, v = hsv
    h = (h + hue_shift) % 1.0
    rgb = np.vectorize(colorsys.hsv_to_rgb)(h, s, v)
    arr = np.dstack((rgb[0], rgb[1], rgb[2])) * 255
    return Image.fromarray(arr.astype("uint8"))

def add_noise(img, amount=0.05):
    arr = np.array(img)
    noise = np.random.randint(-int(255*amount), int(255*amount)+1, arr.shape, dtype='int16')
    arr = np.clip(arr.astype('int16') + noise, 0, 255).astype('uint8')
    return Image.fromarray(arr)

def gamma_correction(img, gamma=1.0):
    lut = [min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)]
    return img.point(lut * 3)

def posterize(img, bits=4):
    return ImageOps.posterize(img, bits)

def solarize(img, threshold=128):
    return ImageOps.solarize(img, threshold)

def color_balance(img, r_shift=0, g_shift=0, b_shift=0):
    arr = np.array(img).astype('int16')
    arr[..., 0] = np.clip(arr[..., 0] + r_shift, 0, 255)
    arr[..., 1] = np.clip(arr[..., 1] + g_shift, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] + b_shift, 0, 255)
    return Image.fromarray(arr.astype('uint8'))

def simple_color_temp(img, temp=0):
    return color_balance(img, r_shift=temp*10, b_shift=-temp*10)

def pixelate(img, pixel_size=10):
    w, h = img.size
    img_small = img.resize((max(1, w//pixel_size), max(1, h//pixel_size)), resample=Image.NEAREST)
    return img_small.resize((w, h), Image.NEAREST)

def sketch(img):
    gray = ImageOps.grayscale(img)
    inverted = ImageOps.invert(gray)
    blur = inverted.filter(ImageFilter.GaussianBlur(10))
    return Image.blend(gray, blur, 0.5).convert("RGB")

def oil_painting(img, radius=3):
    return img.filter(ImageFilter.ModeFilter(size=radius))

def glitch_effect(img, shift=5):
    arr = np.array(img)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    g = np.roll(g, shift, axis=0)
    b = np.roll(b, -shift, axis=1)
    glitched = np.stack([r, g, b], axis=2)
    return Image.fromarray(glitched)

def isolate_red(img):
    arr = np.array(img)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    mask = (r > 100) & (r > g*1.2) & (r > b*1.2)
    gray = ImageOps.grayscale(img)
    arr_gray = np.array(gray)
    arr_new = np.stack([arr_gray, arr_gray, arr_gray], axis=2)
    arr_new[mask] = arr[mask]
    return Image.fromarray(arr_new)

def vignette(img):
    w, h = img.size
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xv, yv = np.meshgrid(x, y)
    mask = np.sqrt(xv**2 + yv**2)
    mask = np.clip(1 - mask, 0, 1)
    arr = np.array(img).astype(np.float32)
    arr *= mask[..., None]
    return Image.fromarray(arr.astype("uint8"))

def add_border(img, border=20, color=(255,192,203)):
    return ImageOps.expand(img, border=border, fill=color)

# --- 비율별 크롭 ---
def crop_to_ratio(img, ratio_w, ratio_h):
    w, h = img.size
    target_ratio = ratio_w / ratio_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        # 너무 넓으면 가로 자르기
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # 너무 높으면 세로 자르기
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    return img

# --- 메인 앱 ---
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="✨ 원본 이미지", use_column_width=True)
    st.markdown("---")

    # 필터 선택
    filter_option = st.selectbox("🖌️ 필터 선택", [
        "없음","흑백","세피아","블러","엠보스","엣지 강화","샤픈","컨투어","스무딩",
        "윤곽선","디테일","포스터화","색상 반전","솔라라이즈","노이즈","모션 블러",
        "픽셀화","연필 스케치","유화","글리치","빨강만 남기기","비네팅"
    ])

    st.markdown("### 🎛️ 보정 옵션")
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

    st.markdown("### 🔄 변환 옵션")
    rotate_angle = st.selectbox("↪️ 회전", [0, 90, 180, 270])
    flip_horizontal = st.checkbox("↔️ 좌우 반전")
    flip_vertical = st.checkbox("↕️ 상하 반전")

    st.markdown("### ✂️ 추가 편집 기능")

    crop_on = st.checkbox("크롭 적용")
    if crop_on:
        ratio = st.selectbox("비율 선택", ["원본 유지", "1:1", "4:3", "16:9", "3:4", "9:16"])
    else:
        ratio = "
