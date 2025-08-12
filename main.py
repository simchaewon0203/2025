import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io
import colorsys
import random

st.title("🖼️ 고급 이미지 필터 & 보정기")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

def apply_sepia(img):
    img = img.convert("RGB")
    width, height = img.size
    pixels = img.load()

    for py in range(height):
        for px in range(width):
            r, g, b = img.getpixel((px, py))

            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)

            r = min(255, tr)
            g = min(255, tg)
            b = min(255, tb)

            pixels[px, py] = (r, g, b)
    return img

def shift_hue(img, hue_shift):
    img = img.convert('RGB')
    pixels = img.load()
    width, height = img.size
    for x in range(width):
        for y in range(height):
            r, g, b = img.getpixel((x, y))
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
            nr = max(0, min(255, nr))
            ng = max(0, min(255, ng))
            nb = max(0, min(255, nb))
            pixels[x, y] = (nr, ng, nb)
    return img

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="원본 이미지", use_column_width=True)

    # 필터 선택
    filter_option = st.selectbox(
        "필터 선택",
        ["없음", "흑백", "세피아", "블러", "엠보스", "엣지 강화"]
    )

    # 보정 슬라이더
    st.markdown("### 이미지 보정")
    sharpness_val = st.slider("선명도", 0.0, 3.0, 1.0, 0.1)
    brightness_val = st.slider("밝기", 0.0, 3.0, 1.0, 0.1)
    contrast_val = st.slider("대비", 0.0, 3.0, 1.0, 0.1)
    saturation_val = st.slider("채도", 0.0, 3.0, 1.0, 0.1)
    hue_val = st.slider("색조 (Hue Shift)", -0.5, 0.5, 0.0, 0.01)
    noise_amount = st.slider("노이즈 양", 0.0, 0.2, 0.0, 0.01)

    # 변환 선택
    st.markdown("### 이미지 변환")
    rotate_angle = st.selectbox("회전", [0, 90, 180, 270])
    flip_horizontal = st.checkbox("좌우 반전")
    flip_vertical = st.checkbox("상하 반전")

    # 필터 적용
    filtered = image.copy()

    if filter_option == "흑백":
        filtered = ImageOps.grayscale(filtered)
        filtered = filtered.convert("RGB")
    elif filter_option == "세피아":
        filtered = apply_sepia(filtered)
    elif filter_option == "블러":
        filtered = filtered.filter(ImageFilter.BLUR)
    elif filter_option == "엠보스":
        filtered = filtered.filter(ImageFilter.EMBOSS)
    elif filter_option == "엣지 강화":
        filtered = filtered.filter(ImageFilter.EDGE_ENHANCE)

    # 보정 적용
    filtered = ImageEnhance.Sharpness(filtered).enhance(sharpness_val)
    filtered = ImageEnhance.Brightness(filtered).enhance(brightness_val)
    filtered = ImageEnhance.Contrast(filtered).enhance(contrast_val)
    filtered = ImageEnhance.Color(filtered).enhance(saturation_val)
    if hue_val != 0.0:
        filtered = shift_hue(filtered, hue_val)
    if noise_amount > 0.0:
        filtered = add_noise(filtered, noise_amount)

    # 변환 적용
    if rotate_angle != 0:
        filtered = filtered.rotate(rotate_angle, expand=True)
    if flip_horizontal:
        filtered = ImageOps.mirror(filtered)
    if flip_vertical:
        filtered = ImageOps.flip(filtered)

    st.image(filtered, caption="적용된 이미지", use_column_width=True)

    # 다운로드 버튼
    buf = io.BytesIO()
    filtered.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="이미지 다운로드",
        data=byte_im,
        file_name="edited_image.png",
        mime="image/png"
    )
