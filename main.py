import streamlit as st
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import io
import colorsys

st.title("🖼️ 이미지 필터 및 보정기")

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
    # Convert to HSV and shift hue
    img = img.convert('RGB')
    np_img = img.convert('RGB')
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

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="원본 이미지", use_column_width=True)

    filter_option = st.selectbox("적용할 필터 선택", ["없음", "흑백", "세피아", "블러"])

    # 보정 슬라이더
    st.markdown("### 이미지 보정")
    sharpness_val = st.slider("선명도", 0.0, 3.0, 1.0, 0.1)
    brightness_val = st.slider("밝기", 0.0, 3.0, 1.0, 0.1)
    contrast_val = st.slider("대비", 0.0, 3.0, 1.0, 0.1)
    hue_val = st.slider("색조 (Hue Shift)", -0.5, 0.5, 0.0, 0.01)

    # 필터 적용
    if filter_option == "흑백":
        filtered = ImageOps.grayscale(image)
        filtered = filtered.convert("RGB")  # 보정 위해 RGB 변환
    elif filter_option == "세피아":
        filtered = apply_sepia(image.copy())
    elif filter_option == "블러":
        filtered = image.filter(ImageFilter.BLUR)
    else:
        filtered = image.copy()

    # 보정 적용
    filtered = ImageEnhance.Sharpness(filtered).enhance(sharpness_val)
    filtered = ImageEnhance.Brightness(filtered).enhance(brightness_val)
    filtered = ImageEnhance.Contrast(filtered).enhance(contrast_val)
    if hue_val != 0.0:
        filtered = shift_hue(filtered, hue_val)

    st.image(filtered, caption=f"{filter_option} 필터 + 보정 적용된 이미지", use_column_width=True)

    # 다운로드 버튼
    buf = io.BytesIO()
    filtered.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="필터 적용 이미지 다운로드",
        data=byte_im,
        file_name="filtered_image.png",
        mime="image/png"
    )
