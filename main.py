import streamlit as st
from PIL import Image, ImageFilter, ImageOps
import io

st.title("🖼️ 이미지 필터 및 변환기")

uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="원본 이미지", use_column_width=True)

    filter_option = st.selectbox("적용할 필터 선택", ["없음", "흑백", "세피아", "블러"])

    def apply_sepia(img):
        width, height = img.size
        pixels = img.load()  # 픽셀 접근

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

    if filter_option == "흑백":
        filtered = ImageOps.grayscale(image)
    elif filter_option == "세피아":
        filtered = image.convert("RGB")
        filtered = apply_sepia(filtered)
    elif filter_option == "블러":
        filtered = image.filter(ImageFilter.BLUR)
    else:
        filtered = image

    st.image(filtered, caption=f"{filter_option} 필터 적용된 이미지", use_column_width=True)

    # 저장 버튼
    buf = io.BytesIO()
    filtered.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="필터 적용 이미지 다운로드",
        data=byte_im,
        file_name="filtered_image.png",
        mime="image/png"
    )
