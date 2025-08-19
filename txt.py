import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import io
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="핑크톤 이미지 편집기", page_icon="🌸", layout="wide")
st.title("🌸 핑크톤 이미지 편집기")
st.write("다양한 필터와 보정 기능을 제공하는 올인원 앱입니다 ✨")

# 업로드
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

# 세션 상태 저장
if "edited_image" not in st.session_state:
    st.session_state.edited_image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    if st.session_state.edited_image is None:
        st.session_state.edited_image = image.copy()

    st.sidebar.header("🛠 보정 및 필터 옵션")
    
    # 기본 보정
    brightness = st.sidebar.slider("밝기", 0.5, 2.0, 1.0)
    contrast = st.sidebar.slider("대비", 0.5, 2.0, 1.0)
    sharpness = st.sidebar.slider("선명도", 0.5, 2.0, 1.0)
    color = st.sidebar.slider("색감", 0.0, 2.0, 1.0)

    # 자유 회전
    angle = st.sidebar.slider("회전 (도)", 0, 360, 0)

    # 필터 선택
    filter_option = st.sidebar.selectbox("필터 선택", [
        "없음", "흑백", "세피아", "엣지", "블러", "만화 효과", "이진화"
    ])

    # 워터마크
    watermark_text = st.sidebar.text_input("워터마크 텍스트")

    # 랜덤 필터 버튼
    if st.sidebar.button("🎲 랜덤 필터 적용"):
        filter_option = random.choice(["흑백", "세피아", "엣지", "블러", "만화 효과", "이진화"])
        st.sidebar.write(f"선택된 랜덤 필터: {filter_option}")

    # 편집본 복사
    edited = image.copy()

    # 보정 적용
    edited = ImageEnhance.Brightness(edited).enhance(brightness)
    edited = ImageEnhance.Contrast(edited).enhance(contrast)
    edited = ImageEnhance.Sharpness(edited).enhance(sharpness)
    edited = ImageEnhance.Color(edited).enhance(color)

    # 회전 적용
    if angle != 0:
        edited = edited.rotate(angle, expand=True)

    # 필터 적용
    if filter_option == "흑백":
        edited = ImageOps.grayscale(edited)
    elif filter_option == "세피아":
        sepia = np.array(edited)
        tr = [0.393, 0.769, 0.189]
        tg = [0.349, 0.686, 0.168]
        tb = [0.272, 0.534, 0.131]
        R = sepia[:,:,0]*tr[0] + sepia[:,:,1]*tr[1] + sepia[:,:,2]*tr[2]
        G = sepia[:,:,0]*tg[0] + sepia[:,:,1]*tg[1] + sepia[:,:,2]*tg[2]
        B = sepia[:,:,0]*tb[0] + sepia[:,:,1]*tb[1] + sepia[:,:,2]*tb[2]
        sepia = np.stack([R,G,B], axis=2).clip(0,255).astype(np.uint8)
        edited = Image.fromarray(sepia)
    elif filter_option == "엣지":
        edited = edited.filter(ImageFilter.FIND_EDGES)
    elif filter_option == "블러":
        edited = edited.filter(ImageFilter.GaussianBlur(3))
    elif filter_option == "만화 효과":
        gray = ImageOps.grayscale(edited).filter(ImageFilter.MedianFilter(5))
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges = ImageOps.invert(edges)
        edges = edges.convert("L")
        edited = ImageOps.posterize(edited, 3)
        edited = Image.composite(edited, Image.new("RGB", edited.size, (255,255,255)), edges)
    elif filter_option == "이진화":
        gray = ImageOps.grayscale(edited)
        threshold = gray.point(lambda p: 255 if p > 128 else 0)
        edited = threshold

    # 워터마크 추가
    if watermark_text:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(edited)
        font = ImageFont.load_default()
        draw.text((10, 10), watermark_text, fill="pink", font=font)

    # 결과 저장
    st.session_state.edited_image = edited

    # 원본 vs 편집본 비교
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("편집본")
        st.image(edited, use_column_width=True)

    # 히스토그램 표시
    if st.checkbox("히스토그램 보기"):
        fig, ax = plt.subplots()
        arr = np.array(edited)
        if arr.ndim == 3:
            for i, color in enumerate(["red", "green", "blue"]):
                ax.hist(arr[:,:,i].flatten(), bins=256, alpha=0.5, color=color)
        else:
            ax.hist(arr.flatten(), bins=256, color="gray")
        st.pyplot(fig)

    # 다운로드
    buf = io.BytesIO()
    edited.save(buf, format="PNG")
    st.download_button("편집본 다운로드", buf.getvalue(), "edited.png", "image/png")
