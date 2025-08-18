import io
import time
from datetime import datetime
from typing import Tuple

import numpy as np
from PIL import Image

import streamlit as st

# --- Optional: suppress torch compile warnings on CPU only environments ---
st.set_page_config(page_title="AI Style Transfer", page_icon="🎨", layout="wide")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as T
    from torchvision.models import vgg19, VGG19_Weights
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    TORCH_ERR = e

# =============================
# Utilities
# =============================

def pil_to_tensor(img: Image.Image, max_size: int = 512) -> torch.Tensor:
    """Convert PIL to normalized tensor [1,3,H,W] in 0..1."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_size / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    tfm = T.Compose([
        T.ToTensor(),  # [0,1]
    ])
    t = tfm(img).unsqueeze(0)
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    return T.ToPILImage()(t.squeeze(0))


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # reshape to [C,1,1]
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor, weight: float = 1.0):
        super().__init__()
        self.target = target.detach()
        self.weight = weight
        self.loss = torch.tensor(0.0)

    def forward(self, x):
        self.loss = self.weight * nn.functional.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target_feature: torch.Tensor, weight: float = 1.0):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.weight = weight
        self.loss = torch.tensor(0.0)

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = self.weight * nn.functional.mse_loss(G, self.target)
        return x


def build_style_transfer_model(cnn: nn.Module,
                               normalization_mean,
                               normalization_std,
                               style_img: torch.Tensor,
                               content_img: torch.Tensor,
                               style_layers=None,
                               content_layers=None,
                               style_weight: float = 1e6,
                               content_weight: float = 1.0,
                               device: str = "cpu"):
    if style_layers is None:
        # Typical layers from Gatys paper / PyTorch tutorial
        style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    if content_layers is None:
        content_layers = ["conv4_2"]

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment for conv layers
    for layer in cnn.features.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv{i}_1" if i not in [1,2,3,4,5] else f"conv{i}_1"  # simple naming per block's first conv
        elif isinstance(layer, nn.ReLU):
            name = f"relu{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn{i}"
        else:
            name = f"layer_{len(model)}"
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target, content_weight)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature, style_weight)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim the model after the last loss to speed up
    for idx in range(len(model) - 1, -1, -1):
        if isinstance(model[idx], (ContentLoss, StyleLoss)):
            model = model[: idx + 1]
            break

    return model, style_losses, content_losses


def run_style_transfer(content_img: Image.Image,
                       style_img: Image.Image,
                       max_size: int = 512,
                       num_steps: int = 300,
                       style_weight: float = 1e6,
                       content_weight: float = 1.0,
                       lr: float = 0.03,
                       device: str = "cpu") -> Image.Image:
    """Run neural style transfer using Adam optimizer on the input image."""
    device = torch.device(device)

    content_t = pil_to_tensor(content_img, max_size=max_size).to(device)
    style_t = pil_to_tensor(style_img, max_size=max_size).to(device)

    # Start from content image for faster convergence
    input_img = content_t.clone().requires_grad_(True)

    # Load VGG19 weights
    weights = VGG19_Weights.DEFAULT
    cnn = vgg19(weights=weights).to(device).eval()

    # Normalization stats expected by VGG19
    norm_mean = weights.meta["mean"]
    norm_std = weights.meta["std"]

    model, style_losses, content_losses = build_style_transfer_model(
        cnn, norm_mean, norm_std, style_t, content_t,
        style_weight=style_weight, content_weight=content_weight, device=str(device)
    )

    optimizer = optim.Adam([input_img], lr=lr)

    pbar = st.progress(0, text="스타일 변환 진행 중…")
    status = st.empty()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        model(input_img)
        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        loss = style_score + content_score
        loss.backward()
        optimizer.step()

        if step % max(1, num_steps // 100) == 0:
            pbar.progress(step / num_steps, text=f"Step {step}/{num_steps} | Style: {style_score.item():.2f} | Content: {content_score.item():.2f}")
            status.text(f"Style Loss: {style_score.item():.2f} | Content Loss: {content_score.item():.2f}")

    result = tensor_to_pil(input_img)
    pbar.empty()
    status.empty()
    return result


# =============================
# UI
# =============================

st.title("🎨 AI Style Transfer (Neural Style Transfer)")
st.caption("컨텐츠 이미지에 스타일 이미지를 입혀 새로운 이미지를 생성하는 앱입니다. PyTorch + VGG19")

if not TORCH_OK:
    st.error("PyTorch/torchvision을 불러오지 못했습니다. 아래 오류를 확인하고 환경을 준비해 주세요.\n\n" + str(TORCH_ERR))
    st.stop()

with st.sidebar:
    st.header("① 이미지 업로드")
    content_file = st.file_uploader("컨텐츠 이미지 (사진)", type=["jpg", "jpeg", "png"])
    style_file = st.file_uploader("스타일 이미지 (그림/패턴)", type=["jpg", "jpeg", "png"])

    st.divider()
    st.header("② 파라미터")
    max_size = st.slider("최대 해상도 (긴변)", 256, 1024, 512, step=32, help="클수록 선명하지만 느려집니다.")
    num_steps = st.slider("반복 횟수 (steps)", 50, 800, 300, step=50)
    style_weight = st.number_input("Style Weight", value=1e6, min_value=1e3, max_value=1e8, step=1e5, format="%.0f")
    content_weight = st.number_input("Content Weight", value=1.0, min_value=0.0001, max_value=10.0, step=0.1, format="%.4f")
    lr = st.number_input("Learning Rate (Adam)", value=0.03, min_value=0.001, max_value=0.5, step=0.01, format="%.3f")

    st.divider()
    device_opt = "cuda" if torch.cuda.is_available() else "cpu"
    device = st.selectbox("장치 (Device)", [device_opt, "cpu", "cuda"], index=0 if device_opt=="cuda" else 1, help="GPU가 있다면 cuda를 선택하세요.")

st.markdown("""
**사용 방법**
1) 좌측 사이드바에서 컨텐츠 이미지(사진)와 스타일 이미지(그림/패턴)를 업로드합니다.  
2) 해상도/반복 횟수/가중치를 조절합니다.  
3) ✨ **'스타일 변환 실행'** 버튼을 누릅니다.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("컨텐츠 이미지")
    if content_file:
        content_img = Image.open(content_file)
        st.image(content_img, caption="Content", use_column_width=True)
    else:
        content_img = None
        st.info("사진을 업로드 해주세요.")

with col2:
    st.subheader("스타일 이미지")
    if style_file:
        style_img = Image.open(style_file)
        st.image(style_img, caption="Style", use_column_width=True)
    else:
        style_img = None
        st.info("스타일 이미지를 업로드 해주세요.")

st.divider()

run_btn = st.button("✨ 스타일 변환 실행", type="primary", use_container_width=True)

if run_btn:
    if content_img is None or style_img is None:
        st.warning("두 이미지를 모두 업로드해주세요.")
        st.stop()
    start = time.time()
    result = run_style_transfer(
        content_img=content_img,
        style_img=style_img,
        max_size=max_size,
        num_steps=int(num_steps),
        style_weight=float(style_weight),
        content_weight=float(content_weight),
        lr=float(lr),
        device=device,
    )
    elapsed = time.time() - start
    st.success(f"완료! 경과 시간: {elapsed:.1f}초")
    st.image(result, caption="Result", use_column_width=True)

    # Downloads
    buf_png = io.BytesIO()
    result.save(buf_png, format="PNG")
    buf_png.seek(0)

    buf_jpg = io.BytesIO()
    result.convert("RGB").save(buf_jpg, format="JPEG", quality=95)
    buf_jpg.seek(0)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ PNG 다운로드",
            data=buf_png,
            file_name=f"style_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "⬇️ JPG 다운로드",
            data=buf_jpg,
            file_name=f"style_transfer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

st.divider()

st.markdown(
    """
### ⚙️ 설치
```bash
pip install streamlit torch torchvision pillow
```

### 💡 팁
- 결과가 뭉개지면: **반복 횟수 증가** 또는 **LR 감소(0.01~0.03)**
- 스타일이 너무 쎄면: **Style Weight 감소** 또는 **Content Weight 증가**
- 빠르게 테스트: 해상도 384, Steps 150부터 시작
- GPU 사용 시: CUDA가 설치된 환경에서 장치 옵션을 `cuda`로 선택

### 📌 참고
- 본 코드는 VGG19 사전학습 특징을 이용한 **Neural Style Transfer (Gatys et al.)** 구현입니다.
"""
)
