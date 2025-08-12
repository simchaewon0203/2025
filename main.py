import streamlit as st
import random
import math
import streamlit.components.v1 as components

st.set_page_config(page_title="Streamlit Brick Breaker", layout="centered")

# --- settings ---
GRID_W = 12
GRID_H = 16
BRICK_ROWS = 4
PADDLE_WIDTH = 5
CELL_SIZE = 28  # pixels

# --- initialize session state ---
if "bricks" not in st.session_state:
    st.session_state.bricks = {(r, c) for r in range(BRICK_ROWS) for c in range(GRID_W)}
if "ball_x" not in st.session_state:
    st.session_state.ball_x = GRID_W // 2
if "ball_y" not in st.session_state:
    st.session_state.ball_y = BRICK_ROWS + 3
if "vel_x" not in st.session_state:
    st.session_state.vel_x = random.choice([-1, 1])
if "vel_y" not in st.session_state:
    st.session_state.vel_y = -1
if "paddle_x" not in st.session_state:
    st.session_state.paddle_x = (GRID_W - PADDLE_WIDTH) // 2
if "score" not in st.session_state:
    st.session_state.score = 0
if "lives" not in st.session_state:
    st.session_state.lives = 3
if "running_steps" not in st.session_state:
    st.session_state.running_steps = 0
if "level" not in st.session_state:
    st.session_state.level = 1

# --- game logic ---
def reset_ball():
    st.session_state.ball_x = GRID_W // 2
    st.session_state.ball_y = BRICK_ROWS + 3
    st.session_state.vel_x = random.choice([-1, 1])
    st.session_state.vel_y = -1


def advance_step():
    # move ball step-by-step with collision detection
    x = st.session_state.ball_x
    y = st.session_state.ball_y
    vx = st.session_state.vel_x
    vy = st.session_state.vel_y

    nx = x + vx
    ny = y + vy

    # wall collisions (left/right/top)
    if nx < 0:
        nx = 0
        vx *= -1
    if nx >= GRID_W:
        nx = GRID_W - 1
        vx *= -1
    if ny < 0:
        ny = 0
        vy *= -1

    # brick collision
    if (ny, nx) in st.session_state.bricks:
        st.session_state.bricks.remove((ny, nx))
        st.session_state.score += 10
        vy *= -1
        # small chance to change horizontal direction
        if random.random() < 0.2:
            vx *= -1

    # paddle collision
    paddle_row = GRID_H - 1
    if ny == paddle_row:
        if st.session_state.paddle_x <= nx < st.session_state.paddle_x + PADDLE_WIDTH:
            # reflect ball
            vy *= -1
            # change horizontal velocity based on hit position
            hit_pos = nx - st.session_state.paddle_x
            center = (PADDLE_WIDTH - 1) / 2
            delta = (hit_pos - center)
            # scale delta to -1..1
            vx = int(math.copysign(1, delta)) if abs(delta) >= 0.5 else vx
            ny = paddle_row - 1
        else:
            # missed paddle -> lose life
            st.session_state.lives -= 1
            if st.session_state.lives > 0:
                reset_ball()
                st.warning("패들을 놓쳤어요! 남은 목숨: {}".format(st.session_state.lives))
                return
            else:
                st.session_state.running_steps = 0
                st.error("💀 게임 오버! 최종 점수: {}".format(st.session_state.score))
                return

    # floor detection (below paddle)
    if ny >= GRID_H:
        st.session_state.lives -= 1
        if st.session_state.lives > 0:
            reset_ball()
            return
        else:
            st.session_state.running_steps = 0
            st.error("💀 게임 오버! 최종 점수: {}".format(st.session_state.score))
            return

    st.session_state.ball_x = nx
    st.session_state.ball_y = ny
    st.session_state.vel_x = vx
    st.session_state.vel_y = vy

    # level up when bricks cleared
    if not st.session_state.bricks:
        st.session_state.level += 1
        st.session_state.score += 100 * st.session_state.level
        st.success(f"레벨 업! 레벨 {st.session_state.level} — 브릭 재배치")
        # new brick formation: more rows (capped)
        new_rows = min(BRICK_ROWS + st.session_state.level - 1, GRID_H // 3)
        st.session_state.bricks = {(r, c) for r in range(new_rows) for c in range(GRID_W)}
        reset_ball()


# --- UI ---
st.title("🧱 Streamlit 벽돌깨기 — 실사 느낌 모드 (턴/버튼 기반)")
col_status, col_controls = st.columns([2, 3])
with col_status:
    st.markdown(f"**점수:** {st.session_state.score}  ")
    st.markdown(f"**목숨:** {st.session_state.lives}  ")
    st.markdown(f"**레벨:** {st.session_state.level}  ")

with col_controls:
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    if c1.button("⬅️ 왼쪽"):
        st.session_state.paddle_x = max(0, st.session_state.paddle_x - 1)
    if c3.button("➡️ 오른쪽"):
        st.session_state.paddle_x = min(GRID_W - PADDLE_WIDTH, st.session_state.paddle_x + 1)
    if c2.button("▶ 한 스텝"):
        advance_step()
    if c4.button("🔁 10스텝 자동진행"):
        # advance multiple steps in one click for pseudo-animation
        for _ in range(10):
            advance_step()

# speed control for how many steps to run when auto clicked
steps = st.slider("자동 진행 스텝 수", 1, 50, 10)
if st.button("🔁 자동 진행 (설정한 스텝 수만큼)"):
    for _ in range(steps):
        advance_step()

if st.button("⟲ 리스타트 (점수 유지 여부 선택)"):
    # provide a mini-choice
    if st.checkbox("점수 리셋", value=False):
        st.session_state.score = 0
    st.session_state.lives = 3
    st.session_state.level = 1
    st.session_state.bricks = {(r, c) for r in range(BRICK_ROWS) for c in range(GRID_W)}
    reset_ball()
    st.experimental_rerun()

# --- render board as HTML grid for nicer visuals ---

def render_board_html():
    html = f"""
    <div style='font-family:monospace;'>
    <div style='display:inline-block; background:#111; padding:10px; border-radius:8px;'>
    """
    for r in range(GRID_H):
        html += "<div style='display:flex; line-height:0;'>"
        for c in range(GRID_W):
            style = "width:%dpx; height:%dpx; margin:2px; border-radius:4px; display:inline-block;" % (CELL_SIZE, CELL_SIZE)
            if (r, c) in st.session_state.bricks:
                # color by row for variety
                hue = int(200 - (r * 10)) % 360
                style += f" background: linear-gradient(135deg, hsl({hue} 70% 50%), hsl({(hue+30)%360} 70% 40%)); box-shadow: 0 2px 4px rgba(0,0,0,0.4);"
            elif r == st.session_state.ball_y and c == st.session_state.ball_x:
                style += " background: radial-gradient(circle at 30% 30%, #ffffff, #b3e5ff 40%, #0095DD 60%); box-shadow: 0 0 6px rgba(0,0,0,0.6);"
            elif r == GRID_H - 1 and st.session_state.paddle_x <= c < st.session_state.paddle_x + PADDLE_WIDTH:
                style += " background: linear-gradient(90deg,#2b8cff,#0057b7); box-shadow: 0 2px 6px rgba(0,0,0,0.5);"
            else:
                style += " background: linear-gradient(180deg,#222,#111);"
            html += f"<div style='{style}'></div>"
        html += "</div>"
    html += "</div></div>"
    return html

board_html = render_board_html()
components.html(board_html, height=(CELL_SIZE+6) * GRID_H + 40)

# --- helpful tips ---
st.markdown("**팁:** 방향 버튼으로 패들을 옮기고 '한 스텝' 또는 '자동 진행'을 눌러 플레이하세요.\n\n한 번에 여러 스텝을 진행하면 실제 게임처럼 빠른 진행을 흉내낼 수 있습니다.")

# --- footer ---
st.caption("Streamlit만 사용한 턴 기반 벽돌깨기 — 추가 기능 원하면 말해줘 (사운드, 레벨별 브릭 패턴, 랭킹 등)")
