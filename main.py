import streamlit as st
import random
import math
import streamlit.components.v1 as components

# Streamlit Turn-Based Shooter
st.set_page_config(page_title="Turn-Based Shooter", layout="centered")

# --- settings ---
GRID_W = 10
GRID_H = 10
INITIAL_ENEMIES = 4
PLAYER_MAX_HP = 5

# --- session state init ---
if "player" not in st.session_state:
    st.session_state.player = {"x": GRID_W // 2, "y": GRID_H - 1, "hp": PLAYER_MAX_HP}
if "enemies" not in st.session_state:
    # enemy: dict with x,y,hp,id
    st.session_state.enemies = []
if "bullets" not in st.session_state:
    # bullet: dict with x,y,dx,dy,owner ('player' or 'enemy')
    st.session_state.bullets = []
if "score" not in st.session_state:
    st.session_state.score = 0
if "turn" not in st.session_state:
    st.session_state.turn = 1
if "msg" not in st.session_state:
    st.session_state.msg = "게임 시작!"
if "enemy_count" not in st.session_state:
    st.session_state.enemy_count = INITIAL_ENEMIES
if "game_over" not in st.session_state:
    st.session_state.game_over = False

# utility

def spawn_enemies(n):
    st.session_state.enemies = []
    ids = 0
    for _ in range(n):
        while True:
            x = random.randint(0, GRID_W - 1)
            y = random.randint(0, max(0, GRID_H // 2 - 1))  # spawn in upper half
            # don't spawn on player
            if not (x == st.session_state.player["x"] and y == st.session_state.player["y"]):
                break
        st.session_state.enemies.append({"id": ids, "x": x, "y": y, "hp": 1})
        ids += 1


def reset_game(enemy_count=None):
    st.session_state.player = {"x": GRID_W // 2, "y": GRID_H - 1, "hp": PLAYER_MAX_HP}
    st.session_state.bullets = []
    st.session_state.score = 0
    st.session_state.turn = 1
    st.session_state.msg = "새로운 게임"
    st.session_state.game_over = False
    if enemy_count is not None:
        st.session_state.enemy_count = enemy_count
    spawn_enemies(st.session_state.enemy_count)


# actions

def move_player(dx, dy):
    if st.session_state.game_over:
        return
    nx = max(0, min(GRID_W - 1, st.session_state.player["x"] + dx))
    ny = max(0, min(GRID_H - 1, st.session_state.player["y"] + dy))
    st.session_state.player["x"] = nx
    st.session_state.player["y"] = ny
    st.session_state.msg = f"플레이어 이동 -> ({nx}, {ny})"


def fire(dx, dy):
    if st.session_state.game_over:
        return
    # spawn bullet in front of player
    bx = st.session_state.player["x"] + dx
    by = st.session_state.player["y"] + dy
    if 0 <= bx < GRID_W and 0 <= by < GRID_H:
        st.session_state.bullets.append({"x": bx, "y": by, "dx": dx, "dy": dy, "owner": "player"})
        st.session_state.msg = f"발사! 방향 ({dx},{dy})"


def enemy_fire(enemy):
    # enemy fires towards player (normalized step)
    ex, ey = enemy["x"], enemy["y"]
    px, py = st.session_state.player["x"], st.session_state.player["y"]
    dx = px - ex
    dy = py - ey
    if dx == 0 and dy == 0:
        return
    # normalize to -1,0,1
    sdx = int(math.copysign(1, dx)) if dx != 0 else 0
    sdy = int(math.copysign(1, dy)) if dy != 0 else 0
    bx = ex + sdx
    by = ey + sdy
    if 0 <= bx < GRID_W and 0 <= by < GRID_H:
        st.session_state.bullets.append({"x": bx, "y": by, "dx": sdx, "dy": sdy, "owner": "enemy"})


def advance_bullets():
    new_bullets = []
    for b in st.session_state.bullets:
        nx = b["x"] + b["dx"]
        ny = b["y"] + b["dy"]
        # check bounds
        if not (0 <= nx < GRID_W and 0 <= ny < GRID_H):
            continue
        # collision with enemies (player bullets)
        if b["owner"] == "player":
            hit_enemy = None
            for e in st.session_state.enemies:
                if e["x"] == nx and e["y"] == ny:
                    hit_enemy = e
                    break
            if hit_enemy:
                st.session_state.score += 10
                try:
                    st.session_state.enemies.remove(hit_enemy)
                except ValueError:
                    pass
                st.session_state.msg = f"적 격파! 점수 +10 (총 {st.session_state.score})"
                continue
        # collision with player (enemy bullets)
        if b["owner"] == "enemy":
            if st.session_state.player["x"] == nx and st.session_state.player["y"] == ny:
                st.session_state.player["hp"] -= 1
                st.session_state.msg = f"플레이어가 피격당함! HP -1 (남은 HP: {st.session_state.player['hp']})"
                if st.session_state.player["hp"] <= 0:
                    st.session_state.game_over = True
                    st.session_state.msg = "플레이어 사망 — 게임 오버"
                continue
        # otherwise bullet continues
        new_bullets.append({"x": nx, "y": ny, "dx": b["dx"], "dy": b["dy"], "owner": b["owner"]})
    st.session_state.bullets = new_bullets


def enemies_act():
    # each enemy randomly moves one step (or stays), and has chance to fire
    for e in st.session_state.enemies[:]:
        if random.random() < 0.6:
            # move towards player with some randomness
            dx = st.session_state.player["x"] - e["x"]
            dy = st.session_state.player["y"] - e["y"]
            step_x = int(math.copysign(1, dx)) if dx != 0 and random.random() < 0.7 else (random.choice([-1,0,1]) if random.random()<0.3 else 0)
            step_y = int(math.copysign(1, dy)) if dy != 0 and random.random() < 0.7 else (random.choice([-1,0,1]) if random.random()<0.3 else 0)
            nx = max(0, min(GRID_W - 1, e["x"] + step_x))
            ny = max(0, min(GRID_H - 1, e["y"] + step_y))
            # don't move onto another enemy
            if not any(other["x"] == nx and other["y"] == ny for other in st.session_state.enemies if other is not e):
                e["x"] = nx
                e["y"] = ny
        # chance to fire
        if random.random() < 0.4:
            enemy_fire(e)
        # if enemy moves into player
        if e["x"] == st.session_state.player["x"] and e["y"] == st.session_state.player["y"]:
            st.session_state.player["hp"] -= 1
            st.session_state.msg = f"적과 충돌! HP -1 (남은 HP: {st.session_state.player['hp']})"
            if st.session_state.player["hp"] <= 0:
                st.session_state.game_over = True
                st.session_state.msg = "플레이어 사망 — 게임 오버"


# game step

def next_turn():
    if st.session_state.game_over:
        return
    st.session_state.turn += 1
    # advance bullets first (player fired this turn)
    advance_bullets()
    # enemies act (move + shoot)
    enemies_act()
    # bullets from enemies move
    advance_bullets()
    # spawn simple enemy reinforcements occasionally
    if st.session_state.turn % 10 == 0 and len(st.session_state.enemies) < GRID_W:
        # spawn one enemy at top row
        nx = random.randint(0, GRID_W - 1)
        st.session_state.enemies.append({"id": max([e["id"] for e in st.session_state.enemies], default=0) + 1, "x": nx, "y": 0, "hp": 1})
        st.session_state.msg = "보강 병력 도착!"
    # win condition
    if not st.session_state.enemies:
        st.session_state.msg = f"모든 적 제거! 스테이지 클리어 — 점수 {st.session_state.score}"
        st.session_state.game_over = True


# UI helpers

def render_grid_html():
    CELL = 36
    html = "<div style='font-family:Arial, monospace;'>"
    html += "<div style='display:inline-block; padding:8px; background:#111; border-radius:8px;'>"
    for y in range(GRID_H):
        html += "<div style='display:flex;'>"
        for x in range(GRID_W):
            style = f"width:{CELL}px; height:{CELL}px; margin:3px; border-radius:6px; display:inline-block;" + "background:linear-gradient(180deg,#222,#111); box-shadow:inset 0 1px 0 rgba(255,255,255,0.02);"
            char = ""
            # bullets
            is_bullet = any(b["x"] == x and b["y"] == y for b in st.session_state.bullets)
            if is_bullet:
                style += " background: radial-gradient(circle at 30% 30%, #fff, #ffea00 30%, #ff6f00 60%); box-shadow:0 0 6px rgba(255,140,0,0.6);"
            # player
            elif st.session_state.player["x"] == x and st.session_state.player["y"] == y:
                style += " background: linear-gradient(180deg,#3ec6ff,#006fb3); box-shadow:0 2px 6px rgba(0,0,0,0.5);"
            # enemy
            elif any(e["x"] == x and e["y"] == y for e in st.session_state.enemies):
                style += " background: linear-gradient(180deg,#ff6b6b,#b30000); box-shadow:0 2px 6px rgba(0,0,0,0.5);"
            html += f"<div style='{style}'></div>"
        html += "</div>"
    html += "</div></div>"
    return html


# Header
st.title("🎯 Streamlit 턴 기반 슈팅 게임")
col1, col2 = st.columns([2, 3])
with col1:
    st.markdown(f"**턴:** {st.session_state.turn}  ")
    st.markdown(f"**점수:** {st.session_state.score}  ")
    st.markdown(f"**플레이어 HP:** {st.session_state.player['hp']} / {PLAYER_MAX_HP}  ")
    st.markdown(f"**적 수:** {len(st.session_state.enemies)}  ")
    st.markdown(f"**메시지:** {st.session_state.msg}  ")

with col2:
    # controls: move
    mv1, mv2, mv3 = st.columns([1,1,1])
    with mv2:
        if st.button("⬆️ 이동"):
            move_player(0, -1)
            next_turn()
    row = st.columns([1,1,1])
    with row[0]:
        if st.button("⬅️ 이동"):
            move_player(-1, 0)
            next_turn()
    with row[1]:
        if st.button("⏭ 다음 턴(무동작)"):
            next_turn()
    with row[2]:
        if st.button("➡️ 이동"):
            move_player(1, 0)
            next_turn()
    with mv3:
        if st.button("⬇️ 이동"):
            move_player(0, 1)
            next_turn()

    st.markdown("---")
    # fire controls (4 directions)
    f1, f2, f3, f4 = st.columns([1,1,1,1])
    with f2:
        if st.button("🔼 발사"):
            fire(0, -1)
            next_turn()
    with f1:
        if st.button("◀️ 발사"):
            fire(-1, 0)
            next_turn()
    with f3:
        if st.button("▶️ 발사"):
            fire(1, 0)
            next_turn()
    with f4:
        if st.button("🔽 발사"):
            fire(0, 1)
            next_turn()

# right-side controls
st.sidebar.title("게임 설정 & 조작")
if st.sidebar.button("리셋"): 
    reset_game()

enemy_slider = st.sidebar.slider("초기 적 수", 1, 12, st.session_state.enemy_count)
if enemy_slider != st.session_state.enemy_count:
    st.sidebar.button("적 재배치 (적 수 적용)", key="apply_enemy")
    st.session_state.enemy_count = enemy_slider

if st.sidebar.button("새 게임 (설정 반영)"):
    reset_game(enemy_count=st.session_state.enemy_count)

if st.sidebar.button("자동 5턴 진행"):
    for _ in range(5):
        next_turn()

st.sidebar.markdown("---")
st.sidebar.markdown("**설명**: 이동 또는 발사 버튼을 눌러 행동하세요. 행동 후 '다음 턴'이 자동으로 진행됩니다. 적은 이동하거나 발사로 대응합니다.")

# render grid
board_html = render_grid_html()
components.html(board_html, height=460)

# end conditions
if st.session_state.game_over:
    st.warning("게임이 종료되었습니다. 사이드바의 '새 게임' 버튼으로 다시 시작하세요.")

# small tips
st.caption("팁: '자동 5턴 진행' 버튼으로 빠르게 턴을 감아볼 수 있어요. 추가 기능(무기 업그레이드, 아이템, 보스전)을 원하면 말해줘요!")
