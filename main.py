import streamlit as st
import random

st.set_page_config(page_title="Streamlit 경마 게임", layout="centered")

TRACK_LENGTH = 30
HORSE_COUNT = 6
START_MONEY = 1000

# --- 초기 상태 세팅 ---
if "money" not in st.session_state:
    st.session_state.money = START_MONEY
if "horses" not in st.session_state:
    # 각 말 위치
    st.session_state.horses = [0] * HORSE_COUNT
if "bet" not in st.session_state:
    st.session_state.bet = None
if "bet_amount" not in st.session_state:
    st.session_state.bet_amount = 0
if "race_over" not in st.session_state:
    st.session_state.race_over = False
if "winner" not in st.session_state:
    st.session_state.winner = None

# --- 함수 ---
def reset_race():
    st.session_state.horses = [0] * HORSE_COUNT
    st.session_state.bet = None
    st.session_state.bet_amount = 0
    st.session_state.race_over = False
    st.session_state.winner = None

def advance_turn():
    if st.session_state.race_over:
        return
    for i in range(HORSE_COUNT):
        move = random.randint(1, 3)  # 1~3칸 이동
        st.session_state.horses[i] += move
        if st.session_state.horses[i] >= TRACK_LENGTH:
            st.session_state.horses[i] = TRACK_LENGTH
            st.session_state.race_over = True
            st.session_state.winner = i

def render_track(horse_pos):
    track = ""
    for i in range(TRACK_LENGTH + 1):
        if i == horse_pos:
            track += "🐎"
        elif i == TRACK_LENGTH:
            track += "🏁"
        else:
            track += "-"
    return track

# --- UI ---
st.title("🏇 Streamlit 경마 게임")
st.markdown(f"**현재 보유 머니:** {st.session_state.money} 원")

if not st.session_state.race_over:
    st.markdown("### 베팅하기")
    horse_selected = st.radio("어느 말에 베팅하시겠습니까?", list(range(1, HORSE_COUNT+1)), horizontal=True)
    bet_amount = st.number_input("베팅 금액을 입력하세요", min_value=1, max_value=st.session_state.money, value=1, step=1)

    if st.button("베팅 완료"):
        st.session_state.bet = horse_selected - 1
        st.session_state.bet_amount = bet_amount
        st.success(f"말 {horse_selected}번에 {bet_amount}원 베팅 완료!")

if st.session_state.bet is not None:
    st.markdown(f"### 경주 진행 (말 {st.session_state.bet + 1}번에 베팅 중)")
    if not st.session_state.race_over:
        if st.button("한 턴 진행"):
            advance_turn()

    for idx, pos in enumerate(st.session_state.horses):
        st.markdown(f"말 {idx+1}: {render_track(pos)}")

if st.session_state.race_over:
    st.success(f"🏆 말 {st.session_state.winner + 1}번이 우승했습니다!")
    if st.session_state.bet == st.session_state.winner:
        win_money = st.session_state.bet_amount * 2
        st.session_state.money += win_money
        st.success(f"축하합니다! 베팅에 성공하여 {win_money}원을 벌었습니다.")
    else:
        st.session_state.money -= st.session_state.bet_amount
        st.error(f"아쉽네요. 베팅에 실패하여 {st.session_state.bet_amount}원을 잃었습니다.")

    if st.button("새 경주 시작"):
        reset_race()

if st.session_state.money <= 0:
    st.error("더 이상 베팅할 머니가 없습니다. 게임을 종료하거나 새로 시작하세요.")

st.caption("한 턴씩 버튼을 눌러 경주를 진행하세요. 베팅에 성공하면 머니가 늘어납니다!")
