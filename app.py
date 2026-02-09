# app.py
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")
st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관 체크인 → 달성률/차트 → 날씨/강아지 + AI 코치 리포트까지!")

# -----------------------------
# Sidebar: API Keys
# -----------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="AI 리포트 생성에 필요")
    owm_api_key = st.text_input("OpenWeatherMap API Key", type="password", help="날씨 불러오기에 필요")
    st.divider()
    st.caption("Tip: 키는 세션에서만 사용되며 저장되지 않아요.")

# -----------------------------
# Constants
# -----------------------------
HABITS = [
    ("기상 미션", "⏰"),
    ("물 마시기", "💧"),
    ("공부/독서", "📚"),
    ("운동하기", "🏃"),
    ("수면", "😴"),
]

# ✅ OpenWeatherMap 모호성/404 방지: “도시,KR” 형태로 고정
CITY_OPTIONS = [
    ("Seoul", "Seoul,KR"),
    ("Busan", "Busan,KR"),
    ("Incheon", "Incheon,KR"),
    ("Daegu", "Daegu,KR"),
    ("Daejeon", "Daejeon,KR"),
    ("Gwangju", "Gwangju,KR"),
    ("Ulsan", "Ulsan,KR"),
    ("Suwon", "Suwon,KR"),
    ("Changwon", "Changwon,KR"),
    ("Jeju", "Jeju City,KR"),
]

COACH_STYLES = {
    "스파르타 코치": "엄격하고 직설적이며 행동을 강하게 요구하는 코치",
    "따뜻한 멘토": "다정하고 공감하며 작은 성취도 크게 칭찬하는 멘토",
    "게임 마스터": "RPG 퀘스트/레벨업 톤으로 재미있게 이끄는 게임 마스터",
}

# -----------------------------
# Session State Init
# -----------------------------
def init_demo_history():
    """데모용 6일 샘플 데이터 생성"""
    today = datetime.now().date()
    rows = []
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        achieved = max(0, min(5, 1 + (i % 5)))
        mood = max(1, min(10, 5 + (2 - (i % 5))))
        rows.append(
            {
                "date": d.isoformat(),
                "achieved": achieved,
                "rate": round(achieved / 5 * 100, 1),
                "mood": mood,
            }
        )
    return rows


if "history" not in st.session_state:
    st.session_state["history"] = init_demo_history()  # 6일
if "latest_report" not in st.session_state:
    st.session_state["latest_report"] = None
if "latest_share_text" not in st.session_state:
    st.session_state["latest_share_text"] = None

# -----------------------------
# API Helpers
# -----------------------------
def get_weather(city_query: str, api_key: str):
    """
    OpenWeatherMap에서 날씨 가져오기 (한국어, 섭씨)
    - 실패 시 (None, 에러메시지) 반환
    - timeout=10
    """
    if not city_query or not api_key:
        return None, "Missing city or API key"

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_query,
        "appid": api_key.strip(),
        "units": "metric",
        "lang": "kr",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            try:
                msg = r.json().get("message", "")
            except Exception:
                msg = (r.text or "")[:200]
            return None, f"HTTP {r.status_code}: {msg}"

        data = r.json()
        weather_desc = (data.get("weather") or [{}])[0].get("description")
        main = data.get("main", {}) or {}
        wind = data.get("wind", {}) or {}
        return (
            {
                "city": city_query,
                "description": weather_desc,
                "temp_c": main.get("temp"),
                "feels_like_c": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "wind_ms": wind.get("speed"),
            },
            None,
        )
    except Exception as e:
        return None, f"Exception: {e}"


def extract_breed_from_url(image_url: str):
    """Dog CEO 이미지 URL에서 품종 추정"""
    try:
        breed_part = image_url.split("/breeds/")[1].split("/")[0]
        breed_part = breed_part.replace("-", " ")
        words = breed_part.split()
        if len(words) >= 2:
            return f"{words[1].title()} {words[0].title()}"
        return breed_part.title()
    except Exception:
        return "Unknown"


def get_dog_image():
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기
    - 실패 시 None 반환
    - timeout=10
    """
    url = "https://dog.ceo/api/breeds/image/random"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None
        image_url = data.get("message")
        if not image_url:
            return None
        return {"image_url": image_url, "breed": extract_breed_from_url(image_url)}
    except Exception:
        return None


def system_prompt_for_style(style: str) -> str:
    """코치 스타일별 시스템 프롬프트"""
    if style == "스파르타 코치":
        return (
            "너는 매우 엄격하고 직설적인 코치다. "
            "핑계를 허용하지 않고, 구체적 행동을 강하게 요구한다. "
            "짧고 임팩트 있게 말하되, 실천 가능한 지시를 반드시 포함해라."
        )
    if style == "게임 마스터":
        return (
            "너는 RPG 세계관의 게임 마스터다. "
            "사용자는 플레이어이며, 습관은 퀘스트/스탯/레벨업으로 표현한다. "
            "재미있고 몰입감 있게, 하지만 실제로 실행 가능한 조언을 제공해라."
        )
    return (
        "너는 따뜻하고 공감하는 멘토다. "
        "사용자의 노력과 감정을 인정하고, 작은 성취도 칭찬한다. "
        "부담 없는 다음 행동을 제안해라."
    )


def openai_call(openai_key: str, system_prompt: str, user_prompt: str):
    """
    OpenAI 호출 래퍼
    - 모델: gpt-5-mini
    - Responses API 우선, 실패 시 Chat Completions fallback
    - 실패 시 (None, err) 반환
    """
    if not openai_key:
        return None, "OpenAI API Key가 필요해요."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=openai_key.strip())

        # Responses API (우선)
        try:
            resp = client.responses.create(
                model="gpt-5-mini",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            text = getattr(resp, "output_text", None)
            if not text:
                text = str(resp)
            return text, None
        except Exception:
            # Chat Completions fallback
            chat = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return chat.choices[0].message.content, None

    except Exception as e:
        return None, f"OpenAI 호출 실패: {e}"


def generate_report(openai_key: str, coach_style: str, habits_checked: dict, mood: int, weather: dict | None, dog: dict | None):
    """습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달"""
    achieved = sum(1 for v in habits_checked.values() if v)
    rate = achieved / 5 * 100

    habit_lines = []
    for name, emoji in HABITS:
        ok = habits_checked.get(name, False)
        habit_lines.append(f"- {emoji} {name}: {'완료' if ok else '미완료'}")

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')} / {weather.get('description')} / "
            f"{weather.get('temp_c')}°C(체감 {weather.get('feels_like_c')}°C) / "
            f"습도 {weather.get('humidity')}% / 바람 {weather.get('wind_ms')}m/s"
        )

    dog_text = "강아지 정보 없음"
    if dog:
        dog_text = f"{dog.get('breed')} (이미지 URL 제공됨)"

    sys_p = system_prompt_for_style(coach_style)

    user_p = (
        "[오늘 체크인 요약]\n"
        f"달성률: {rate:.0f}%\n"
        f"완료 습관 수: {achieved}/5\n"
        f"기분(1~10): {mood}\n\n"
        "[습관 상세]\n"
        + "\n".join(habit_lines)
        + "\n\n[날씨]\n"
        + weather_text
        + "\n\n[오늘의 랜덤 강아지]\n"
        + dog_text
        + "\n\n[출력 형식 - 반드시 아래 섹션 제목 그대로 출력]\n"
        "컨디션 등급: (S/A/B/C/D 중 하나)\n"
        "습관 분석: (2~5줄, 핵심만)\n"
        "날씨 코멘트: (1~2줄)\n"
        "내일 미션: (불릿 3개)\n"
        "오늘의 한마디: (한 문장)\n"
    )

    return openai_call(openai_key, sys_p, user_p)

# -----------------------------
# Habit Check-in UI
# -----------------------------
st.subheader("✅ 오늘의 습관 체크인")

left, right = st.columns([1.2, 1])

with left:
    st.markdown("**습관 체크** (5개, 2열)")
    c1, c2 = st.columns(2)
    checked = {}

    left_items = HABITS[:3]
    right_items = HABITS[3:]

    with c1:
        for name, emoji in left_items:
            checked[name] = st.checkbox(f"{emoji} {name}", value=False, key=f"habit_{name}")
    with c2:
        for name, emoji in right_items:
            checked[name] = st.checkbox(f"{emoji} {name}", value=False, key=f"habit_{name}")

    mood = st.slider("🙂 오늘 기분은 어때요?", min_value=1, max_value=10, value=6, step=1)

with right:
    st.markdown("**환경 설정**")
    city_label = st.selectbox("🏙️ 도시 선택", [c[0] for c in CITY_OPTIONS], index=0)
    city_query = dict(CITY_OPTIONS)[city_label]
    coach_style = st.radio("🎭 코치 스타일", list(COACH_STYLES.keys()), index=1)
    st.caption(f"설명: {COACH_STYLES[coach_style]}")

# -----------------------------
# Metrics
# -----------------------------
achieved_cnt = sum(1 for v in checked.values() if v)
rate_pct = round(achieved_cnt / 5 * 100, 1)

m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{rate_pct}%")
m2.metric("달성 습관", f"{achieved_cnt}/5")
m3.metric("기분", f"{mood}/10")

# -----------------------------
# Exact 7-day Chart (6 days + today)
# -----------------------------
st.subheader("📈 최근 7일 달성률")

today = datetime.now().date()
today_iso = today.isoformat()

# Dedupe history by date
hist_map = {}
for r in st.session_state["history"]:
    d = r.get("date")
    if d:
        hist_map[d] = r

dates_prev6 = [(today - timedelta(days=i)).isoformat() for i in range(6, 0, -1)]

rows = []
for d in dates_prev6:
    if d in hist_map:
        rows.append(hist_map[d])
    else:
        rows.append({"date": d, "achieved": 0, "rate": 0.0, "mood": 5})

rows.append({"date": today_iso, "achieved": achieved_cnt, "rate": float(rate_pct), "mood": mood})

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

st.bar_chart(df[["rate"]])

# -----------------------------
# AI Coach Report
# -----------------------------
st.subheader("🧠 AI 코치 리포트")
btn = st.button("컨디션 리포트 생성", type="primary")

if btn:
    # Save today into history (update if exists)
    new_row = {"date": today_iso, "achieved": achieved_cnt, "rate": float(rate_pct), "mood": mood}
    st.session_state["history"] = [r for r in st.session_state["history"] if r.get("date") != today_iso] + [new_row]

    # Keep last 30 days
    cutoff = today - timedelta(days=30)
    filtered = []
    for r in st.session_state["history"]:
        try:
            d = datetime.fromisoformat(r["date"]).date()
            if d >= cutoff:
                filtered.append(r)
        except Exception:
            pass
    st.session_state["history"] = sorted(filtered, key=lambda x: x["date"])

    # APIs
    weather, weather_err = get_weather(city_query, owm_api_key)
    dog = get_dog_image()

    with st.spinner("AI 코치가 리포트를 작성 중..."):
        report, report_err = generate_report(
            openai_key=openai_api_key,
            coach_style=coach_style,
            habits_checked=checked,
            mood=mood,
            weather=weather,
            dog=dog,
        )

    # Cards: Weather + Dog
    cL, cR = st.columns(2)

    with cL:
        st.markdown("### 🌦️ 오늘의 날씨")
        if weather:
            st.info(
                f"**{city_label}**  (`{weather.get('city')}`)\n\n"
                f"- 상태: {weather.get('description')}\n"
                f"- 기온: {weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C)\n"
                f"- 습도: {weather.get('humidity')}%\n"
                f"- 바람: {weather.get('wind_ms')} m/s"
            )
        else:
            st.warning("날씨 정보를 불러오지 못했어요.")
            if weather_err:
                st.caption(f"원인: {weather_err}")

    with cR:
        st.markdown("### 🐶 오늘의 강아지 카드")
        if dog:
            st.image(dog["image_url"], caption=f"품종: {dog.get('breed')}", use_container_width=True)
        else:
            st.warning("강아지 이미지를 불러오지 못했어요. (네트워크 확인)")

    # Report
    st.markdown("### 🧾 AI 코치 리포트")
    if report_err:
        st.error(report_err)
        report = None

    if report:
        st.success("리포트 생성 완료!")
        st.write(report)
    else:
        st.info("리포트를 생성하지 못했어요. 키/네트워크/요금제 상태를 확인해 주세요.")

    # Share Text
    share_payload = {
        "date": today_iso,
        "city": city_label,
        "city_query": city_query,
        "coach_style": coach_style,
        "rate_percent": rate_pct,
        "achieved": f"{achieved_cnt}/5",
        "mood": mood,
        "weather": weather,
        "weather_error": weather_err,
        "dog": dog,
        "report": report,
    }
    share_text = (
        "[AI 습관 트래커 공유]\n"
        f"- 날짜: {today_iso}\n"
        f"- 도시: {city_label} ({city_query})\n"
        f"- 코치: {coach_style}\n"
        f"- 달성률: {rate_pct}% ({achieved_cnt}/5)\n"
        f"- 기분: {mood}/10\n\n"
        "[리포트]\n"
        f"{report or '(리포트 없음)'}\n\n"
        "[원본 데이터(JSON)]\n"
        f"{json.dumps(share_payload, ensure_ascii=False, indent=2)}"
    )
    st.session_state["latest_report"] = report
    st.session_state["latest_share_text"] = share_text

# Show previous share text
if st.session_state.get("latest_share_text"):
    st.markdown("### 🔗 공유용 텍스트")
    st.code(st.session_state["latest_share_text"], language="text")

# -----------------------------
# Footer: API 안내
# -----------------------------
with st.expander("📌 API 안내 (준비물/주의사항)"):
    st.markdown(
        """
**1) OpenAI API Key**
- AI 코치 리포트 생성에 필요해요.
- 사이드바에 입력하면 현재 세션에서만 사용됩니다.

**2) OpenWeatherMap API Key**
- 날씨 카드에 필요해요.
- 호출 옵션: `units=metric`(섭씨), `lang=kr`(한국어)
- 이 앱은 도시를 `Seoul,KR`처럼 국가코드를 붙여서 요청합니다(404 방지).

**3) Dog CEO (무료, 키 불필요)**
- 랜덤 강아지 이미지를 가져옵니다.

**날씨가 안 나올 때**
- “원인: HTTP 401/404/429 …” 메시지를 보고 키/도시/레이트리밋을 확인해 주세요.
"""
    )
