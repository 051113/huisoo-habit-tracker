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
    st.markdown("**Tip**:
