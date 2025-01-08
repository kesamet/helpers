"""
Streamlit app.
Timestamps used are in SGT while app server is in (naive) UTC
"""
from datetime import datetime, timedelta, timezone
from os import getenv
from pathlib import Path

import pytz
import requests
import streamlit as st
from jinja2 import Environment, FileSystemLoader, select_autoescape
from streamlit.file_util import get_static_dir

from utils.serve.kinesis import devices
from utils.serve.general import set_sgtz, set_utcz

JINJA_ENV = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(["html"]),
)
VIDEO_PLAYER = JINJA_ENV.get_template("native.html")
DOMAIN = getenv("DOMAIN") or "video.pub.model.amoy.ai"
DEBUG = False


def camera_page(timenow):
    """Camera feed."""
    today = timenow.date()

    col0, col1, col2 = st.beta_columns((2, 1, 1))
    last_30 = today - timedelta(days=29)
    dt = col0.date_input("Date", today, min_value=last_30, max_value=today)
    hr = col1.selectbox("Hr", list(range(24)), 8)
    mins = int(col2.selectbox("Mins", [str(i).zfill(2) for i in range(60)], 0))
    select_time = set_sgtz(datetime(dt.year, dt.month, dt.day, hr, mins))
    utc_time = select_time.astimezone(timezone.utc)

    readable_text = [
        "Camera 1",
        "Camera 2",
    ]
    select_camera = st.selectbox(
        "Select camera.", list(range(len(readable_text))), format_func=lambda i: readable_text[i])

    real_time = set_utcz(datetime.utcnow() - timedelta(minutes=2))
    if utc_time > real_time:
        now = real_time.replace(second=0, microsecond=0)
        st.info(f"Showing live footage from {now.time()} UTC")
    else:
        download = st.button("Download footage")
        if download:
            clip_end = utc_time + timedelta(minutes=2)
            video = Path(get_static_dir()) / "video.mp4"
            friendly_name = f"b{select_camera}_{select_time.isoformat()}.mp4"
            payload = devices[select_camera].get_clip(utc_time, clip_end)
            with open(video, "wb") as f:
                chunk = payload.read()
                while chunk:
                    f.write(chunk)
                    chunk = payload.read()
            button = f'Your video is ready: <a download="{friendly_name}" href="{video.name}">{friendly_name}</a>'
            st.markdown(button, unsafe_allow_html=True)

    params = {"camera": select_camera}
    start = utc_time.isoformat() if utc_time < real_time else ""
    if start:
        params["start"] = start
    api = getenv("API") or f"https://{DOMAIN}/api"
    if st.button("Show video stream"):
        # Streamlit will double load the video if it's placed outside the button
        resp = requests.get(api, params=params)
        if resp.status_code != 200:
            st.error(f"Failed to load video stream: {resp.text}")
        else:
            player = VIDEO_PLAYER.render(domain=DOMAIN, src=resp.json()[0])
            st.components.v1.html(html=player, height=524)


def main():
    if DEBUG:
        st.sidebar.warning("DEBUG")
        timenow = set_sgtz(datetime.now())
    else:
        # App server in UTC
        timenow = set_utcz(datetime.now()).astimezone(pytz.timezone("Asia/Singapore"))

    dict_pages = {
        "Camera Feed": camera_page,
    }

    select_page = st.sidebar.radio("", list(dict_pages.keys()))
    st.title(select_page)
    st.sidebar.info("Note: results are available hourly one hour after the hour.")
    st.sidebar.write(timenow.strftime("%Y-%m-%d %H:%M SGT"))

    dict_pages[select_page](timenow)


if __name__ == "__main__":
    main()
