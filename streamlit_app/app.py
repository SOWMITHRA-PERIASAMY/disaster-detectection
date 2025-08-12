# streamlit_app/app.py
import streamlit as st
from streamlit import session_state as ss
from PIL import Image
import base64, io, os, time
from streamlit_app.utils import run_unet_on_pil, load_sample_tweets, get_tweet_clf, get_unet
from dotenv import load_dotenv
import requests

load_dotenv()

st.set_page_config(page_title="DisasterHybrid Demo", layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("Demo Controls")
mode = st.sidebar.radio("Tweet Mode", ["Offline (default)", "Live (requires token)"])
live_token = os.getenv("TWITTER_BEARER_TOKEN", "")
if mode == "Live (requires token)" and not live_token:
    st.sidebar.warning("No TWITTER_BEARER_TOKEN found in .env — live mode will not stream.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload sample image** or use the provided sample.")
uploaded = st.sidebar.file_uploader("Satellite / Drone Image (jpg/png)", type=["jpg","jpeg","png"])
if st.sidebar.button("Use sample image"):
    uploaded = open("data/sample/sample_satellite.jpg","rb")

st.sidebar.markdown("---")
st.sidebar.write("Presentation Tips:")
st.sidebar.write("- Start with map + tweet panel\n- Upload sample image then toggle live tweets\n- 5-min script included in README")

# Layout
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("<h2 style='color:#00ff99'>Disaster Hybrid — CV + NLP</h2>", unsafe_allow_html=True)
    st.markdown("Upload an image to get segmentation mask (damage / flood overlay).")
    if uploaded:
        try:
            img = Image.open(uploaded)
            with st.spinner("Running segmentation..."):
                blended, mask = run_unet_on_pil(img)
            st.image(blended, caption="Segmentation overlay", use_column_width=True)
            st.markdown("**Download mask**")
            buf = io.BytesIO()
            Image.fromarray(mask).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="mask.png">Download mask.png</a>'
            st.markdown(href, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error running model: {e}")

    else:
        st.info("No image uploaded yet — upload one on the left sidebar or click 'Use sample image'.")

    st.markdown("---")
    st.markdown("### Map (demo)")
    st.info("Map view is a placeholder with demo severity markers.")
    import folium
    from streamlit_folium import st_folium
    m = folium.Map(location=[20,0], zoom_start=2, tiles="CartoDB dark_matter")
    folium.CircleMarker(location=[37.7749,-122.4194], radius=8, color="#00ff99", fill=True, popup="San Francisco - Moderate").add_to(m)
    folium.CircleMarker(location=[29.76,-95.36], radius=10, color="#ff6b6b", fill=True, popup="Houston - Severe").add_to(m)
    st_folium(m, width=700, height=350)

with col2:
    st.markdown("<h3 style='color:#00ff99'>Live / Simulated Tweets</h3>", unsafe_allow_html=True)
    tweet_clf = get_tweet_clf()
    tweets = load_sample_tweets()

    # Simulate streaming tweets
    if "tweet_idx" not in ss: ss.tweet_idx = 0
    if st.button("Reset tweet stream"):
        ss.tweet_idx = 0

    display_area = st.empty()
    controls = st.empty()
    mode_text = st.text("Running in offline simulation mode." if mode.startswith("Offline") else "Live mode (may require .env token).")

    # display some tweets
    for i in range(5):
        if ss.tweet_idx >= len(tweets): ss.tweet_idx = 0
        t = tweets[ss.tweet_idx]
        res = tweet_clf.predict(t["text"])
        color = "#00ff99" if res["severity"]=="Not disaster" else ("#ffcc00" if res["severity"]=="Moderate" else "#ff6b6b")
        with display_area.container():
            st.markdown(f"<div style='background:#121212;padding:10px;border-radius:8px;margin-bottom:8px;'>"
                        f"<b style='color:{color}'>[{res['severity']}]</b> <span style='color:#ddd'>{t['text']}</span><br>"
                        f"<small style='color:#888'>{t.get('time','')}</small></div>", unsafe_allow_html=True)
        ss.tweet_idx += 1
        time.sleep(0.2)

    st.markdown("---")
    st.markdown("### Single Tweet Classifier")
    txt = st.text_area("Paste tweet text here", value="Buildings collapsed after the earthquake, people trapped!")
    if st.button("Classify text"):
        r = tweet_clf.predict(txt)
        st.markdown(f"**Severity:** {r['severity']}  \n**Label:** {r['label']}  \n**Score:** {r['score']:.3f}")

