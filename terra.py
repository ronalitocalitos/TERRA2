import streamlit as st
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="TERRA - AI Fertilizer System",
    page_icon="🌱",
    layout="wide"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>

/* ===== SIDEBAR FLEX ===== */
section[data-testid="stSidebar"] > div:first-child {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

/* ===== HISTORY CARD STYLE ===== */
.history-card {
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 10px;
    background-color: #1e1e1e;
    border: 1px solid #2e2e2e;
    transition: all 0.2s ease-in-out;
}

.history-card:hover {
    background-color: #2a2a2a;
    border: 1px solid #4ade80;
    cursor: pointer;
}

.history-selected {
    background-color: #14532d !important;
    border: 1px solid #22c55e !important;
}

/* ===== LOGOUT BUTTON ===== */
.logout-container {
    margin-top: auto;
}

.logout-container button {
    border: 2px solid #e53935 !important;
    background-color: rgba(229, 57, 53, 0.08) !important;
    color: #e53935 !important;
    font-weight: 600 !important;
}

/* ===== TIME TEXT ===== */
.time-text {
    color: white;
    text-align: right;
    font-weight: 600;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- FIREBASE INIT ----------------
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        key_dict = st.secrets["firebase_key"]
        cred = credentials.Certificate(dict(key_dict))
        firebase_admin.initialize_app(cred)
    return firestore.client()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("terra_model.pkl")

db = init_firebase()
model_data = load_model()
clf = model_data['classifier']
reg = model_data['regressor']

# ---------------- FORMAT TIME ----------------
def format_thai_datetime(timestamp_str):
    try:
        dt = datetime.strptime(timestamp_str, "%d%m%Y_%H%M%S")

        thai_months = {
            1: "มกราคม", 2: "กุมภาพันธ์", 3: "มีนาคม", 4: "เมษายน",
            5: "พฤษภาคม", 6: "มิถุนายน", 7: "กรกฎาคม", 8: "สิงหาคม",
            9: "กันยายน", 10: "ตุลาคม", 11: "พฤศจิกายน", 12: "ธันวาคม"
        }

        return (
            f"{dt.day} {thai_months[dt.month]} {dt.year}",
            f"{dt.hour}:{dt.minute:02d}"
        )
    except:
        return timestamp_str, ""

# ---------------- GET HISTORY ----------------
def get_sensor_history(device_id, limit=10):
    try:
        query = db.collection('devices') \
                  .document(device_id) \
                  .collection('soilData') \
                  .order_by("__name__", direction=firestore.Query.DESCENDING) \
                  .limit(limit)

        docs = query.stream()

        history = []
        for doc in docs:
            data = doc.to_dict()
            history.append({
                'timestamp': doc.id,
                'N': data.get('N', 0),
                'P': data.get('P', 0),
                'K': data.get('K', 0),
                'pH': data.get('pH', 0),
                'Moist': data.get('moisture', 0),
                'temp': data.get('temperature', 0),
                'cond': data.get('conductivity', 0)
            })
        return history
    except:
        return []

# ---------------- SESSION ----------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_device' not in st.session_state:
    st.session_state.current_device = None
if 'selected_timestamp' not in st.session_state:
    st.session_state.selected_timestamp = None

# ==================================================
# LOGIN PAGE
# ==================================================
if not st.session_state.logged_in:

    st.title("เข้าสู่ระบบ TERRA")
    device_input = st.text_input("Serial Number", placeholder="TERRA0001")

    if st.button("เข้าสู่ระบบ", use_container_width=True):
        if device_input:
            device_id_upper = device_input.strip().upper()
            doc_ref = db.collection('devices').document(device_id_upper).get()

            if doc_ref.exists:
                st.session_state.logged_in = True
                st.session_state.current_device = device_id_upper
                st.rerun()
            else:
                st.error("❌ ไม่พบรหัสเครื่องนี้ในระบบ")
        else:
            st.warning("⚠️ กรุณากรอกรหัสเครื่อง")

# ==================================================
# DASHBOARD
# ==================================================
else:

    device_id = st.session_state.current_device
    history_list = get_sensor_history(device_id)

    # -------- SIDEBAR --------
    with st.sidebar:

        st.success(f"🟢 เชื่อมต่อกับเครื่อง:\n**{device_id}**")
        st.divider()
        st.subheader("📜 History")

        if history_list:
            for item in history_list:

                date_part, time_part = format_thai_datetime(item['timestamp'])
                is_selected = item['timestamp'] == st.session_state.selected_timestamp
                selected_class = "history-selected" if is_selected else ""

                # clickable invisible button
                if st.button("", key=item['timestamp']):
                    st.session_state.selected_timestamp = item['timestamp']

                # visual card
                st.markdown(
                    f"""
                    <div class="history-card {selected_class}">
                        <div style="font-size:13px;">{date_part}</div>
                        <div style="font-size:12px; opacity:0.7;">{time_part}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.divider()
        st.markdown("<div class='logout-container'>", unsafe_allow_html=True)

        if st.button("ออกจากระบบ", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_device = None
            st.session_state.selected_timestamp = None
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # -------- SELECT DATA --------
    sensor_data = None

    if history_list:
        if st.session_state.selected_timestamp:
            for item in history_list:
                if item['timestamp'] == st.session_state.selected_timestamp:
                    sensor_data = item
                    break
        else:
            sensor_data = history_list[0]

    # -------- HEADER --------
    col_left, col_right = st.columns([3,1])

    with col_left:
        st.title("TERRA Dashboard")

    with col_right:
        if sensor_data:
            date_part, time_part = format_thai_datetime(sensor_data['timestamp'])
            st.markdown(
                f"""
                <div class='time-text'>
                    <div style="font-size:20px;">{date_part}</div>
                    <div style="font-size:20px;">{time_part}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    if sensor_data:

        st.subheader("ข้อมูลจากเซนเซอร์")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("N", sensor_data['N'])
        m2.metric("P", sensor_data['P'])
        m3.metric("K", sensor_data['K'])
        m4.metric("pH", sensor_data['pH'])
        m5.metric("Moisture", sensor_data['Moist'])

    else:
        st.error("❌ ไม่พบข้อมูล")

    st.divider()
    st.caption("Project Terra | Engineering CMU 2026")