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

/* ===== SIDEBAR FLEX LAYOUT ===== */
section[data-testid="stSidebar"] > div:first-child {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

/* ===== TERRA LOGO TEXT ===== */
.sidebar-title {
    font-size: 28px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 10px;
    letter-spacing: 2px;
}

/* ===== PUSH LOGOUT TO BOTTOM ===== */
.logout-container {
    margin-top: auto;
}

/* ===== HISTORY BUTTON STYLE ===== */
section[data-testid="stSidebar"] button {
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: linear-gradient(145deg, #1e1e1e, #161616) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    padding: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

section[data-testid="stSidebar"] button:hover {
    transform: translateY(-2px);
    background: linear-gradient(145deg, #252525, #1c1c1c) !important;
    border: 1px solid rgba(0,255,150,0.4) !important;
}

/* ===== ACTIVE HISTORY ===== */
.active-history button {
    background: linear-gradient(145deg, #00c853, #009624) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
}

/* ===== LOGOUT BUTTON RED ONLY ===== */
div.logout-container button {
    border: 2px solid #e53935 !important;
    background-color: rgba(229, 57, 53, 0.08) !important;
    color: #e53935 !important;
    font-weight: 700 !important;
}

div.logout-container button:hover {
    background-color: rgba(229, 57, 53, 0.18) !important;
}

/* ===== WHITE TIME TEXT ===== */
.time-text {
    color: white;
    text-align: right;
    font-weight: 600;
    margin-top: 10px;
}

/* ===== METRIC BIGGER ===== */
div[data-testid="metric-container"] label {
    font-size: 22px !important;
    font-weight: 700 !important;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 30px !important;
    font-weight: 700 !important;
}

/* ===== BIG ANALYZE BUTTON ONLY ===== */
div.stButton > button[kind="secondary"] {
    font-size: 22px !important;
    font-weight: 800 !important;
    padding: 16px 20px !important;
    border-radius: 12px !important;
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
    thai_months_full = {
        1: "มกราคม", 2: "กุมภาพันธ์", 3: "มีนาคม", 4: "เมษายน",
        5: "พฤษภาคม", 6: "มิถุนายน", 7: "กรกฎาคม", 8: "สิงหาคม",
        9: "กันยายน", 10: "ตุลาคม", 11: "พฤศจิกายน", 12: "ธันวาคม"
    }

    timestamp_str = timestamp_str.strip()

    formats = ["%d%m%Y_%H%M%S", "%Y%m%d_%H%M%S"]

    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            date_part = f"{dt.day} {thai_months_full[dt.month]} {dt.year}"
            time_part = f"{dt.hour}:{dt.minute:02d}"
            return date_part, time_part
        except:
            continue

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
    except Exception as e:
        st.error(f"History error: {e}")
        return []

# ---------------- SESSION ----------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_device' not in st.session_state:
    st.session_state.current_device = None
if 'selected_timestamp' not in st.session_state:
    st.session_state.selected_timestamp = None

# ================= DASHBOARD =================
if st.session_state.logged_in:

    device_id = st.session_state.current_device
    history_list = get_sensor_history(device_id)

    sensor_data = history_list[0] if history_list else None

    if sensor_data:

        stage_name = st.selectbox(
            "ระยะการเจริญเติบโต:",
            ["ฟื้นต้น", "สะสมอาหาร", "ขยายผล", "ก่อนเก็บเกี่ยว"]
        )

        yield_target = st.number_input(
            "เป้าหมายผลผลิต (กก./ต้น)",
            min_value=1,
            value=100
        )

        # ✅ ปุ่มใหญ่ขึ้น (แก้แค่ตรงนี้)
        if st.button("เริ่มวิเคราะห์", use_container_width=True, type="secondary"):

            stage_map = {
                "ฟื้นต้น":1,
                "สะสมอาหาร":2,
                "ขยายผล":3,
                "ก่อนเก็บเกี่ยว":4
            }

            input_df = pd.DataFrame([[ 
                sensor_data['N'],
                sensor_data['P'],
                sensor_data['K'],
                sensor_data['pH'],
                sensor_data['Moist'],
                stage_map[stage_name],
                yield_target
            ]], columns=[
                'N_soil','P_soil','K_soil',
                'pH','Moisture','Stage','Target_Yield_kg'
            ])

            action_result = clf.predict(input_df)[0]
            nums_result = reg.predict(input_df)[0]

            n_pred = max(0, nums_result[1])
            p_pred = max(0, nums_result[2])
            k_pred = max(0, nums_result[3])

            st.success(f"💡 ผลวิเคราะห์จาก AI: {action_result}")

            colA, colB, colC = st.columns(3)
            colA.info(f"N: {n_pred:.1f} กรัม")
            colB.info(f"P: {p_pred:.1f} กรัม")
            colC.info(f"K: {k_pred:.1f} กรัม")

    else:
        st.error("❌ ไม่พบข้อมูลเซนเซอร์")