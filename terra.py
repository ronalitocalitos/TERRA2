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

/* ===== HISTORY CARD ===== */
.history-card {
    padding: 14px;
    margin-bottom: 12px;
    border-radius: 14px;
    background-color: #1f2937;
    border: 1px solid #374151;
}

.history-selected {
    background-color: #14532d !important;
    border: 1px solid #22c55e !important;
}

.history-date {
    font-size: 14px;
    font-weight: 600;
}

.history-time {
    font-size: 13px;
    opacity: 0.8;
}

/* ===== LOGOUT PUSH BOTTOM ===== */
.logout-container {
    margin-top: auto;
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

# ---------------- GET LAST 10 HISTORY ----------------
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
        st.subheader("📜 History (10 ล่าสุด)")

        if history_list:
            for item in history_list:

                date_part, time_part = format_thai_datetime(item['timestamp'])
                is_selected = (
                    item['timestamp'] == st.session_state.selected_timestamp
                )

                selected_class = "history-selected" if is_selected else ""

                if st.button(
                    f"{date_part} {time_part}",
                    key=item['timestamp'],
                    use_container_width=True
                ):
                    st.session_state.selected_timestamp = item['timestamp']

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
    st.title("TERRA Dashboard")

    if sensor_data:

        date_part, time_part = format_thai_datetime(sensor_data['timestamp'])
        st.caption(f"{date_part} | {time_part}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("N", sensor_data['N'])
        m2.metric("P", sensor_data['P'])
        m3.metric("K", sensor_data['K'])
        m4.metric("pH", sensor_data['pH'])
        m5.metric("Moisture", sensor_data['Moist'])

        st.divider()

        stage_name = st.selectbox(
            "ระยะการเจริญเติบโต:",
            ["ฟื้นต้น", "สะสมอาหาร", "ขยายผล", "ก่อนเก็บเกี่ยว"]
        )

        yield_target = st.number_input(
            "เป้าหมายผลผลิต (กก./ต้น)",
            min_value=1,
            value=100
        )

        if st.button("เริ่มวิเคราะห์", use_container_width=True):

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

            st.success(f"💡 {action_result}")

            c1, c2, c3 = st.columns(3)
            c1.info(f"N: {n_pred:.1f} กรัม")
            c2.info(f"P: {p_pred:.1f} กรัม")
            c3.info(f"K: {k_pred:.1f} กรัม")

    else:
        st.error("❌ ไม่พบข้อมูล")

    st.divider()
    st.caption("Project Terra | Engineering CMU 2026")