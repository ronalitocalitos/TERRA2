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

# ---------------- FORCE SIDEBAR LOGOUT STYLE ----------------
st.markdown("""
<style>

/* ===== FORCE ALL SIDEBAR BUTTON STYLE ===== */
section[data-testid="stSidebar"] button {
    border: 2px solid #e53935 !important;
    background-color: rgba(229, 57, 53, 0.08) !important;
    color: #e53935 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] button:hover {
    background-color: rgba(229, 57, 53, 0.18) !important;
}

/* ===== WHITE TIME TEXT ===== */
.time-text {
    color: white;
    text-align: right;
    line-height: 1.4;
    margin-top: 10px;
    font-size: 22px;
    font-weight: 600;
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
            1: "ม.ค.", 2: "ก.พ.", 3: "มี.ค.", 4: "เม.ย.",
            5: "พ.ค.", 6: "มิ.ย.", 7: "ก.ค.", 8: "ส.ค.",
            9: "ก.ย.", 10: "ต.ค.", 11: "พ.ย.", 12: "ธ.ค."
        }
        return f"{dt.day} {thai_months[dt.month]} {dt.year}<br>{dt.hour}:{dt.minute:02d}"
    except:
        return timestamp_str

# ---------------- GET SENSOR DATA ----------------
def get_sensor_latest(device_id):
    try:
        query = db.collection('devices').document(device_id).collection('soilData')
        docs = query.order_by("__name__", direction=firestore.Query.DESCENDING).limit(1).get()

        for doc in docs:
            data = doc.to_dict()
            return {
                'timestamp': doc.id,
                'N': data.get('N', 0),
                'P': data.get('P', 0),
                'K': data.get('K', 0),
                'pH': data.get('pH', 0),
                'Moist': data.get('moisture', 0),
                'temp': data.get('temperature', 0),
                'cond': data.get('conductivity', 0)
            }
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
    return None

# ---------------- SESSION ----------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_device' not in st.session_state:
    st.session_state.current_device = None

# ==================================================
# LOGIN PAGE
# ==================================================
if not st.session_state.logged_in:

    st.markdown("""
        <h1 style='text-align: center; margin-top: 100px;'>
            เข้าสู่ระบบ TERRA
        </h1>
        <p style='text-align: center; font-size:18px;'>
            กรุณากรอกรหัสเครื่องเซนเซอร์ (Serial Number)
        </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        with st.form("login_form"):
            device_input = st.text_input("Serial Number:", placeholder="เช่น TERRA0001")
            submit_login = st.form_submit_button("เข้าสู่ระบบ", use_container_width=True)

            if submit_login:
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

    # -------- SIDEBAR --------
    with st.sidebar:

        st.success(f"🟢 เชื่อมต่อกับเครื่อง:\n**{device_id}**")

        # Spacer ดันปุ่มลงล่าง
        st.markdown("<br>" * 20, unsafe_allow_html=True)

        logout = st.button("🚪 ออกจากระบบ", use_container_width=True)

        if logout:
            st.session_state.logged_in = False
            st.session_state.current_device = None
            st.rerun()

    # -------- FETCH DATA --------
    sensor_data = get_sensor_latest(device_id)

    # -------- HEADER --------
    col_left, col_right = st.columns([3,1])

    with col_left:
        st.title("TERRA Dashboard")

    with col_right:
        if sensor_data:
            formatted_time = format_thai_datetime(sensor_data['timestamp'])
            st.markdown(
                f"<div class='time-text'>{formatted_time}</div>",
                unsafe_allow_html=True
            )

    st.markdown("วิเคราะห์ธาตุอาหารในดินและแนะนำการใส่ปุ๋ยด้วย AI")

    if sensor_data:
        st.subheader("ข้อมูลล่าสุดจากเซนเซอร์")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Nitrogen (N)", sensor_data['N'])
        m2.metric("Phosphorus (P)", sensor_data['P'])
        m3.metric("Potassium (K)", sensor_data['K'])
        m4.metric("ค่า pH", sensor_data['pH'])
        m5.metric("ความชื้น (%)", sensor_data['Moist'])

        with st.expander("ดูค่าเพิ่มเติม"):
            st.write(f"อุณหภูมิ: {sensor_data['temp']} °C")
            st.write(f"Conductivity: {sensor_data['cond']}")

        st.divider()

        st.subheader("⚙️ ตั้งค่าการวิเคราะห์")

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

            st.success(f"💡 ผลวิเคราะห์จาก AI: {action_result}")

            colA, colB, colC = st.columns(3)
            colA.info(f"N: {n_pred:.1f} กรัม")
            colB.info(f"P: {p_pred:.1f} กรัม")
            colC.info(f"K: {k_pred:.1f} กรัม")

    else:
        st.error("❌ ไม่พบข้อมูลเซนเซอร์")

    st.divider()
    st.caption("Project Terra | Engineering CMU 2026")