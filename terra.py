import streamlit as st
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# --- 1. การตั้งค่าหน้าเว็บ ---
st.set_page_config(
    page_title="TERRA - AI Fertilizer System",
    page_icon="🌱",
    layout="wide"
)

# --- 2. เชื่อมต่อ Firebase ---
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        key_dict = st.secrets["firebase_key"]
        cred = credentials.Certificate(dict(key_dict))
        firebase_admin.initialize_app(cred)
    return firestore.client()

# --- 3. โหลดโมเดล AI ---
@st.cache_resource
def load_terra_model():
    return joblib.load("terra_model.pkl")

db = init_firebase()
model_data = load_terra_model()
clf = model_data['classifier']
reg = model_data['regressor']

# --- 4. ดึงข้อมูลเซนเซอร์ล่าสุด ---
def get_sensor_latest(device_id):
    try:
        query = db.collection('devices').document(device_id).collection('soilData')
        docs = query.order_by("__name__", direction=firestore.Query.DESCENDING).limit(1).get()
        for doc in docs:
            data = doc.to_dict()
            return {
                'N': data.get('N', 0),
                'P': data.get('P', 0),
                'K': data.get('K', 0),
                'pH': data.get('pH', 0),
                'Moist': data.get('moisture', 0),
                'temp': data.get('temperature', 0),
                'cond': data.get('conductivity', 0)
            }
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}")
    return None

# --- 5. Session ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_device' not in st.session_state:
    st.session_state.current_device = None

# ==================================================
# 🚪 LOGIN PAGE (Centered)
# ==================================================
if not st.session_state.logged_in:

    st.markdown("""
        <h1 style='text-align: center; margin-top: 80px;'>
            เข้าสู่ระบบ TERRA (WEFARM)
        </h1>
        <p style='text-align: center; font-size:18px;'>
            กรุณากรอกรหัสเครื่องเซนเซอร์ (Serial Number) เพื่อเข้าดูข้อมูลแปลงเกษตรของคุณ
        </p>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

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
                        st.error("❌ ไม่พบรหัสเครื่องนี้ในระบบ กรุณาตรวจสอบความถูกต้อง")
                else:
                    st.warning("⚠️ กรุณากรอกรหัสเครื่อง")

# ==================================================
# 🌾 DASHBOARD PAGE
# ==================================================
else:
    device_id = st.session_state.current_device

    # Sidebar
    with st.sidebar:
        st.success(f"🟢 เชื่อมต่อกับเครื่อง:\n**{device_id}**")
        st.divider()
        if st.button("🚪 ออกจากระบบ", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_device = None
            st.rerun()

    st.title("TERRA")
    st.markdown("วิเคราะห์ธาตุอาหารในดินและแนะนำการใส่ปุ๋ยด้วย AI โดยกลุ่ม WEFARM")

    sensor_data = get_sensor_latest(device_id)

    if sensor_data:
        st.subheader("ข้อมูลล่าสุดจากเซนเซอร์")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Nitrogen (N)", f"{sensor_data['N']}")
        m2.metric("Phosphorus (P)", f"{sensor_data['P']}")
        m3.metric("Potassium (K)", f"{sensor_data['K']}")
        m4.metric("ค่า pH", f"{sensor_data['pH']}")
        m5.metric("ความชื้น", f"{sensor_data['Moist']}%")

        with st.expander("ดูค่าเพิ่มเติม"):
            st.write(f"อุณหภูมิ: {sensor_data['temp']} °C")
            st.write(f"Conductivity: {sensor_data['cond']}")

        st.divider()

        stage_name = st.selectbox(
            "ระยะการเจริญเติบโต:",
            ["ฟื้นต้น", "สะสมอาหาร", "ขยายผล", "ก่อนเก็บเกี่ยว"]
        )

        yield_target = st.number_input("เป้าหมายผลผลิต (กก./ต้น)", min_value=1, value=100)

        if st.button("เริ่มวิเคราะห์", use_container_width=True):

            stage_map = {"ฟื้นต้น":1, "สะสมอาหาร":2, "ขยายผล":3, "ก่อนเก็บเกี่ยว":4}

            input_df = pd.DataFrame([[
                sensor_data['N'],
                sensor_data['P'],
                sensor_data['K'],
                sensor_data['pH'],
                sensor_data['Moist'],
                stage_map[stage_name],
                yield_target
            ]], columns=[
                'N_soil','P_soil','K_soil','pH','Moisture','Stage','Target_Yield_kg'
            ])

            action_result = clf.predict(input_df)[0]
            nums_result = reg.predict(input_df)[0]

            n_pred = max(0, nums_result[1])
            p_pred = max(0, nums_result[2])
            k_pred = max(0, nums_result[3])

            st.success(f"💡 ผลวิเคราะห์: {action_result}")

            colA, colB, colC = st.columns(3)
            colA.info(f"N: {n_pred:.1f} กรัม")
            colB.info(f"P: {p_pred:.1f} กรัม")
            colC.info(f"K: {k_pred:.1f} กรัม")

    else:
        st.error("❌ ไม่พบข้อมูลเซนเซอร์")

    st.divider()
    st.caption("Project Terra | Engineering CMU 2026")