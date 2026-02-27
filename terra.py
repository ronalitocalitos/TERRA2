import streamlit as st
import pandas as pd
import joblib
import firebase_admin
from firebase_admin import credentials, firestore

# --- 1. SETUP (ต้องอยู่บนสุดและห้ามย้าย) ---
st.set_page_config(
    page_title="TERRA - AI System",
    page_icon="🌱",
    layout="wide"
)

# --- 2. INITIALIZE SESSION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_device' not in st.session_state:
    st.session_state.current_device = None

# --- 3. CACHED RESOURCES ---
@st.cache_resource
def init_firebase():
    if not firebase_admin._apps:
        try:
            key_dict = st.secrets["firebase_key"]
            cred = credentials.Certificate(dict(key_dict))
            firebase_admin.initialize_app(cred)
        except Exception:
            return None
    return firestore.client()

@st.cache_resource
def load_terra_model():
    try:
        # ตรวจสอบชื่อไฟล์ให้ตรงกับที่อัปโหลด (terra_model.pkl)
        return joblib.load("terra_model.pkl")
    except Exception:
        return None

# --- 4. DATA FUNCTIONS ---
def get_sensor_latest(db, device_id):
    if db is None or device_id is None:
        return None
    try:
        query = db.collection('devices').document(device_id).collection('soilData')
        docs = query.order_by("__name__", direction=firestore.Query.DESCENDING).limit(1).get()
        for doc in docs:
            data = doc.to_dict()
            return {
                'N': data.get('N', 0), 'P': data.get('P', 0), 'K': data.get('K', 0),
                'pH': data.get('pH', 0), 'Moist': data.get('moisture', 0), 
                'temp': data.get('temperature', 0), 'cond': data.get('conductivity', 0)
            }
    except Exception:
        return None
    return None

# --- 5. MAIN APPLICATION ---
def main():
    db = init_firebase()
    model_data = load_terra_model()

    if not st.session_state.logged_in:
        # === VIEW: LOGIN PAGE ===
        # ใช้ st.columns เพื่อจัดตำแหน่งแคปชั่นแทน HTML ดิบ
        top_col1, top_col2 = st.columns([8, 2])
        top_col2.caption("🏠 TERRA - home")

        st.markdown("<br><br>", unsafe_html=True)
        st.markdown("<h1 style='text-align: center; color: #2E7D32;'>เข้าสู่ระบบ TERRA (WEFARM)</h1>", unsafe_html=True)
        st.markdown("<p style='text-align: center;'>กรุณากรอกรหัสเครื่องเซนเซอร์เพื่อดูข้อมูลแปลงเกษตร</p>", unsafe_html=True)

        _, login_col, _ = st.columns([1, 1.5, 1])
        with login_col:
            with st.form("login_form"):
                device_input = st.text_input("Serial Number:", placeholder="เช่น TERRA0001")
                submit_login = st.form_submit_button("เข้าสู่ระบบ", use_container_width=True)
                
                if submit_login:
                    if device_input:
                        device_id_upper = device_input.strip().upper()
                        if db:
                            doc_ref = db.collection('devices').document(device_id_upper).get()
                            if doc_ref.exists:
                                st.session_state.logged_in = True
                                st.session_state.current_device = device_id_upper
                                st.rerun()
                            else:
                                st.error("❌ ไม่พบรหัสเครื่องนี้ในระบบ")
                        else:
                            st.error("Firebase Connection Error")
                    else:
                        st.warning("⚠️ กรุณากรอกรหัสเครื่อง")

    else:
        # === VIEW: DASHBOARD PAGE ===
        top_col1, top_col2 = st.columns([8, 2])
        top_col2.caption("📊 TERRA - dashboard")

        device_id = st.session_state.current_device
        
        with st.sidebar:
            st.success(f"🟢 เครื่อง: **{device_id}**")
            st.divider()
            if st.button("🚪 ออกจากระบบ", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.current_device = None
                st.rerun()

        st.title("🌱 TERRA Dashboard")
        sensor_data = get_sensor_latest(db, device_id)

        if sensor_data:
            st.subheader("📊 ข้อมูลล่าสุดจากเซนเซอร์")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Nitrogen (N)", f"{sensor_data['N']}")
            m2.metric("Phosphorus (P)", f"{sensor_data['P']}")
            m3.metric("Potassium (K)", f"{sensor_data['K']}")
            m4.metric("ค่า pH (ดิน)", f"{sensor_data['pH']}")
            m5.metric("ความชื้น", f"{sensor_data['Moist']}%")
            
            st.divider()
            st.subheader("⚙️ วิเคราะห์การใส่ปุ๋ยด้วย AI")
            
            c1, c2 = st.columns(2)
            with c1:
                stage_name = st.selectbox("ระยะการเจริญเติบโต:", ["ฟื้นต้น", "สะสมอาหาร", "ขยายผล", "ก่อนเก็บเกี่ยว"])
            with c2:
                yield_target = st.number_input("เป้าหมายผลผลิต (กก./ต้น):", min_value=1, value=100)

            if st.button("🚀 เริ่มวิเคราะห์แผนการใส่ปุ๋ย", use_container_width=True):
                if model_data:
                    clf = model_data.get('classifier')
                    reg = model_data.get('regressor')
                    
                    if clf and reg:
                        stage_map = {"ฟื้นต้น": 1, "สะสมอาหาร": 2, "ขยายผล": 3, "ก่อนเก็บเกี่ยว": 4}
                        
                        if stage_name == "ฟื้นต้น" and (sensor_data['pH'] < 5.5 or sensor_data['pH'] > 7.0):
                            st.error("⚠️ ค่า pH ไม่เหมาะสมสำหรับการให้ปุ๋ยเคมี กรุณาปรับสภาพดินก่อน")
                        else:
                            input_df = pd.DataFrame([[
                                sensor_data['N'], sensor_data['P'], sensor_data['K'],
                                sensor_data['pH'], sensor_data['Moist'], 
                                stage_map[stage_name], yield_target
                            ]], columns=['N_soil', 'P_soil', 'K_soil', 'pH', 'Moisture', 'Stage', 'Target_Yield_kg'])

                            action = clf.predict(input_df)[0]
                            nums = reg.predict(input_df)[0]

                            st.success(f"### 💡 คำแนะนำจาก AI: \n {action}")
                            r1, r2, r3 = st.columns(3)
                            r1.info(f"**N**: {max(0, nums[1]):.1f} g")
                            r2.info(f"**P**: {max(0, nums[2]):.1f} g")
                            r3.info(f"**K**: {max(0, nums[3]):.1f} g")
                    else:
                        st.error("Model format error")
                else:
                    st.error("ไม่สามารถโหลดโมเดล AI ได้")
        else:
            st.error("❌ ไม่พบข้อมูลเซนเซอร์ล่าสุด")

        st.divider()
        st.caption("Project Terra | Engineering CMU 2026")

if __name__ == "__main__":
    main()