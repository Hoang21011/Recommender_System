import sys
import os
import streamlit as st
import tempfile
import re
import ast
import cv2  

# Import YOLO
from ultralytics import YOLO

# --- CẤU HÌNH ĐƯỜNG DẪN ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

# --- CẤU HÌNH ĐƯỜNG DẪN MODEL YOLO ---
# Đường dẫn máy của bạn
YOLO_MODEL_PATH = r"D:\project\cv_final\model\best.pt"

# Import module của bạn
try:
    from retrieval.search import FoodRetrievalSystem
except ImportError:
    class FoodRetrievalSystem:
        def search_by_text(self, q, k):
            return [{"title": f"Recipe with {q} #{i}", "image_path": "https://via.placeholder.com/150", "ingredients": f"{q}, salt, oil", "instructions": "Cook it"} for i in range(k)]

        def search_by_image(self, p, k):
            return [{"title": f"Similar Image Food {i}", "image_path": "https://via.placeholder.com/150", "ingredients": "A, B", "instructions": "Cook it"} for i in range(k)]

st.set_page_config(page_title="Food Search", layout="wide")

# ---------------------------------------------
# 1. LOAD MODEL & ENGINE (CACHING)
# ---------------------------------------------
@st.cache_resource
def load_search_engine():
    """Load Search Engine một lần duy nhất"""
    return FoodRetrievalSystem()

@st.cache_resource
def load_yolo_model():
    """Load YOLO Model một lần duy nhất"""
    if not os.path.exists(YOLO_MODEL_PATH):
        st.error(f"⚠️ Model file not found at: {YOLO_MODEL_PATH}")
        return None
    return YOLO(YOLO_MODEL_PATH)

# ---------------------------------------------
# 2. UI COMPONENTS
# ---------------------------------------------
@st.dialog("Recipe Detail")
def show_recipe_dialog(item):
    try:
        st.image(item["image_path"], width=450)
    except:
        st.warning("Image not found")
        
    st.subheader(item["title"])
    
    # 2. Xử lý và hiển thị Ingredients
    st.markdown("### 🥦 Ingredients")
    
    raw_ingredients = item["ingredients"]
    
    # Kiểm tra xem dữ liệu có phải là chuỗi dạng list không "['a', 'b']"
    if isinstance(raw_ingredients, str) and raw_ingredients.startswith("["):
        try:
            # Chuyển chuỗi thành list an toàn
            ing_list = ast.literal_eval(raw_ingredients)
            
            # Cách 1: Hiển thị từng dòng bằng Markdown bullet point
            for ing in ing_list:
                st.markdown(f"- {ing}")
                
        except (ValueError, SyntaxError):
            # Nếu lỗi parse, hiển thị nguyên gốc
            st.info(raw_ingredients)
    else:
        # Nếu data đã là list hoặc chuỗi thường
        if isinstance(raw_ingredients, list):
            for ing in raw_ingredients:
                st.markdown(f"- {ing}")
        else:
            st.info(raw_ingredients)

    # --- XỬ LÝ INSTRUCTIONS (GOM NHÓM) ---
    st.markdown("### 🍳 Instructions")
    
    raw_instructions = item["instructions"]
    
    if isinstance(raw_instructions, str):
        # 1. Tách chuỗi thành list các câu đơn lẻ
        sentences = [s.strip() for s in raw_instructions.split('. ') if s.strip()]
        
        # 2. Gom nhóm: Cứ 3 câu thành 1 Step (Chunking)
        # step_size = 3 nghĩa là 10 câu sẽ chia thành: 3, 3, 3, 1 (Tổng 4 steps)
        step_size = 3 
        grouped_steps = []
        
        for i in range(0, len(sentences), step_size):
            # Lấy 3 câu liên tiếp
            group = sentences[i : i + step_size]
            
            # Nối lại thành 1 đoạn văn
            combined_text = ". ".join(group)
            
            # Đảm bảo kết thúc bằng dấu chấm
            if not combined_text.endswith('.'):
                combined_text += "."
            
            grouped_steps.append(combined_text)
            
        # 3. Xử lý Regex và hiển thị từng nhóm
        for i, step_text in enumerate(grouped_steps):
            
            # --- ÁP DỤNG REGEX IN ĐẬM (Như đã sửa ở trên) ---
            # In đậm Thời gian (vd: 10-12 minutes)
            step_text = re.sub(
                r'(\d+(?:[-–]\d+)?\s+(?:hours?|hr|minutes?|mins?|seconds?|secs?))', 
                r'**\1**', step_text, flags=re.IGNORECASE
            )
            # In đậm Nhiệt độ (vd: 190°C)
            step_text = re.sub(r'(\d+\s?°[CF])', r'**\1**', step_text)
            # In đậm Gas mark
            step_text = re.sub(r'(gas mark\s+\d+)', r'**\1**', step_text, flags=re.IGNORECASE)

            # Hiển thị
            st.markdown(f"**Step {i+1}:** {step_text}")

    elif isinstance(raw_instructions, list):
        # Nếu dữ liệu gốc đã là list, ta hiển thị luôn (hoặc cũng có thể gom nếu muốn)
        for i, step in enumerate(raw_instructions):
             st.markdown(f"**Step {i+1}:** {step}")


# --- HÀM RESET STATE ---
def reset_state():
    """Xóa kết quả cũ khi chuyển chế độ"""
    keys = ["search_results", "detected_img", "detected_ingredients"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]

# ... (Giữ nguyên hàm load_search_engine, load_yolo_model, show_recipe_dialog) ...

# --- UI HIỂN THỊ KẾT QUẢ ---
def display_results_grid(results):
    if not results:
        st.warning("No results found.")
        return

    st.markdown(f"### 🍽️ Suggested Recipes ({len(results)})")
    cols = st.columns(3)
    for i, item in enumerate(results):
        col = cols[i % 3]
        with col:
            with st.container(border=True):
                # Xử lý hiển thị ảnh an toàn
                try:
                    # Nếu là đường dẫn local hoặc URL
                    st.image(item["image_path"], use_container_width=True)
                except:
                    st.image("https://via.placeholder.com/300?text=No+Image", use_container_width=True)
                
                st.markdown(f"**{item['title']}**")
                
                if st.button(f"Recipe", key=f"btn_{i}_{hash(item['title'])}"):
                    show_recipe_dialog(item)

# ---------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------
def main():
    st.set_page_config(page_title="Food Search", layout="wide", page_icon="🥗")
    st.title("🥗 Smart Food Assistant")

    search_engine = load_search_engine()
    yolo_model = load_yolo_model()

    # --- SIDEBAR: CHỈ CÒN 2 CHẾ ĐỘ ---
    with st.sidebar:
        st.header("Search Modes")
        mode = st.radio(
            "Select Mode",
            [
                "🔍 Recommend Recipe", 
                "📸 Similar Dishes"
            ],
            on_change=reset_state
        )
        
        st.markdown("---")
        k = st.slider("Number of results", 1, 20, 5)

    # Khởi tạo session state
    if "search_results" not in st.session_state:
        st.session_state.search_results = None

    # ============================================================
    # MODE 1: TRA CỨU CÔNG THỨC (TEXT + YOLO INGREDIENTS)
    # ============================================================
    if mode == "🔍 Recommend Recipe":
        st.subheader("What do you want to cook today?")
        
        with st.container(border=True):
            # Giao diện nhập liệu: Text bên trái, Upload ảnh bên phải
            col_text, col_img = st.columns([3, 1], gap="medium")
            
            with col_text:
                text_query = st.text_input(
                    "Enter dish name or ingredients:", 
                    placeholder="e.g., chicken, rice, broccoli",
                )
            
            with col_img:
                uploaded_file = st.file_uploader(
                    "Or upload an ingredient image:", 
                    type=["jpg", "png", "jpeg"],
                    help="AI sẽ nhận diện nguyên liệu trong ảnh để tìm công thức."
                )

            # Nút Search chung
            if st.button("🔍 Search", type="primary", use_container_width=True):
                
                # Reset kết quả cũ
                st.session_state.search_results = []
                st.session_state.detected_img = None
                st.session_state.detected_ingredients = []

                # --- TRƯỜNG HỢP A: DÙNG ẢNH (YOLO) ---
                if uploaded_file is not None:
                    if yolo_model is None:
                        st.error("Chưa load được model YOLO.")
                    else:
                        with st.spinner("Detecting ingredients..."):
                            # Lưu file tạm an toàn
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                tmp_path = tmp_file.name
                            
                            try:
                                # 1. YOLO Predict
                                results = yolo_model.predict(tmp_path, conf=0.25)
                                result = results[0]

                                # 2. Vẽ ảnh kết quả (Bounding Boxes)
                                bgr_array = result.plot()
                                st.session_state.detected_img = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)

                                # 3. Lấy tên nguyên liệu
                                detected_cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                                if len(detected_cls_ids) > 0:
                                    names_dict = result.names
                                    # Lấy danh sách tên (unique)
                                    detected_names = list(set([names_dict[cls_id] for cls_id in detected_cls_ids]))
                                    st.session_state.detected_ingredients = detected_names
                                    
                                    # Tạo query từ tên nguyên liệu
                                    query_from_image = ", ".join(detected_names)
                                    st.success(f"Detected ingredients: {query_from_image}")
                                    
                                    # Gọi hàm search text với từ khóa vừa tìm được
                                    search_results = search_engine.search_by_text(query_from_image, k)
                                    st.session_state.search_results = search_results
                                else:
                                    st.warning("Could not detect any ingredients in the image.")
                            
                            except Exception as e:
                                st.error(f"Error: {e}")
                            finally:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)

                # --- TRƯỜNG HỢP B: DÙNG TEXT (Nếu không có ảnh) ---
                elif text_query.strip():
                    with st.spinner(f"Loading: {text_query}..."):
                        results = search_engine.search_by_text(text_query, k)
                        st.session_state.search_results = results
                
                else:
                    st.warning("Please enter a dish name or upload an ingredient image.")

    # ============================================================
    # MODE 2: TÌM MÓN TƯƠNG TỰ (VISUAL SIMILARITY - CLIP)
    # ============================================================
    elif mode == "📸 Similar Dishes":
        st.subheader("Find dishes with similar images")
        
        uploaded_file = st.file_uploader("Upload a sample dish image:", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(uploaded_file, caption="Ảnh gốc", use_container_width=True)
            with col2:
                if st.button("🚀 Find similar dishes", type="primary"):
                    with st.spinner("Searching for similar dishes..."):
                        # Lưu file tạm
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        try:
                            # Gọi hàm search by image (CLIP)
                            results = search_engine.search_by_image(tmp_path, k)
                            st.session_state.search_results = results
                        finally:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

    # ============================================================
    # PHẦN HIỂN THỊ KẾT QUẢ CHUNG (GLOBAL DISPLAY)
    # ============================================================
    st.divider()

    # 1. Hiển thị ảnh nhận diện YOLO (Chỉ hiện khi ở Mode 1 và có ảnh)
    if mode == "🔍 Recommend Recipe" and "detected_img" in st.session_state and st.session_state.detected_img is not None:
        st.markdown("### 👁️ Detected Ingredients")
        col_yolo1, col_yolo2 = st.columns([1, 2])
        with col_yolo1:
            st.image(st.session_state.detected_img, caption="Detected ingredients", use_container_width=True)
        with col_yolo2:
            if st.session_state.detected_ingredients:
                st.info("The system has automatically found recipes based on the detected ingredients.")
                # Hiển thị tags
                st.write("Detected ingredients:")
                tags = "".join([f"<span style='background:#e8f5e9; color:#2e7d32; padding:5px 10px; border-radius:15px; margin:2px; font-weight:bold'>{name}</span>" for name in st.session_state.detected_ingredients])
                st.markdown(tags, unsafe_allow_html=True)

    # 2. Hiển thị Grid kết quả tìm kiếm (Chung cho cả 2 mode)
    if st.session_state.search_results is not None:
        display_results_grid(st.session_state.search_results)

if __name__ == "__main__":
    main()
