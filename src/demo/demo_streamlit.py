import sys
import os
import streamlit as st
from pathlib import Path

# --- CẤU HÌNH ĐƯỜNG DẪN ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.append(SRC_DIR)

# Import module của bạn (Giả định code này chạy trong môi trường của bạn)
try:
    from retrieval.search import FoodRetrievalSystem
except ImportError:
    # Class giả lập để code chạy được nếu người khác test mà không có module của bạn
    class FoodRetrievalSystem:
        def search_by_text(self, q, k): return [{"title": f"Demo Food {i}", "image_path": "https://via.placeholder.com/150", "ingredients": "A, B", "instructions": "Cook it"} for i in range(k)]
        def search_by_image(self, p, k): return [{"title": f"Demo Food {i}", "image_path": "https://via.placeholder.com/150", "ingredients": "A, B", "instructions": "Cook it"} for i in range(k)]

st.set_page_config(page_title="Food Search", layout="wide")

# ---------------------------------------------
# 1. TỐI ƯU HÓA: Caching Model
# ---------------------------------------------
@st.cache_resource
def load_search_engine():
    """Load model một lần duy nhất, tránh load lại khi reload trang"""
    return FoodRetrievalSystem()

# ---------------------------------------------
# 2. UI COMPONENTS: Tách hàm hiển thị
# ---------------------------------------------
@st.dialog("Recipe Detail")
def show_recipe_dialog(item):
    """Hàm hiển thị Modal chi tiết"""
    # Xử lý hiển thị ảnh (local path hoặc url)
    try:
        st.image(item["image_path"], width=450)
    except:
        st.warning("Image not found")
        
    st.subheader(item["title"])
    
    st.markdown("### 🥦 Ingredients")
    st.info(item["ingredients"]) # Dùng info box cho đẹp
    
    st.markdown("### 🍳 Instructions")
    st.write(item["instructions"])

def display_results_grid(results):
    """Hàm hiển thị kết quả dạng lưới dùng chung cho cả Text và Image"""
    if not results:
        st.warning("No results found.")
        return

    cols = st.columns(3) # Grid 3 cột
    for i, item in enumerate(results):
        col = cols[i % 3]
        with col:
            with st.container(border=True): # Tạo khung viền cho đẹp
                try:
                    st.image(item["image_path"], use_container_width=True)
                except:
                    st.text("Image N/A")
                
                # Nút bấm mở modal
                if st.button(f"📖 {item['title']}", key=f"btn_{i}_{item['title']}"):
                    show_recipe_dialog(item)

# ---------------------------------------------
# Streamlit Application
# ---------------------------------------------
def main():
    st.title("🥗 Food Retrieval System")

    # Load engine (đã cache)
    search_engine = load_search_engine()

    # Sidebar setup
    with st.sidebar:
        st.header("Search Settings")
        mode = st.radio("Query Mode", ["Text Search", "Image Search"])
        k = st.slider("Top-K results", 1, 20, 5)
        st.markdown("---")
        if st.button("Clear History"):
            if "search_results" in st.session_state:
                del st.session_state["search_results"]
            st.rerun()

    # Khởi tạo state cho kết quả tìm kiếm nếu chưa có
    if "search_results" not in st.session_state:
        st.session_state.search_results = None

    # --- LOGIC XỬ LÝ ---
    if mode == "Text Search":
        st.subheader("🔎 Search by Text")
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input("What would you like to eat?", placeholder="e.g. Pasta with tomato sauce")
        with col2:
            st.write("") 
            st.write("") 
            search_btn = st.button("Search", use_container_width=True)

        if search_btn:
            if not query.strip():
                st.warning("Please enter a query")
            else:
                with st.spinner("Searching delicious recipes..."):
                    # Lưu kết quả vào session_state
                    results = search_engine.search_by_text(query, k)
                    st.session_state.search_results = results

    else: # Image Search
        st.subheader("📸 Search by Image")
        uploaded_file = st.file_uploader("Upload food image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            col_img, col_btn = st.columns([1, 2])
            with col_img:
                st.image(uploaded_file, caption="Query Image", width=200)
            
            with col_btn:
                if st.button("Search Similar Food"):
                    with st.spinner("Analyzing image..."):
                        # Lưu ảnh tạm thời
                        temp_path = Path("uploaded_query.jpg")
                        temp_path.write_bytes(uploaded_file.getvalue())
                        
                        # Search và lưu kết quả vào session state
                        results = search_engine.search_by_image(str(temp_path), k)
                        st.session_state.search_results = results

    st.divider()

    # --- HIỂN THỊ KẾT QUẢ TỪ SESSION STATE ---
    # Việc hiển thị nằm ngoài logic nút bấm để không bị mất khi rerun
    if st.session_state.search_results is not None:
        st.markdown(f"### Top Results")
        display_results_grid(st.session_state.search_results)

if __name__ == "__main__":
    main()