import streamlit as st
#import lib.common as tools

st.set_page_config(
    page_title="Đồ án cuối kỳ",
    page_icon="📖",
)

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-vector/gradient-grainy-texture_23-2148981502.jpg?w=1380&t=st=1701576249~exp=1701576849~hmac=c31e813f8bff0b5218da38e2815b2218f516bc79e9c21fc9b281f3c0dd5a557b");
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("https://img.freepik.com/free-photo/abstract-textured-backgound_1258-30489.jpg?w=740&t=st=1701580457~exp=1701581057~hmac=25da83b2b1b6cf489fbc5a0ee890c79ff26156910c8b2e6a218c0a12bf939422");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img,unsafe_allow_html=True)


# logo_path = "./VCT.png"
# st.sidebar.image(logo_path, width=200)

st.markdown('<div>'
                '<h1 style="color: white;">📖 Đồ án cuối kỳ</h1>'
		        '<h1 style="color: white;">🧑 GVHD: Ths.Trần Tiến Đức</h1>'
                '<h1 style="color: white;">🧑 Trịnh Ngọc Thương - 21110673</h1>'
                '<h1 style="color: white;">👩 Bùi Thị Xuân Lương - 21142316</h1>'
                '<h1 style="color: white;">🏫 Mã lớp : DIPR430685_23_1_02</h1>'
                '<h2 style="color: white;">Sản phẩm</h2>'
                '<h5 style="color: white;">Project cuối kỳ cho môn học Xử Lý Ảnh Số, mã môn học DIPR430685, thuộc Trường Đại Học Sư Phạm Kỹ Thuật TPHCM.</h5>'
                '<h2 style="color: white;">10 chức năng chính trong bài</h2>'
                '<h5 style="color: white;">6 chức năng cơ bản</h5>'
                '<h5 style="color: white;">- 🌝 Giải phương trình bậc hai</h5>'
                '<h5 style="color: white;">- 🌝 Nhận diện chữ số Mnist</h5>'
                '<h5 style="color: white;">- 🌝 Nhận diện khuôn mặt</h5>'
                '<h5 style="color: white;">- 🌝 Nhận diện 5 loại trái cây</h5>'
                '<h5 style="color: white;">- 🌝 Nhận diện đối tượng YOLO4</h5>'
                '<h5 style="color: white;">- 🌝 Xử lý ảnh số</h5>'
                '<h5 style="color: white;">4 chức năng bổ sung</h5>'
                '<h5 style="color: white;">- 🌟 Đếm số lần nháy mắt</h5>'
                '<h5 style="color: white;">- 🌟 Face Mesh</h5>'
                '<h5 style="color: white;">- 🌟 Nhận diện cảm xúc</h5>'
                '<h5 style="color: white;">- 🌟 Trò chơi trả lời câu đố</h5>'
            '</div>', unsafe_allow_html=True)



