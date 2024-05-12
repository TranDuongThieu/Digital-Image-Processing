import streamlit as st
import math

st.markdown('<div>'
                '<h1 style="color: white;">Giải phương trình bậc 2</h1>'
                '<p style="color: white;">Phương trình bậc 2 là phương trình có dạng ax^2+bx+c=0</p>'
            '</div>', unsafe_allow_html=True)

def gptb2(a, b, c):
    if a == 0:
        if b == 0:
            if c == 0:
                ket_qua = 'PTB1 có vô số nghiệm'
            else:
                ket_qua = 'PTB1 vô nghiệm'
        else:
            x = -c/b
            ket_qua = 'PTB1 có nghiệm %.2f' % x
    else:
        delta = b**2 - 4*a*c
        if delta < 0:
            ket_qua = 'PTB2 vô nghiệm'
        else:
            x1 = (-b + math.sqrt(delta))/(2*a)
            x2 = (-b - math.sqrt(delta))/(2*a)
            ket_qua = 'PTB2 có nghiệm x1 = %.2f và x2 = %.2f' % (x1, x2)
    return ket_qua

def clear_input():
    st.session_state["nhap_a"] = 0.0
    st.session_state["nhap_b"] = 0.0
    st.session_state["nhap_c"] = 0.0

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

with st.form(key='columns_in_form', clear_on_submit = False):
    a = st.number_input('Nhập a', key = 'nhap_a')
    b = st.number_input('Nhập b', key = 'nhap_b')
    c = st.number_input('Nhập c', key = 'nhap_c')
    c1, c2 = st.columns(2)
    with c1:
        btn_giai = st.form_submit_button('Giải')
    with c2:
        btn_xoa = st.form_submit_button('Xóa', on_click=clear_input)
    if btn_giai:
        s = gptb2(a, b, c)
        st.markdown('Kết quả: ' + s)
    else:
        st.markdown('Kết quả:')
