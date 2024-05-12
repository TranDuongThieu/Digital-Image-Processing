import cv2
import mediapipe as mp
import streamlit as st

st.markdown('<div>'
                '<h1 style="color: white;">Face Mesh</h1>'
                '<p style="color: white;">Face Mesh là một bài toán nhận diện một loạt các điểm trên khuôn mặt, từ đó tạo thành 1 lưới (mesh) của mặt.</p>'
            '</div>', unsafe_allow_html=True)

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

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

video = cv2.VideoCapture(0)

# Tạo một placeholder để hiển thị frame trong Streamlit
frame_placeholder = st.empty()

while True:
    check, frame = video.read()
    height, width, _ = frame.shape
    result = face_mesh.process(frame)

    try:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                landmrk = facial_landmarks.landmark[i]
                locx = int(landmrk.x * width)
                locy = int(landmrk.y * height)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.circle(frame, (locx, locy), 1, (0, 200, 0), 0)

    except:
        pass

    frame_placeholder.image(frame, channels="RGB")

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()