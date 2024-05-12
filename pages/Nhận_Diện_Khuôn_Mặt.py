import streamlit as st
import numpy as np
import cv2 as cv
import joblib

st.markdown('<div>'
                '<h1 style="color: white;">Nhận diện khuôn mặt</h1>'
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

FRAME_WINDOW = st.image([])
cap = cv.VideoCapture(0)

if 'stop' not in st.session_state:
    st.session_state.stop = False
    stop = False

press = st.button('Stop')
if press:
    if st.session_state.stop == False:
        st.session_state.stop = True
        cap.release()
    else:
        st.session_state.stop = False

print('Trang thai nhan Stop', st.session_state.stop)

if 'frame_stop' not in st.session_state:
    frame_stop = cv.imread('D:/3year/XLAS/streamlit_final/streamlit_final/NhanDangKhuonMat_onnx_Streamlit/stop.jpg')
    st.session_state.frame_stop = frame_stop
    print('Đã load stop.jpg')

if st.session_state.stop == True:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')


svc = joblib.load('D:/3year/XLAS/streamlit_final/streamlit_final/NhanDangKhuonMat_onnx_Streamlit/svc.pkl')
mydict = ['Han', 'Lam', 'Luong', 'Nghi', 'Thuong']

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
 'D:/3year/XLAS/streamlit_final/streamlit_final/NhanDangKhuonMat_onnx_Streamlit/face_detection_yunet_2023mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    
    recognizer = cv.FaceRecognizerSF.create(
    'D:/3year/XLAS/streamlit_final/streamlit_final/NhanDangKhuonMat_onnx_Streamlit/face_recognition_sface_2021dec.onnx',"")

    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        y = 50

        if faces[1] is not None:
            for face_info in faces[1]:
                face_align = recognizer.alignCrop(frame, face_info)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]
                cv.putText(frame, result, (1, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y = y + 20
        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()
