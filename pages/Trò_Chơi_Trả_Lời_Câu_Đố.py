import cv2
import csv
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import cvzone
import time

st.markdown('<div>'
                '<h1 style="color: white;">Trò chơi trả lời câu đố</h1>'
                '<p style="color: white;">Trò chơi trả lời câu đố dựa trên chuyển động của ngón tay.</p>'
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

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)


class MCQ():
    def __init__(self, data):
        self.question = data[0]
        self.choice1 = data[1]
        self.choice2 = data[2]
        self.choice3 = data[3]
        self.choice4 = data[4]
        self.answer = int(data[5])

        self.userAns = None

    def update(self, cursor, bboxs):

        for x, bbox in enumerate(bboxs):
            x1, y1, x2, y2 = bbox
            if x1 < cursor[0] < x2 and y1 < cursor[1] < y2:
                self.userAns = x + 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)


# Import csv file data
pathCSV = "D:/3year/XLAS/streamlit_final/streamlit_final/QuizGame_streamlit/Mcqs.csv"
with open(pathCSV, newline='\n') as f:
    reader = csv.reader(f)
    dataAll = list(reader)[1:]

# Create Object for each MCQ
mcqList = []
for q in dataAll:
    mcqList.append(MCQ(q))

qNo = 0
qTotal = len(dataAll)

frame = st.empty()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if qNo < qTotal:
        mcq = mcqList[qNo]

        img, bbox = cvzone.putTextRect(img, mcq.question, [100, 100], 2, 2, offset=50, border=5)
        img, bbox1 = cvzone.putTextRect(img, mcq.choice1, [100, 250], 2, 2, offset=50, border=5)
        img, bbox2 = cvzone.putTextRect(img, mcq.choice2, [400, 250], 2, 2, offset=50, border=5)
        img, bbox3 = cvzone.putTextRect(img, mcq.choice3, [100, 400], 2, 2, offset=50, border=5)
        img, bbox4 = cvzone.putTextRect(img, mcq.choice4, [400, 400], 2, 2, offset=50, border=5)

        if hands:
            lmList = hands[0]['lmList']
            cursor = lmList[8]
            length, info, img = detector.findDistance(lmList[8][0:2], lmList[12][0:2], img)
            if length < 35:
                mcq.update(cursor, [bbox1, bbox2, bbox3, bbox4])
                if mcq.userAns is not None:
                    time.sleep(0.3)
                    qNo += 1
    else:
        score = 0
        for mcq in mcqList:
            if mcq.answer == mcq.userAns:
                score += 1
        score = round((score / qTotal) * 100, 2)
        img, _ = cvzone.putTextRect(img, "Quiz Completed", [250, 300], 2, 2, offset=50, border=5)
        img, _ = cvzone.putTextRect(img, f'Your Score: {score}%', [700, 300], 2, 2, offset=50, border=5)

    # Draw Progress Bar
    barValue = 150 + (950 // qTotal) * qNo
    cv2.rectangle(img, (150, 600), (barValue, 650), (0, 255, 0), cv2.FILLED)
    cv2.rectangle(img, (150, 600), (1100, 650), (255, 0, 255), 5)
    img, _ = cvzone.putTextRect(img, f'{round((qNo / qTotal) * 100)}%', [1130, 635], 2, 2, offset=16)

    frame.image(img, channels="BGR")
    #st.image(img, channels="BGR")