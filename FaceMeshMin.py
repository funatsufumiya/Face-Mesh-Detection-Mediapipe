import cv2
import mediapipe as mp
import time
import argparse
import sys
from cv2_enumerate_cameras import enumerate_cameras

def print_camera_list():
    for camera_info in enumerate_cameras():
        print(f'{camera_info.index}: {camera_info.name}')

parser = argparse.ArgumentParser(description="face mesh detector",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c","--camera", type=int, default=0, help="camera id")
parser.add_argument("-l","--list-camera", action="store_true", default=False, help="print camera list and exit")

args = parser.parse_args()

print_camera_list()

if args.list_camera:
    sys.exit(0)

cap = cv2.VideoCapture(args.camera)
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=10)
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 0:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                    print(mpFaceMesh)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    