import threading
import cv2
import numpy as np

time_count = 0
posTot = [0, 0]
posAvg = [0, 0]


def getvelocity():
    prevPos = [0, 0]
    nowPos = [0, 0]
    veloVec = [0, 0]
    if time_count < 1:
        prevPos = posAvg
    else:
        nowPos = posAvg
        for i in range(2):
            veloVec[i] = abs(nowPos[i] - prevPos[i])
        print(veloVec)
        prevPos = nowPos


def startTimer():
    global time_count
    time_count += 1
    getvelocity()
    timer = threading.Timer(1, startTimer)
    timer.start()

    if time_count > 10:
        print("stop")
        timer.cancel()


class cameraCV:
    global posTot
    global posAvg

    def __init__(self, cam_w=640, cam_h=480):  # slef는 뭘까
        self.cap = cv2.VideoCapture(1)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

        self.rect = 1
        self.threshold = 100
        self.h = 180
        self.lower_blue1 = np.array([105, 35, 35])
        self.upper_blue1 = np.array([115, 255, 255])
        self.lower_blue2 = np.array([95, 35, 35])
        self.upper_blue2 = np.array([105, 255, 255])
        self.lower_blue3 = np.array([95, 35, 35])
        self.upper_blue3 = np.array([105, 255, 255])

    def asdf(self):
        while True:
            ret, img_color = self.cap.read()
            org_height, org_width = img_color.shape[:2]
            # print(height, width)

            img_color = cv2.resize(
                img_color, (org_width, org_height), interpolation=cv2.INTER_AREA
            )

            # 원본 영상을 HSV 영상으로 변환합니다.
            img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

            # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
            img_mask1 = cv2.inRange(img_hsv, self.lower_blue1, self.upper_blue1)
            img_mask2 = cv2.inRange(img_hsv, self.lower_blue2, self.upper_blue2)
            img_mask3 = cv2.inRange(img_hsv, self.lower_blue3, self.upper_blue3)

            img_mask = img_mask1 | img_mask2 | img_mask3

            # 보간법을 사용해서, 마스킹을 진행해 준다
            kernel = np.ones((11, 11), np.uint8)
            img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel)
            img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel)
            # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
            img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)

            # 파이썬에서 추적한 물체에 대한 정보를 주는 함수, 이것으로 쉽게 박스를 그릴 수 있다.
            numOfLables, img_label, stats, centroids = cv2.connectedComponentsWithStats(
                img_mask
            )
            centers = 0
            for idx, centroid in enumerate(centroids):
                centers += 1
                if stats[idx][0] == 0 and stats[idx][1] == 0:
                    continue
                if np.any(np.isnan(centroid)):
                    continue

                x, y, width, height, area = stats[idx]
                centerX, centerY = int(centroid[0]), int(centroid[1])
                # print(centerX, centerY)

                # tracker를 떠올리자. 전 프레임에 추적했던 위치, 넓이와 이번 위치 넓이의 일치율이 40% 이상이 되면 같은 물체라 판단하자
                if area > 50:
                    posTot[0] += x
                    posTot[1] += y
                    cv2.circle(img_color, (centerX, centerY), 10, (0, 0, 255), 10)
                    cv2.rectangle(
                        img_color, (x, y), (x + width, y + height), (0, 0, 255)
                    )

            for i in range(2):
                posAvg[i] = posTot[i] / 2
                # 초기화
                posTot[i] = 0

            cv2.imshow("img_color", img_color)
            cv2.imshow("img_mask", img_mask)
            cv2.imshow("img_result", img_result)

            k = cv2.waitKey(5)
            if k == ord("q") or time_count > 10:
                break


startTimer()

cameraCV().asdf()
