import cv2
import numpy as np

c = cv2.VideoCapture(0)
c.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
c.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

rect = 1
threshold = 100
h = 180
lower_blue1 = np.array([105, 35, 35])
upper_blue1 = np.array([115, 255, 255])
lower_blue2 = np.array([95, 35, 35])
upper_blue2 = np.array([105, 255, 255])
lower_blue3 = np.array([95, 35, 35])
upper_blue3 = np.array([105, 255, 255])

cap = cv2.VideoCapture(1)

centerPos = [0, 0, 0]


def drawRct(rctNum, width, height, img):
    if rctNum == 0:
        cv2.rectangle(img, (0, 0), (int(width / 3), height), (0, 255, 0), 3)
    elif rctNum == 1:
        cv2.rectangle(
            img, (int(width / 3), 0), (int(width / 3 * 2), height), (255, 0, 0), 3
        )
    elif rctNum == 2:
        cv2.rectangle(img, (int(width / 3 * 2), 0), (width, height), (0, 0, 255), 3)
    else:
        return


def maxPos():
    max = -1
    pivot = -1
    for i in range(3):
        if max < centerPos[i]:
            max = centerPos[i]
            pivot = i

    drawRct(pivot, org_width, org_height, img_result)
    for i in range(3):
        centerPos[i] = 0


def window():
    while True:
        # img_color = cv.imread('C:/Users/COM-11/Documents/hsv.jpg')
        ret, img_color = cap.read()
        org_height, org_width = img_color.shape[:2]
        # print(height, width)

        img_color = cv2.resize(
            img_color, (org_width, org_height), interpolation=cv2.INTER_AREA
        )

        # 원본 영상을 HSV 영상으로 변환합니다.
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

        # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
        img_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
        img_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
        img_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)

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
        for idx, centroid in enumerate(centroids):
            if stats[idx][0] == 0 and stats[idx][1] == 0:
                continue
            if np.any(np.isnan(centroid)):
                continue

            x, y, width, height, area = stats[idx]
            centerX, centerY = int(centroid[0]), int(centroid[1])
            # print(centerX, centerY)

            # tracker를 떠올리자. 전 프레임에 추적했던 위치, 넓이와 이번 위치 넓이의 일치율이 40% 이상이 되면 같은 물체라 판단하자
            """
            if area > 50:
                cv2.circle(img_color, (centerX, centerY), 10, (0, 0, 255), 10)
                cv2.rectangle(img_color, (x, y), (x + width, y + height), (0, 0, 255))
            if centerX < int(org_width / 3):
                centerPos[0] += 1
            elif centerX < int(org_width / 3 * 2):
                centerPos[1] += 1
            else:
                centerPos[2] += 1
            """
        # print(numOfLables)
        # maxPos()

        # if numOfLables == 1:
        #    print(centroids)
        # if numOfLables == 2:
        #    print("target locking sequencing activated")

        cv2.imshow("img_color", img_color)
        cv2.imshow("img_mask", img_mask)
        cv2.imshow("img_result", img_result)

        k = cv2.waitKey(5)
        if k == ord("q"):
            break


cv2.destroyAllWindows()
