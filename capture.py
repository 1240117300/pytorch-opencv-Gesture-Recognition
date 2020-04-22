import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 48)
i = 0
while 1:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (cv2.waitKey(1) & 0xFF) == ord('s'):  # 不断刷新图像，这里是1ms 返回值为当前键盘按键值
        cv2.imwrite('./image/%d.jpg' % i, gray)
        i += 1
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break
    cv2.imshow("frame", gray)
cap.release()
cv2.destroyAllWindows()
