# coding=gbk
import time
import cv2
cap = cv2.VideoCapture(0)  # 读取文件
start_time = time.time()
counter = 0
# 获取视频宽度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 获取视频高度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) #视频平均帧率
print(fps)
while (True):

    ret, frame = cap.read()
    # 键盘输入空格暂停，输入q退出
    key = cv2.waitKey(1) & 0xff
    if key == ord(" "):
        cv2.waitKey(0)
    if key == ord("q"):
        break
    counter += 1  # 计算帧数
    if (time.time() - start_time) != 0:  # 实时显示帧数
        cv2.putText(frame, "FPS {0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
        # src = cv2.resize(frame, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_CUBIC)  # 窗口大小
        cv2.imshow('frame', frame)
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
    # time.sleep(1 / fps)  # 按原帧率播放
cap.release()
cv2.destroyAllWindows()
