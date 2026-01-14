import cv2
import numpy as np

print(cv2.getBuildInformation())

# 创建一个黑色画布
img = np.zeros((400, 400, 3), dtype=np.uint8)

# 画一个绿色圆
cv2.circle(img, (200, 200), 100, (0, 255, 0), -1)

# 画一个红色矩形
cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), 3)

# 显示窗口
cv2.imshow("OpenCV Test", img)

# 等待用户按键关闭
cv2.waitKey(0)
cv2.destroyAllWindows()
