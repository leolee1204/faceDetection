import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe('face-detection-master/deploy.prototxt.txt',
                               'face-detection-master/res10_300x300_ssd_iter_140000.caffemodel')

image = cv2.imread('peoples.jpg')
h,w,c = image.shape

'''
模型要求是BGR
圖片尺寸是用我們的圖片尺寸
要求每個通道像素要減除一個常數
[104.,117.,123.] 
mean: 用於各通道減去的值,以降低亮度的影響
'''
blob = cv2.dnn.blobFromImage(image,1.0,(512,512),[104.,117.,123.],False,False)

# image 進入模型
net.setInput(blob)
#開始計算
detections = net.forward()

# [0,1,信心水準,x比例,y比例,end x比例 end y比例]
# print(detections[0][0][0])

detected_faces = 0
# detections.shape[2] 是偵測到的個數
for i in range(detections.shape[2]):
    confidence = detections[0][0][i][2]

    if confidence > 0.3:
        detected_faces +=1

        # 把算出來還原成原本的尺寸，為了取得座標位置
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])

        #座標轉為整數
        (startX,startY,endX,endY) = box.astype('int')

        cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),3)
        #信心水準
        text = '{:.2f}%'.format(confidence*100)
        y = startY - 10 if startY -10 > 10 else startY + 10
        cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

cv2.imwrite('peoples_result.png',image)
cv2.destroyAllWindows()

