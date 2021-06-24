import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('models/instance_norm/mosaic.t7')
net2 = cv2.dnn.readNetFromTorch('models/instance_norm/the_scream.t7')
net3 = cv2.dnn.readNetFromTorch('models/instance_norm/candy.t7')

cap = cv2.VideoCapture('imgs/03.mp4')

while True:
    ret, img = cap.read()

    if ret == False:
        break

    h, w, c = img.shape

    img = cv2.resize(img, dsize=(500, int(h / w * 500)))

    MEAN_VALUE = [103.939, 116.779, 123.680]
    blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

    net.setInput(blob)
    output = net.forward()

    output = output.squeeze().transpose((1, 2, 0))

    output += MEAN_VALUE
    output = np.clip(output, 0, 255)
    output = output.astype('uint8')

    net2.setInput(blob)
    output2 = net2.forward()

    output2 = output2.squeeze().transpose((1, 2, 0))

    output2 += MEAN_VALUE
    output2 = np.clip(output2, 0, 255)
    output2 = output2.astype('uint8')

    net3.setInput(blob)
    output3 = net3.forward()

    output3 = output3.squeeze().transpose((1, 2, 0))

    output3 += MEAN_VALUE
    output3 = np.clip(output3, 0, 255)
    output3 = output3.astype('uint8')

    output = output[0:100, :]
    output2 = output2[100:200, :]
    output3 = output3[200:, :]

    output4 = np.concatenate([output, output2, output3], axis=0)

    cv2.imshow('result', output4)

    if cv2.waitKey(1) == ord('q'):
        break