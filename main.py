import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Generate a (2k+1)x(2k+1) gaussian kernel with mean=0 and sigma = s
s, k = 5, 10
probs = [np.exp(-z*z/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-k,k+1)]
kernelLow = np.outer(probs, probs)

# import matplotlib.pylab as plt
# plt.imshow(kernelLow)
# plt.colorbar()
# plt.show()

kernelGradX = np.array([[1, -1]])
kernelGradY = np.array([[1, ],
                        [-1,]])

while True:
    ret, frame = cap.read()
    img = np.double(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))/255
    
    imgLow = cv2.filter2D(img, -1, kernelLow)
    imgHigh = img-imgLow
    
    imgGradX = cv2.filter2D(img, -1, kernelGradX)
    imgGradY = cv2.filter2D(img, -1, kernelGradY)
    imgGrad = np.sqrt(imgGradX**2 + imgGradY**2)

    imgBandReject = imgLow + imgGrad
    imgBandPass = img - imgBandReject
    
    cv2.imshow('Citra Asli', img)
    cv2.imshow('Low Pass', imgLow)
    cv2.imshow('High Pass', imgHigh)
    
    cv2.imshow('Gradient X', imgGradX*3)
    cv2.imshow('Gradient Y', imgGradY*3)
    cv2.imshow('Turunan Pertama', imgGrad*3)
    
    cv2.imshow('Band Reject', imgBandReject)
    cv2.imshow('Band Pass', imgBandPass*3)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()