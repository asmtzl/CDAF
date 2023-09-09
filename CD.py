# Contrast based aouto focus
import  cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
#get frames in folder 
Frames_path = r"archive"
img_names = glob.glob(f"{Frames_path}/*.jpg")

print(f"Iterating over {len(img_names)} frames at: {Frames_path}")

old_grad = None
min = 0
diff_arr =[]
#get all images in for loop
for img_name in img_names:
    diff = int(0)
    img = cv2.imread(img_name)
    #downscale the image for faster processing
    height, width , x= img.shape
    width = round(width/3)
    height = round(height/3)
    img = cv2.resize(img,(width,height))

    #convert grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #apply sobel filter x and y dimension
    grad_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
    grad_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    #combine x and y dimensions of gradient
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    cv2.imshow('1',grad)
    key = cv2.waitKey(0)
    if key == ord("q") or key == 27:
        cv2.destroyAllWindows()
    
    #find the most amount of difference of contrast
    for x in grad:
        i = 0
        while i+1 < len(x):
            if x[i] > x[i+1]:
                diff -= (int(x[i])-int(x[i+1]))
            else:
                diff -= (int(x[i+1])-int(x[i]))
            i+=1

    Tgrad = np.transpose(grad)
    for x in Tgrad:
        i = 0
        while i+1 < len(x):
            if x[i] > x[i+1]:
                diff -= (int(x[i])-int(x[i+1]))
            else:
                diff -= (int(x[i+1])-int(x[i]))
            i+=1
        
    print(diff)
    diff_arr.append(diff/1000000)
    if diff < int(min):
        min = diff
        index = img_name
        imag = img
    
plt.plot(range(len(diff_arr)), diff_arr,marker="o")
plt.xlabel("image")
plt.ylabel("contrast difference (m)")

print("img_name: ", index)
cv2.imshow('focus',imag),plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
