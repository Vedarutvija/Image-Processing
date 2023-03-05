#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import cv2
import numpy 
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, GifImagePlugin


# #### After running the below codes press any key to destroy the windows after the images are viewed

# # 1. Resolution Change

# In[6]:


res20 = cv2.imread("C:/Users/rutvi/Downloads/img300.jpg") #600DPI
res20 = cv2.resize(res20,(300,300))
img_jpg = cv2.imread("img.jpg") #120 DPI
img_jpg = cv2.resize(img_jpg,(300,300))
#different resolution
if res20.shape == img_jpg.shape:
    diff = cv2.subtract(img_jpg,res20)
    b , g , r= cv2.split(diff)
    Conv_hsv_Gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    diff[mask != 255] = [0, 0, 255]
    cv2.imshow("diff", diff)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("images are same")
    else:
        print("images are different")
res20[mask != 255] = [0, 0, 255]
img_jpg[mask != 255] = [0, 0, 255]
cv2.imshow("res20",res20)
cv2.imshow("Rose jpg",img_jpg)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


res20 == img_jpg


# In[ ]:


## we used the red color mask to display the image clearly. If the mask is not created we may find a blank image with minor blue lines 


# ### Using Pillow

# In[7]:


res = Image.open("C:/Users/rutvi/Downloads/img300.jpg") 
res = res.resize((1000,650))
rgb_res = res.convert("RGB")

jpg_img = Image.open("img.jpg")
jpg_img = jpg_img.resize((1000,650))
rgb_jpg = jpg_img.convert("RGB")
diff = ImageChops.difference(rgb_res,rgb_jpg)
pixels = diff.load()
width, height = diff.size
for x in range(width):
    for y in range(height):
        r, g, b = pixels[x, y]
        if r+g+b > 0:
            pixels[x, y] = (255, 0, 0)
if diff.getbbox():
    diff.show()


# In[9]:


res == jpg_img


# In[10]:


rgb_res == rgb_jpg


# # 2.Size Change

# In[16]:


img_s = cv2.imread("C:/Users/rutvi/Downloads/img.jpg")
img_z = cv2.imread("C:/Users/rutvi/Downloads/img_size.jpg")


# In[17]:


img_s.shape


# In[13]:


img_z.shape


# In[24]:


img_s = cv2.resize(img_s,(300,300))
img_z = cv2.resize(img_z,(300,300))


# In[25]:


# JPG & PNG both are different
if img_s.shape == img_z.shape:
    diff = cv2.subtract(img_s,img_z)
    b , g , r= cv2.split(diff)
    
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("images are same")
    else:
        print("images are different")
        
cv2.imshow("diff",diff)
cv2.imshow("small",img_s)
cv2.imshow("large",img_z)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ### using pillow

# In[29]:


size2000 = Image.open("img.jpg") 
size500 = Image.open("img_size.jpg")

diff = ImageChops.difference(size300,size200)
if diff.getbbox():
    diff.show()


# In[31]:


size2000 == size500


# # 3. JPG vs PNG

# In[4]:


jpg = cv2.imread("img.jpg")
png = cv2.imread("img.png")
jpg = cv2.resize(jpg,(500,500))
png = cv2.resize(png,(500,500))

# JPG & PNG both are different
if jpg.shape == png.shape :
    diff = cv2.subtract(jpg,png)
    b , g , r= cv2.split(diff)
    cv2.imshow("difference", diff)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("images are same")
    else:
        print("images are different")
        
cv2.imshow("jpg",jpg)
cv2.imshow("png",png)

cv2.waitKey(0)
cv2.destroyAllWindows()


# #### some pixel values are not same

# In[13]:


jpg == png


# ## Using pillow

# In[7]:


jpg1 = Image.open("img.jpg") 
rgb1 = jpg1.convert("RGB")
png1 = Image.open("img.png")
rgb2 = png1.convert("RGB")
diff = ImageChops.difference(rgb1,rgb2)

if diff.getbbox():
    diff.show()
    print("images are different")
    
else:
    print("images are the same")

   


# In[22]:


jpg1 == png1


# In[23]:


rgb1 == rgb2


# # 4. TIFF Vs BMP

# In[1]:


tiff = cv2.imread("img.tiff")
bmp = cv2.imread("tiff_to.bmp")

tiff = cv2.resize(tiff,(1000,650))
bmp = cv2.resize(bmp,(1000,650))

# tiff & JPG both are same
if tiff.shape == bmp.shape :
    diff = cv2.subtract(tiff,bmp)
    b , g , r= cv2.split(diff)
    cv2.imshow("diff", diff)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("images are same")
    else:
        print("images are different")
        
cv2.imshow("tiff",tiff)
cv2.imshow("bmp",bmp)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ### Using pillow: RUN THE PROGRAM IN PYCHARM 

# In[ ]:


# RUN THIS PROGRAM IN PYCHARM TO ESCAPE KERNAL FROM DYING

tiff = Image.open("C:/Users/rutvi/Downloads/img.tiff")
bmp = Image.open("C:/Users/rutvi/Downloads/tiff_to.bmp")

diff = ImageChops.difference(tiff,bmp)
for x in range(diff.width):
    for y in range(diff.height):
        # Get the color of the current pixel
        r, g, b,a = diff.getpixel((x, y))

        # If the pixel is not black, it means it's a difference
        if r + g + b + a> 0:
            # Set the color of the pixel to red
            diff.putpixel((x, y), (255, 0, 0))

# Show the difference image
diff.show()

rgb1 = tiff.convert("RGB")
rgb2 = bmp.convert("RGB")

dif = ImageChops.difference(rgb1,rgb2)
dif.show()

a = Image.open('C:/Users/rutvi/Downloads/img.tiff')
b = Image.open('C:/Users/rutvi/Downloads/tiff_to.bmp')

print("Are images same: ?? ",tiff==bmp)

a = list(tiff.getdata())
b = list(bmp.getdata())

print("Are pixel values same: ?? ", a == b)
print("Hence 1. images are not same, 2. Pixel values are same, 3. Size is same, 4. But there is a shade of differences seen when observed carefully on graphical image")



# # 5. JPG vs GIF

# ### Comparing a frame with all the other frames iteratively

# In[3]:


gif = cv2.VideoCapture("C:/Users/rutvi/Downloads/gif.gif")

#total number of frames
length = int(gif.get(cv2.CAP_PROP_FRAME_COUNT))

# Set the position of the frame to be extracted
frame_index = 0
gif.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

# Read the current frame
ret, frame = gif.read()

# Loop over all the frames in the GIF file
for i in range(length):
    # current frame position
    gif.set(cv2.CAP_PROP_POS_FRAMES, i)
    
    # Reading the current frame
    ret, curr_frame = gif.read()
    
    # Computing the absolute difference between the extracted frame and the current frame
    diff = cv2.absdiff(frame, curr_frame)
    
    # Computing a metric to quantify the difference
    metric = diff.sum() / diff.size
    
    cv2.imshow('Frame', curr_frame)
    cv2.imshow('Difference', diff)
    cv2.waitKey(0)
    
gif.release()
cv2.destroyAllWindows()


# ### using Pillow

# In[6]:


from PIL import Image, ImageSequence
import numpy as np
import matplotlib.pyplot as plt


gif_image = Image.open("C:/Users/rutvi/Downloads/gif.gif")

# Get the first frame
first_frame = gif_image.convert("RGB")

# Convert the first frame to numpy array
first_frame = np.array(first_frame)

# Initialize a counter
counter = 1
# Iterate over the frames
for frame in ImageSequence.Iterator(gif_image):
    # Convert the frame to RGB
    frame = frame.convert("RGB")
    # Convert the frame to numpy array
    frame = np.array(frame)
    # Compute the difference between the current frame and the first frame
    diff = np.abs(first_frame - frame)
    # Convert the difference array to an image
    diff_image = Image.fromarray(diff)
    # Show the difference image
    plt.imshow(diff_image)
    plt.title("Difference between frame 1 and "+str(counter))
    plt.show()
    counter +=1


# In[ ]:




