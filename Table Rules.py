#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
from math import sqrt
import extcolors
from PIL import Image
import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[2]:


img = cv2.imread("/root/mis/0.jpeg")


# In[3]:


def display(img):
    #plt.rcParams["figure.figsize"] = (100,100)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


# In[4]:


box_corners = [(542, 620), (542, 550), (1678, 514), (1678, 646)]


# In[24]:


def _square(x):
    return x * x

def cie94(L1_a1_b1, L2_a2_b2):
    """Calculate color difference by using CIE94 formulae
    
    See http://en.wikipedia.org/wiki/Color_difference or
    http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html.
    
    cie94(rgb2lab((255, 255, 255)), rgb2lab((0, 0, 0)))
    >>> 58.0
    cie94(rgb2lab(rgb(0xff0000)), rgb2lab(rgb('#ff0000')))
    >>> 0.0
    """
    
    L1, a1, b1 = L1_a1_b1
    L2, a2, b2 = L2_a2_b2

    C1 = sqrt(_square(a1) + _square(b1))
    C2 = sqrt(_square(a2) + _square(b2))
    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_H_square = _square(delta_a) + _square(delta_b) - _square(delta_C)
    return (sqrt(_square(delta_L)
            + _square(delta_C) / _square(1.0 + 0.045 * C1)
            + delta_H_square / _square(1.0 + 0.015 * C1)))


count = 0
out_triangles = []
prev = box_corners[0][0]
for i, j in zip(range(box_corners[0][0], box_corners[0][0] + int((box_corners[2][0]-box_corners[0][0])/3), 20), [box_corners[1][1]]*20):
        
        pts = np.array([box_corners[0], (i, j), [prev, j]], np.int32).reshape((-1, 1, 2))
        
        ## (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x,y,w,h = rect
        croped = img[y:y+h, x:x+w].copy()
        
        ## (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        ## (3) do bit-op
        dst = cv2.bitwise_and(croped, croped, mask=mask)

        ## (4) add the white background
        bg = np.ones_like(croped, np.uint8)*255
        cv2.bitwise_not(bg,bg, mask=mask)
        dst2 = bg+ dst
        out_triangles.append(dst2)
        prev = i
out_triangles.pop(0)
def extract_unq_colors(img):
    colors, pixel_count =  extcolors.extract_from_image(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    
    return colors

def compare_two_triangles(x, y):
    #res = list(set(extract_unq_colors(trg1))^set(extract_unq_colors(trg2)))
    x = extract_unq_colors(x)
    y = extract_unq_colors(y)
    x = list(zip(*x))[0]
    y = list(zip(*y))[0]
    x1 = [i for i in x if i not in y]
    y1 = [i for i in y if i not in x]
    sum_ = []
    for i, j in zip(x1, y1):
        print(i, j)
        sum_.append(int(cie94(cv2.cvtColor(np.uint8([[i]]), cv2.COLOR_BGR2Lab)[0][0], cv2.cvtColor(np.uint8([[j]]), cv2.COLOR_BGR2Lab)[0][0])))
    return x1, y1, sum(sum_)

def all_traingles(out_traingles):
    trgs = []
    
    for i in range(0, len(out_traingles)-1):
        try:
            print("********************************"+ str(i) +"--"+ str(i+1) + "********************************")
            l1, l2, s = compare_two_triangles(out_triangles[i], out_triangles[i+1])
            trgs.append((l1,l2,s,(i,i+1)))
            display(out_triangles[i])
            display(out_triangles[i+1])
            print("********************************"+ str(i) +"--"+ str(i+1) + "********************************")
        except:
            pass
    return trgs


# In[25]:


#l1, l2, s = compare_two_triangles(out_triangles[10], out_triangles[11])
a = all_traingles(out_triangles)


# In[22]:


a


# In[ ]:




