import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def image_spliter(arr):
	#read image
	img2 = cv2.imdecode(arr, -1)

	width, height, s = img2.shape

	#clear the color which does not belong to the number
	for la in range(3):
	    for i in range(width):
	        for j in range(height):
	            if img2[i, j, la] < 115:
	                img2.itemset((i, j, la),255)

	#calculate the number of pixel which is not white and ksvd
	img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	th3 = cv2.adaptiveThreshold(img2Gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	for a in range(1):
	    for _width in range(1, width - 1 ,1):
	        for _height in range(1, height - 1 ,1):
	            area = 0
	            for i in range(_width - 1, _width + 2, 1):
	                for j in range(_height - 1, _height + 2, 1):
	                    if i > width - 1 or j > height - 1 or (i == _width and j == _height):
	                        continue
	                    if th3[i, j] < 233:
	                        area += 1
	            if area < 4:
	                th3.itemset((_width,_height),255)

	#clear the noisy line
	for _width in range(3, width - 2 ,1):
	    for _height in range(3, height - 2 ,1):
	        center_color = th3[_width, _height]

	        ucolor = th3[_width, _height + 1]
	        dcolor = th3[_width, _height - 1]
	        lcolor = th3[_width - 1, _height]
	        rcolor = th3[_width + 1, _height]
	        ulcolor = th3[_width - 1, _height + 1]
	        urcolor = th3[_width + 1, _height - 1]
	        dlcolor = th3[_width - 1, _height - 1]
	        drcolor = th3[_width + 1, _height - 1]

	        u2color = th3[_width, _height + 2]
	        d2color = th3[_width, _height - 2]
	        l2color = th3[_width - 2, _height]
	        r2color = th3[_width + 2, _height]
	        ul2color = th3[_width - 2, _height + 2]
	        ur2color = th3[_width + 2, _height - 2]
	        dl2color = th3[_width - 2, _height - 2]
	        dr2color = th3[_width + 2, _height - 2]

	        u2l1color = th3[_width - 1, _height + 2]
	        u2r1color = th3[_width + 1, _height - 2]
	        d2l1color = th3[_width - 1, _height - 2]
	        d2r1color = th3[_width + 1, _height - 2]
	        u1l2color = th3[_width - 2, _height + 1]
	        u1r2color = th3[_width + 2, _height - 1]
	        d1l2color = th3[_width - 2, _height - 1]
	        d1r2color = th3[_width + 2, _height - 1]

	        flag = 0
	        if ucolor == center_color:
	            flag += 1
	        if dcolor == center_color:
	            flag += 1
	        if lcolor == center_color:
	            flag += 1
	        if rcolor == center_color:
	            flag += 1
	        if ulcolor == center_color:
	            flag += 1
	        if urcolor == center_color:
	            flag += 1
	        if dlcolor == center_color:
	            flag += 1
	        if drcolor == center_color:
	            flag += 1

	        if u2color == center_color:
	            flag += 1
	        if d2color == center_color:
	            flag += 1
	        if l2color == center_color:
	            flag += 1
	        if r2color == center_color:
	            flag += 1
	        if ul2color == center_color:
	            flag += 1
	        if ur2color == center_color:
	            flag += 1
	        if dl2color == center_color:
	            flag += 1
	        if dr2color == center_color:
	            flag += 1

	        if u2l1color == center_color:
	            flag += 1
	        if u2r1color == center_color:
	            flag += 1
	        if d2l1color == center_color:
	            flag += 1
	        if d2r1color == center_color:
	            flag += 1
	        if u1l2color == center_color:
	            flag += 1
	        if u1r2color == center_color:
	            flag += 1
	        if d1l2color == center_color:
	            flag += 1
	        if d1r2color == center_color:
	            flag += 1

	        if flag < 10:
	            th3.itemset((_width,_height),255)

	#clear the edge
	for _width in range(width):
	    for _height in range(height):
	        if _width >= width-3 or _width <= 2 or _height >= height-3 or _height <= 2:
	            th3.itemset((_width,_height),255)


	#black white exchange and found the edge of the number
	th3 = cv2.bitwise_not(th3)
	contours, hierarchy = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


	img2.fill(255)
	arr = []
	draw_img = cv2.drawContours(img2.copy(), contours, -1, (0, 0, 255), 1)
	for cnt in contours:
	    x,y,w,h = cv2.boundingRect(cnt)
	    arr.append([x,y,x+w,y+h])

	#find the big rectangle that surrounds the number
	arr = np.asarray(arr)
	xmin = np.min(arr[:,0])
	xmax = np.max(arr[:,2])
	ymin = np.min(arr[:,1])
	ymax = np.max(arr[:,3])

	#find each number's start x and y
	x1 = xmin-2
	x2 = 21+xmin
	x3 = 43+xmin
	x4 = 65+xmin
	x5 = 95+xmin
	if (x5 >= height) :
	    x5 = height
	y1 = ymin
	y2 = ymin+35

	#draw rectangle(can delete this area)
	cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,255,0),2)
	cv2.rectangle(draw_img,(x2,y1),(x3,y2),(0,255,0),2)
	cv2.rectangle(draw_img,(x3,y1),(x4,y2),(0,255,0),2)
	cv2.rectangle(draw_img,(x4,y1),(x5,y2),(0,255,0),2)

	#create four black background images for splitting the number
	ch1 = np.zeros((40, 30, 1), dtype = "uint8")
	ch2 = np.zeros((40, 30, 1), dtype = "uint8")
	ch3 = np.zeros((40, 30, 1), dtype = "uint8")
	ch4 = np.zeros((40, 30, 1), dtype = "uint8")

	if xmax-xmin <= 90:
	    for i in range(x1,x2-2,1):
	        for j in range(y1,y2-1,1):
	            ch1[j-y1,i-x1] = th3[j,i]
	    for i in range(x2+1,x3-2,1):
	        for j in range(y1,y2-1,1):
	            ch2[j-y1,i-x2-1] = th3[j,i]
	    for i in range(x3+2,x4,1):
	        for j in range(y1,y2-1,1):
	            ch3[j-y1,i-x3-2] = th3[j,i]
	    for i in range(x4+4,x5,1):
	        for j in range(y1,y2-1,1):
	            ch4[j-y1,i-x4-4] = th3[j,i]
	elif xmax-xmin <= 91:
	    #print('here')
	    for i in range(x1,x2,1):
	        for j in range(y1,y2-1,1):
	            ch1[j-y1,i-x1] = th3[j,i]
	    for i in range(x2+1,x3+2,1):
	        for j in range(y1,y2-1,1):
	            ch2[j-y1,i-x2-1] = th3[j,i]
	    for i in range(x3+3,x4+1,1):
	        for j in range(y1,y2-1,1):
	            ch3[j-y1,i-x3-3] = th3[j,i]
	    for i in range(x4+5,x5,1):
	        for j in range(y1,y2-1,1):
	            ch4[j-y1,i-x4-5] = th3[j,i]
	else:
	    #print('there')
	    for i in range(x1,x2-1,1):
	        for j in range(y1,y2-1,1):
	            ch1[j-y1,i-x1] = th3[j,i]
	    for i in range(x2+3,x3+3,1):
	        for j in range(y1,y2-1,1):
	            ch2[j-y1,i-x2-3] = th3[j,i]
	    for i in range(x3+6,x4+3,1):
	        for j in range(y1,y2-1,1):
	            ch3[j-y1,i-x3-6] = th3[j,i]
	    for i in range(x4+6,x5,1):
	        for j in range(y1,y2-1,1):
	            ch4[j-y1,i-x4-6] = th3[j,i]

	'''
	plt.subplot(241), plt.imshow(img)  # original
	plt.subplot(242), plt.imshow(th3)  # denoise
	plt.subplot(243), plt.imshow(draw_img)  # position of each number
	plt.subplot(245), plt.imshow(ch1)  # first number
	plt.subplot(246), plt.imshow(ch2)  # second number
	plt.subplot(247), plt.imshow(ch3)  # third number
	plt.subplot(248), plt.imshow(ch4)  # fourth number
	plt.show()
	'''

	return [ch1, ch2, ch3, ch4]

def image_spliter_bgr(arr):
	#read image
	img2 = arr

	width, height, s = img2.shape

	#clear the color which does not belong to the number
	for la in range(3):
	    for i in range(width):
	        for j in range(height):
	            if img2[i, j, la] < 115:
	                img2.itemset((i, j, la),255)

	#calculate the number of pixel which is not white and ksvd
	img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	th3 = cv2.adaptiveThreshold(img2Gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	for a in range(1):
	    for _width in range(1, width - 1 ,1):
	        for _height in range(1, height - 1 ,1):
	            area = 0
	            for i in range(_width - 1, _width + 2, 1):
	                for j in range(_height - 1, _height + 2, 1):
	                    if i > width - 1 or j > height - 1 or (i == _width and j == _height):
	                        continue
	                    if th3[i, j] < 233:
	                        area += 1
	            if area < 4:
	                th3.itemset((_width,_height),255)

	#clear the noisy line
	for _width in range(3, width - 2 ,1):
	    for _height in range(3, height - 2 ,1):
	        center_color = th3[_width, _height]

	        ucolor = th3[_width, _height + 1]
	        dcolor = th3[_width, _height - 1]
	        lcolor = th3[_width - 1, _height]
	        rcolor = th3[_width + 1, _height]
	        ulcolor = th3[_width - 1, _height + 1]
	        urcolor = th3[_width + 1, _height - 1]
	        dlcolor = th3[_width - 1, _height - 1]
	        drcolor = th3[_width + 1, _height - 1]

	        u2color = th3[_width, _height + 2]
	        d2color = th3[_width, _height - 2]
	        l2color = th3[_width - 2, _height]
	        r2color = th3[_width + 2, _height]
	        ul2color = th3[_width - 2, _height + 2]
	        ur2color = th3[_width + 2, _height - 2]
	        dl2color = th3[_width - 2, _height - 2]
	        dr2color = th3[_width + 2, _height - 2]

	        u2l1color = th3[_width - 1, _height + 2]
	        u2r1color = th3[_width + 1, _height - 2]
	        d2l1color = th3[_width - 1, _height - 2]
	        d2r1color = th3[_width + 1, _height - 2]
	        u1l2color = th3[_width - 2, _height + 1]
	        u1r2color = th3[_width + 2, _height - 1]
	        d1l2color = th3[_width - 2, _height - 1]
	        d1r2color = th3[_width + 2, _height - 1]

	        flag = 0
	        if ucolor == center_color:
	            flag += 1
	        if dcolor == center_color:
	            flag += 1
	        if lcolor == center_color:
	            flag += 1
	        if rcolor == center_color:
	            flag += 1
	        if ulcolor == center_color:
	            flag += 1
	        if urcolor == center_color:
	            flag += 1
	        if dlcolor == center_color:
	            flag += 1
	        if drcolor == center_color:
	            flag += 1

	        if u2color == center_color:
	            flag += 1
	        if d2color == center_color:
	            flag += 1
	        if l2color == center_color:
	            flag += 1
	        if r2color == center_color:
	            flag += 1
	        if ul2color == center_color:
	            flag += 1
	        if ur2color == center_color:
	            flag += 1
	        if dl2color == center_color:
	            flag += 1
	        if dr2color == center_color:
	            flag += 1

	        if u2l1color == center_color:
	            flag += 1
	        if u2r1color == center_color:
	            flag += 1
	        if d2l1color == center_color:
	            flag += 1
	        if d2r1color == center_color:
	            flag += 1
	        if u1l2color == center_color:
	            flag += 1
	        if u1r2color == center_color:
	            flag += 1
	        if d1l2color == center_color:
	            flag += 1
	        if d1r2color == center_color:
	            flag += 1

	        if flag < 10:
	            th3.itemset((_width,_height),255)

	#clear the edge
	for _width in range(width):
	    for _height in range(height):
	        if _width >= width-3 or _width <= 2 or _height >= height-3 or _height <= 2:
	            th3.itemset((_width,_height),255)


	#black white exchange and found the edge of the number
	th3 = cv2.bitwise_not(th3)
	contours, hierarchy = cv2.findContours(th3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


	img2.fill(255)
	arr = []
	draw_img = cv2.drawContours(img2.copy(), contours, -1, (0, 0, 255), 1)
	for cnt in contours:
	    x,y,w,h = cv2.boundingRect(cnt)
	    arr.append([x,y,x+w,y+h])

	#find the big rectangle that surrounds the number
	arr = np.asarray(arr)
	xmin = np.min(arr[:,0])
	xmax = np.max(arr[:,2])
	ymin = np.min(arr[:,1])
	ymax = np.max(arr[:,3])

	#find each number's start x and y
	x1 = xmin-2
	x2 = 21+xmin
	x3 = 43+xmin
	x4 = 65+xmin
	x5 = 95+xmin
	if (x5 >= height) :
	    x5 = height
	y1 = ymin
	y2 = ymin+35

	#draw rectangle(can delete this area)
	cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,255,0),2)
	cv2.rectangle(draw_img,(x2,y1),(x3,y2),(0,255,0),2)
	cv2.rectangle(draw_img,(x3,y1),(x4,y2),(0,255,0),2)
	cv2.rectangle(draw_img,(x4,y1),(x5,y2),(0,255,0),2)

	#create four black background images for splitting the number
	ch1 = np.zeros((40, 30, 1), dtype = "uint8")
	ch2 = np.zeros((40, 30, 1), dtype = "uint8")
	ch3 = np.zeros((40, 30, 1), dtype = "uint8")
	ch4 = np.zeros((40, 30, 1), dtype = "uint8")

	if xmax-xmin <= 90:
	    for i in range(x1,x2-2,1):
	        for j in range(y1,y2-1,1):
	            ch1[j-y1,i-x1] = th3[j,i]
	    for i in range(x2+1,x3-2,1):
	        for j in range(y1,y2-1,1):
	            ch2[j-y1,i-x2-1] = th3[j,i]
	    for i in range(x3+2,x4,1):
	        for j in range(y1,y2-1,1):
	            ch3[j-y1,i-x3-2] = th3[j,i]
	    for i in range(x4+4,x5,1):
	        for j in range(y1,y2-1,1):
	            ch4[j-y1,i-x4-4] = th3[j,i]
	elif xmax-xmin <= 91:
	    #print('here')
	    for i in range(x1,x2,1):
	        for j in range(y1,y2-1,1):
	            ch1[j-y1,i-x1] = th3[j,i]
	    for i in range(x2+1,x3+2,1):
	        for j in range(y1,y2-1,1):
	            ch2[j-y1,i-x2-1] = th3[j,i]
	    for i in range(x3+3,x4+1,1):
	        for j in range(y1,y2-1,1):
	            ch3[j-y1,i-x3-3] = th3[j,i]
	    for i in range(x4+5,x5,1):
	        for j in range(y1,y2-1,1):
	            ch4[j-y1,i-x4-5] = th3[j,i]
	else:
	    #print('there')
	    for i in range(x1,x2-1,1):
	        for j in range(y1,y2-1,1):
	            ch1[j-y1,i-x1] = th3[j,i]
	    for i in range(x2+3,x3+3,1):
	        for j in range(y1,y2-1,1):
	            ch2[j-y1,i-x2-3] = th3[j,i]
	    for i in range(x3+6,x4+3,1):
	        for j in range(y1,y2-1,1):
	            ch3[j-y1,i-x3-6] = th3[j,i]
	    for i in range(x4+6,x5,1):
	        for j in range(y1,y2-1,1):
	            ch4[j-y1,i-x4-6] = th3[j,i]

	'''
	plt.subplot(241), plt.imshow(img)  # original
	plt.subplot(242), plt.imshow(th3)  # denoise
	plt.subplot(243), plt.imshow(draw_img)  # position of each number
	plt.subplot(245), plt.imshow(ch1)  # first number
	plt.subplot(246), plt.imshow(ch2)  # second number
	plt.subplot(247), plt.imshow(ch3)  # third number
	plt.subplot(248), plt.imshow(ch4)  # fourth number
	plt.show()
	'''

	return [ch1, ch2, ch3, ch4]

def image_to_file(dir, img_arr):
	img_arr = img_arr.reshape(img_arr.shape[0], img_arr.shape[1])
	img = Image.fromarray(img_arr)
	img.save(dir)

if __name__ == '__main__':
	nc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	dir_in = 'E3'
	dir_out = 'E3/split'
	ic = 0
	for filename in os.listdir(dir_in + '/img'):
		ic += 1
		print('Task ' + str(ic) + ': ' + filename)
		char_4 = filename[0:4]
		imgs = image_spliter(dir_in + '/img/' + filename)
		for i in range(4):
			image_to_file(dir_out + '/' + char_4[i] + '/' + char_4[i] + '_' + str(nc[int(char_4[i])]) + '.png', imgs[i])
			nc[int(char_4[i])] += 1
