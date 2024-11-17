import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary

def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	# Assuming you have created a warped binary image called "binary_warped"
	# Take a histogram of the bottom half of the image
	histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Find the peak of the left and right halves of the histogram
	# These will be the starting point for the left and right lines
	midpoint = int(histogram.shape[0]/2)
	leftx_base = np.argmax(histogram[100:midpoint]) + 100
	rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

	# Choose the number of sliding windows
	nwindows = 9
	# Set height of windows
	window_height = int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of the windows +/- margin
	margin = 100
	# Set minimum number of pixels found to recenter window
	minpix = 150
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []

	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		##TO DO

		### Detect direction: Upwards
		y_top = binary_warped.shape[0] - window * window_height
		y_bottom = binary_warped.shape[0] - (window + 1) * window_height

		# ----- Left Half
		leftX_L = leftx_current - margin
		leftX_R = leftx_current + margin

		# ----- Right Half
		rightX_L = rightx_current - margin
		rightX_R = rightx_current + margin

		####
		# Draw the windows on the visualization image using cv2.rectangle()
		##TO DO

		# input: (Output Image, top left corner XY, bottom right corner XY, color, thickness)
		# ----- Left Half
		cv2.rectangle(out_img,(leftX_L,y_top), (leftX_R, y_bottom), (0,255,0), 3)
		# ----- Right Half
		cv2.rectangle(out_img,(rightX_L,y_top), (rightX_R, y_bottom), (255,0,0), 3)
		####
		# Identify the nonzero pixels in x and y within the window
		##TO DO

		# ----- Left Half
		nonzeroL = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) & (nonzerox >= leftX_L) & (nonzerox < leftX_R)).nonzero()[0]
		# ----- Right Half
		nonzeroR = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) & (nonzerox >= rightX_L) & (nonzerox < rightX_R)).nonzero()[0]
		####
		# Append these indices to the lists
		##TO DO
		
		# ----- Left Half

		###### left_lane_inds = left_lane_inds.append(nonzeroL)
		left_lane_inds.append(nonzeroL)

		# ----- Right Half

		###### right_lane_inds = right_lane_inds.append(nonzeroR)
		right_lane_inds.append(nonzeroR)

		####
		# If you found > minpix pixels, recenter next window on their mean position
		##TO DO
		if len(nonzeroL) > minpix:

			###### leftx_current = int(np.mean(nonzerox[left_lane_inds]))
			leftx_current = int(np.mean(nonzerox[nonzeroL]))

		if len(nonzeroR) > minpix:

			###### rightx_current = int(np.mean(nonzerox[right_lane_inds]))
			rightx_current = int(np.mean(nonzerox[nonzeroR]))

		####
		pass

	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# Fit a second order polynomial to each using np.polyfit()
	# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
	# the second order polynomial is unable to be sovled.
	# Thus, it is unable to detect edges.
	try:
	##TODO
		# left_fit = np.polyfit(leftx, lefty, 2) # (x,y,deg)
		# right_fit = np.polyfit(rightx, righty, 2)
		left_fit = np.polyfit(lefty, leftx, 2) # (x,y,deg)
		right_fit = np.polyfit(righty, rightx, 2)

	####
	except TypeError:
		print("Unable to detect lanes")
		return None


	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['out_img'] = out_img
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def tune_fit(binary_warped, left_fit, right_fit):
	"""
	Given a previously fit line, quickly try to find the line based on previous lines
	"""
	# Assume you now have a new warped binary image
	# from the next frame of video (also called "binary_warped")
	# It's now much easier to find line pixels!
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	margin = 100
	left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
	right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

	# Again, extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds]
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds]

	# If we don't find enough relevant points, return all None (this means error)
	min_inds = 10
	if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
		return None

	# Fit a second order polynomial to each
	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Return a dict of relevant variables
	ret = {}
	ret['left_fit'] = left_fit
	ret['right_fit'] = right_fit
	ret['nonzerox'] = nonzerox
	ret['nonzeroy'] = nonzeroy
	ret['left_lane_inds'] = left_lane_inds
	ret['right_lane_inds'] = right_lane_inds

	return ret


def viz1(binary_warped, ret, save_file=None):
	"""
	Visualize each sliding window location and predicted lane lines, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	out_img = ret['out_img']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	plt.imshow(out_img)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)
	if save_file is None:
		plt.show()
	else:
		plt.savefig(save_file)
	plt.gcf().clear()


def bird_fit(binary_warped, ret, save_file=None):
	"""
	Visualize the predicted lane lines with margin, on binary warped image
	save_file is a string representing where to save the image (if None, then just display)
	"""
	# Grab variables from ret dictionary
	left_fit = ret['left_fit']
	right_fit = ret['right_fit']
	nonzerox = ret['nonzerox']
	nonzeroy = ret['nonzeroy']
	left_lane_inds = ret['left_lane_inds']
	right_lane_inds = ret['right_lane_inds']

	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	margin = 100  # NOTE: Keep this in sync with *_fit()
	left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
	left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
	left_line_pts = np.hstack((left_line_window1, left_line_window2))
	right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
	right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
	right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	# cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 255))
	# cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 255))
######
	middle_fitx = (left_fitx + right_fitx) / 2

	# 將中間線的 x 和 y 坐標合併為一個 Nx2 的陣列
	pts_middle = np.array([np.transpose(np.vstack([middle_fitx, ploty]))], dtype=np.int32)

	# Draw the lane onto the warped blank image
	cv2.polylines(window_img, [pts_middle], isClosed=False, color=(255, 0, 0), thickness=15)
	waypoint_idx = [0,int(len(left_fitx)/2),len(left_fitx)-1]
	#print(f'waypoint={waypoint_idx}')
	waypoint=[]
	for i in waypoint_idx:
		cv2.circle(window_img, (int(middle_fitx[i]), int(ploty[i])), 25, (0, 0, 255), -1)
		waypoint.append((int(middle_fitx[i]), int(ploty[i])))
	#print(f'waypoint={waypoint}')
######
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	plt.imshow(result)
	plt.plot(left_fitx, ploty, color='yellow')
	plt.plot(right_fitx, ploty, color='yellow')
	plt.xlim(0, 1280)
	plt.ylim(720, 0)

########
	# cv2.imshow('bird',result)
	# cv2.imwrite('bird_from_cv2.png', result)

	# if save_file is None:
	# 	plt.show()
	# else:
	# 	plt.savefig(save_file)
	# plt.gcf().clear()
########





	return result


def final_viz(undist, left_fit, right_fit, m_inv):
	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((720, 1280, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))





	# Draw the lane onto the warped blank image
	# cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

##########
	middle_fitx = (left_fitx + right_fitx) / 2

	# 將中間線的 x 和 y 坐標合併為一個 Nx2 的陣列
	pts_middle = np.array([np.transpose(np.vstack([middle_fitx, ploty]))], dtype=np.int32)

	# 假設 'image' 是目標圖像，繪製中間線
	#cv2.polylines(color_warp, [pts_middle], isClosed=False, color=(255, 0, 0), thickness=5)  # 使用藍色繪製中間線

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	cv2.polylines(color_warp, [pts_middle], isClosed=False, color=(255, 0, 0), thickness=15)
	waypoint_idx = [0,int(len(left_fitx)/2),len(left_fitx)-1]
	#print(f'waypoint={waypoint_idx}')
	waypoint=[]
	for i in waypoint_idx:
		#cv2.circle(color_warp, (int(middle_fitx[i]), int(middle_fitx[i])), 25, (0, 0, 255), -1)
		waypoint.append((int(middle_fitx[i]), int(ploty[i])))
	# 	text = f"({int(middle_fitx[i])}, {int(middle_fitx[i])})"  # 定義要顯示的文字
	# 	font = cv2.FONT_HERSHEY_SIMPLEX  # 字體
	# 	font_scale = 0.6  # 字體大小
	# 	color = (0, 0, 0)  # 白色字體
	# 	thickness = 2  # 字體厚度
    
	# 	# 在圖像上指定位置寫文字（可以調整位置，避免與圓點重疊）
	# 	cv2.putText(color_warp, text, (int(middle_fitx[i]) + 10, int(ploty[i]) - 10), font, font_scale, color, thickness)
	print(f'waypoint={waypoint}')
	#transfer waypoint from birdview to real image
	waypoints_transformed = [] 



	# # Generate waypoints from right lane
	# def get_arc_points(right_fitx, ploty, offset=100):
	# 	# 先端、中央、先端の3点を取得
	# 	waypoint_idx = [0, int(len(right_fitx) / 2), len(right_fitx) - 1]
	# 	waypoints = [(right_fitx[i], ploty[i]) for i in waypoint_idx]

	# 	# 円弧を計算
	# 	arc_points = []
	# 	for i in range(len(waypoints) - 1):
	# 		start_point = waypoints[i]
	# 		end_point = waypoints[i + 1]
	# 		num_points = 100  # 円弧上の点の数
	# 		for t in np.linspace(0, 1, num_points):
	# 			x = (1 - t) * start_point[0] + t * end_point[0]
	# 			y = (1 - t) * start_point[1] + t * end_point[1]
	# 			arc_points.append((x, y))

	# 	# 左側にオフセット
	# 	offset_arc_points = [(x - offset, y) for (x, y) in arc_points]

	# 	return offset_arc_points

	# def draw_offset_arc(image, right_fitx, ploty, offset=100):
	# 	offset_arc_points = get_arc_points(right_fitx, ploty, offset)
	# 	pts = np.array(offset_arc_points, dtype=np.int32)
	# 	cv2.polylines(image, [pts], isClosed=False, color=(0, 0, 0), thickness=10)
	# 	return pts

	# # Road width: 3.1m, 960px => Half width: 1.55m, 480px
	# pts = draw_offset_arc(color_warp, right_fitx, ploty, offset=480)
	# waypoint=[]
	# waypoint_idx = [0,int(len(pts)/2),len(pts)-1]
	# for i in waypoint_idx:
	# 	waypoint.append((int(pts[i][0]), int(pts[i][1])))
	# print(f'waypoint_black={waypoint}')
	def get_arc_points(right_fitx, ploty, offset):	
		waypoint_idx = np.linspace(0, len(right_fitx)-1, 50, dtype=int)
		waypoints = [(right_fitx[i], ploty[i]) for i in waypoint_idx]

		offset_arc_points = []
		for i in range(1, len(waypoints) - 1):
			prev_point = np.array(waypoints[i - 1])
			curr_point = np.array(waypoints[i])
			next_point = np.array(waypoints[i + 1])

			# 前後の点から法線ベクトルを計算
			tangent = next_point - prev_point
			normal = np.array([-tangent[1], tangent[0]])
			normal = normal / np.linalg.norm(normal)  # 正規化

			# 法線方向にオフセットを適用
			# offset_point = curr_point + offset * normal
			#   x: 1280 px -> 4.1 m
			#   y: 720 px -> 10.3 m
			# 720/10.3*4.1/1280 = 0.2239
			offset_point = curr_point + [offset * normal[0], offset * normal[1] * 0.2239]
			offset_arc_points.append(tuple(offset_point))

		return offset_arc_points

	def draw_offset_arc(image, right_fitx, ploty, offset):
		offset_arc_points = get_arc_points(right_fitx, ploty, offset)
		pts = np.array(offset_arc_points, dtype=np.int32)
		# cv2.polylines(image, [pts], isClosed=False, color=(0, 0, 0), thickness=20)
		return pts

	# Gazebo --- Road width: 3.1m, 460px => Half width: 1.55m, 230px
	# GEM --- Road width: 3.1m, 960px => Half width: 1.55m, 480px

	offset = 480 # for right lane: left offset from original lane
	if right_fitx[-1] < 1280/2: # the closest point of lane is on the left side
		offset = -offset # for left lane: right offset from original lane

	pts = draw_offset_arc(color_warp, right_fitx, ploty, offset)
	waypoint=[]
	# waypoint_idx = [0,int(len(pts)/2),len(pts)-1]
	waypoint_idx = np.linspace(0, len(pts)-1, 5, dtype=int)
	for i in waypoint_idx:
		waypoint.append((int(pts[i][0]), int(pts[i][1])))
	# print(f'waypoint_black={waypoint}')


	# Conversion from px to m
	#   x: 1280 px -> 4.1 m
	#   y: 720 px -> 10.3 m
	# id=0: farest point from gem
	# id=1: middle point from gem
	# id=2: nearest point from gem
	waypoints_m = [] 
	for (x, y) in waypoint:
		x_m = x * 4.1 / 1280
		y_m = y * 10.3 / 720
		waypoints_m.append((float(x_m), float(y_m)))

	# id=3: gem position
	waypoints_m.append((4.1/2, 10.3 + 3.46))



	for (x, y) in waypoint:
		# 將 (x, y) 座標轉換為齊次座標 (x, y, 1)
		point_homogeneous = np.array([x, y, 1], dtype=np.float32).reshape(3, 1)
		
		# 應用逆透視矩陣變換
		point_transformed = np.dot(m_inv, point_homogeneous)
		
		# 轉換回正常的 (x, y) 座標，通過齊次座標除以第三個元素
		x_transformed = int(point_transformed[0] / point_transformed[2])
		y_transformed = int(point_transformed[1] / point_transformed[2])
		
		# 將轉換後的座標加入新的 waypoints 列表
		waypoints_transformed.append((x_transformed, y_transformed))
	# print(f'waypoint_trans={waypoints_transformed}')

##########


	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))
	# Combine the result with the original image
	# Convert arrays to 8 bit for later cv to ros image transfer


##########
	#print waypoint on map
	for (x,y) in waypoints_transformed:
		cv2.circle(newwarp, (x,y), 5, (0, 0, 255), -1)
		text = f"({x}, {y})"  # 定義要顯示的文字
		font = cv2.FONT_HERSHEY_SIMPLEX  # 字體
		font_scale = 1  # 字體大小
		color = (255, 0, 0)  # 白色字體
		thickness = 2  # 字體厚度
    
		# 在圖像上指定位置寫文字（可以調整位置，避免與圓點重疊）
		cv2.putText(newwarp, text, (x + 10, y - 10), font, font_scale, color, thickness)
###########


	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	# return result
	return result, waypoints_m
