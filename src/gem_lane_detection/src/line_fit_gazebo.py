import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from combined_thresh import combined_thresh
# from perspective_transform import perspective_transform

# feel free to adjust the parameters in the code if necessary


# GEM e2 in Gazebo
image_w   = 640 # px
image_h   = 480  # px
lane_w_px = 460  # px
lane_h_px = 480  # px
lane_w_m  = 3.1  # m
lane_h_m  = 5.0 # m
# lane_h_m  = 10.3 # m
camera_to_rearwheel = 3.46 # m


def find_lane_startP(binary_warped, threshold=50):
	selected_area_top = 6*binary_warped.shape[0]//7
	selected_area_bottom = 9*binary_warped.shape[0]//10
	histogram_bottom = np.sum(binary_warped[selected_area_top:selected_area_bottom,:], axis=0)
	midpoint = int(histogram_bottom.shape[0]/2)

	# Get left and right histograms
	# left_histogram = histogram_bottom[100:midpoint]
	# right_histogram = histogram_bottom[midpoint:-100]
	left_histogram = histogram_bottom[10:midpoint]
	right_histogram = histogram_bottom[midpoint:-10]
	
	# Compute np.argmax values
	left_argmax = np.argmax(left_histogram)
	right_argmax = np.argmax(right_histogram)
	
	# Get actual maximum values
	left_max_value = left_histogram[left_argmax]
	right_max_value = right_histogram[right_argmax]
	
	# Apply thresholds
	if left_max_value >= threshold:
		# left_lane_startP = left_argmax + 100
		left_lane_startP = left_argmax + 10
	else:
		left_lane_startP = None  # or handle as needed (e.g., skip detection)

	if right_max_value >= threshold:
		right_lane_startP = right_argmax + midpoint
	else:
		right_lane_startP = None  # or handle as needed

	# # Visualize the results
	# plt.figure(figsize=(10, 6))
	# plt.imshow(binary_warped, cmap='gray')  # Display the binary image
	
	# # Draw vertical lines for left and right lane start positions
	# plt.axvline(x=left_lane_start, color='red', linestyle='--', label='Left Lane Start')
	# plt.axvline(x=right_lane_start, color='blue', linestyle='--', label='Right Lane Start')
	
	# # Add legend and title
	# plt.legend()
	# plt.title("Lane Start Points")
	# plt.show()
	return left_lane_startP, right_lane_startP



def line_fit(binary_warped):
	"""
	Find and fit lane lines
	"""
	##### ================ Replace by function "find_lane_startP" ================ #####
	# # Assuming you have created a warped binary image called "binary_warped"
	# # Take a histogram of the bottom half of the image
	# histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
	# # Find the peak of the left and right halves of the histogram
	# # These will be the starting point for the left and right lines
	# midpoint = int(histogram.shape[0]/2)
	# leftx_base = np.argmax(histogram[100:midpoint]) + 100
	# rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint
	##### ====================================================================== #####
	
	# ===== [1] Check the histogram of the bottom of frame =====
 	# If leftx_base = None >> Left lane not detected
	# If rightx_base = None >> Right lane not detected
	leftx_base, rightx_base = find_lane_startP(binary_warped, threshold=15)

	# ===== [2] Basic Parameter Setup (Provided from Original Code) ===== 
	# Create an output image to draw on and visualize the result
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	# Choose the number of sliding windows
	# nwindows = 9
	nwindows = 19
	# Set height of windows
	window_height = int(binary_warped.shape[0]/nwindows)
	# Identify the x and y positions of all nonzero pixels in the image
	nonzero = binary_warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Set the width of the windows +/- margin
	# margin = 100
	margin = 50
	# Set minimum number of pixels found to recenter window
	minpix = 50
	
	# Current positions to be updated for each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Create empty lists to receive left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []


	##### Add: Eliminate the aparted lanes #####
	lefty_current  = image_h
	righty_current = image_h
	############################################


	# ===== [3] Determine which lane(s) is/are detected at bottom: (Both/Left/Right/Nothing) =====
	lane_case = "Nothing"
	if leftx_base is not None and rightx_base is not None:
		lane_case = "Both"
	elif leftx_base is not None and rightx_base is None:
		lane_case = "Left"
	elif leftx_base is None and rightx_base is not None:
		lane_case = "Right"
	else:
		lane_case = "Nothing"
	print(lane_case)
	# --------------- [3.1] Both Lane exits at bottom ---------------
	if lane_case == "Both":
		# 'Step through the windows one by one
		for window in range(nwindows):

			# Identify window boundaries in x and y (and right and left)
			### Detect direction: Upwards
			y_top = binary_warped.shape[0] - window * window_height
			y_bottom = binary_warped.shape[0] - (window + 1) * window_height
			# if window == 0:
			# 	continue
			# 	# y_top = y_top - window_height//10

			leftX_L = leftx_current - margin			# ----- Left Half
			leftX_R = leftx_current + margin
			rightX_L = rightx_current - margin			# ----- Right Half
			rightX_R = rightx_current + margin
			####
			# Draw the windows on the visualization image using cv2.rectangle()
			cv2.rectangle(out_img,(leftX_L,y_top), (leftX_R, y_bottom), (0,255,0), 3)		# ----- Left Half
			cv2.rectangle(out_img,(rightX_L,y_top), (rightX_R, y_bottom), (255,0,0), 3)		# ----- Right Half
			####
			# Identify the nonzero pixels in x and y within the window
			nonzeroL = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) & (nonzerox >= leftX_L) & (nonzerox < leftX_R)).nonzero()[0] #--- Left Half
			nonzeroR = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) & (nonzerox >= rightX_L) & (nonzerox < rightX_R)).nonzero()[0] #--- Right Half
			####


			# # Append these indices to the lists
			# left_lane_inds.append(nonzeroL)			# ----- Left Half
			# right_lane_inds.append(nonzeroR)		# ----- Right Half
			# ####
			# # If you found > minpix pixels, recenter next window on their mean 
			# if len(nonzeroL) > minpix:
			# 	leftx_current = int(np.mean(nonzerox[nonzeroL]))
			# if len(nonzeroR) > minpix:
			# 	rightx_current = int(np.mean(nonzerox[nonzeroR]))


			########## Add: Eliminate the aparted lanes ##########
			if len(nonzeroR) > 0:
				if abs(rightx_current - int(np.mean(nonzerox[nonzeroR])) ) > 200 or abs(righty_current - int(np.mean(nonzeroy[nonzeroR])) ) > 200:
					nonzeroR = []
				else:
					# Append these indices to the lists
					right_lane_inds.append(nonzeroR)
					# If you found > minpix pixels, recenter next window on their mean 
					if len(nonzeroR) > minpix:
						rightx_current = int(np.mean(nonzerox[nonzeroR]))
						righty_current = int(np.mean(nonzeroy[nonzeroR]))
			if len(nonzeroL) > 0:
				if abs(leftx_current - int(np.mean(nonzerox[nonzeroL])) ) > 200 or abs(lefty_current - int(np.mean(nonzeroy[nonzeroL])) ) > 200:
					nonzeroL = []
				else:
					# Append these indices to the lists
					left_lane_inds.append(nonzeroL)
					# If you found > minpix pixels, recenter next window on their mean 
					if len(nonzeroL) > minpix:
						leftx_current = int(np.mean(nonzerox[nonzeroL]))
						lefty_current = int(np.mean(nonzeroy[nonzeroL]))
			######################################################

		
			####
	# --------------- [3.2] Only Left Lane exits at bottom---------------
	elif lane_case == "Left":
		# Since only one lane is in frame, we will detect the whole frame without splitting.
		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y
			### Detect direction: Upwards
			y_top = binary_warped.shape[0] - window * window_height
			y_bottom = binary_warped.shape[0] - (window + 1) * window_height

			# if window == 0:
			# 	continue
			# 	# y_top = y_top - window_height//10

			leftX_L = leftx_current - margin*5			# ----- Only Left Lane needed
			leftX_R = leftx_current + margin*5
			####
			# Draw the windows on the visualization image using cv2.rectangle()
			cv2.rectangle(out_img,(leftX_L,y_top), (leftX_R, y_bottom), (0,255,0), 3)
			####
			# Identify the nonzero pixels in x and y within the window
			nonzeroL = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) & (nonzerox >= leftX_L) & (nonzerox < leftX_R)).nonzero()[0] 
			####


			# # Append these indices to the lists
			# left_lane_inds.append(nonzeroL)	
			# # If you found > minpix pixels, recenter next window on their mean 
			# if len(nonzeroL) > minpix:
			# 	leftx_current = int(np.mean(nonzerox[nonzeroL]))


			########## Add: Eliminate the aparted lanes ##########
			if len(nonzeroL) > 0:
				if abs(leftx_current - int(np.mean(nonzerox[nonzeroL])) ) > 200 or abs(lefty_current - int(np.mean(nonzeroy[nonzeroL])) ) > 200:
					nonzeroL = []
				else:
					# Append these indices to the lists
					left_lane_inds.append(nonzeroL)
					# If you found > minpix pixels, recenter next window on their mean 
					if len(nonzeroL) > minpix:
						leftx_current = int(np.mean(nonzerox[nonzeroL]))
						lefty_current = int(np.mean(nonzeroy[nonzeroL]))
			######################################################


			####
		
	# --------------- [3.3] Only Right Lane exits at bottom ---------------
	elif lane_case == "Right":
		# Similar thoughts of [3.2]
		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y
			### Detect direction: Upwards
			y_top = binary_warped.shape[0] - window * window_height
			y_bottom = binary_warped.shape[0] - (window + 1) * window_height

			# if window == 0:
			# 	continue
			# 	# y_top = y_top - window_height//10

			rightX_L = rightx_current - margin*5			# ----- Only Right Lane
			rightX_R = rightx_current + margin*5
			####
			# Draw the windows on the visualization image using cv2.rectangle()
			cv2.rectangle(out_img,(rightX_L,y_top), (rightX_R, y_bottom), (255,0,0), 3)	
			####
			# Identify the nonzero pixels in x and y within the window
			nonzeroR = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) & (nonzerox >= rightX_L) & (nonzerox < rightX_R)).nonzero()[0]
			####

			# # Append these indices to the lists
			# right_lane_inds.append(nonzeroR)
			# # If you found > minpix pixels, recenter next window on their mean 
			# if len(nonzeroR) > minpix:
			# 	rightx_current = int(np.mean(nonzerox[nonzeroR]))


			########## Add: Eliminate the aparted lanes ##########
			if len(nonzeroR) > 0:
				if abs(rightx_current - int(np.mean(nonzerox[nonzeroR])) ) > 200 or abs(righty_current - int(np.mean(nonzeroy[nonzeroR])) ) > 200:
					nonzeroR = []
				else:
					# Append these indices to the lists
					right_lane_inds.append(nonzeroR)
					# If you found > minpix pixels, recenter next window on their mean 
					if len(nonzeroR) > minpix:
						rightx_current = int(np.mean(nonzerox[nonzeroR]))
						righty_current = int(np.mean(nonzeroy[nonzeroR]))
			######################################################


			####
	
	# --------------- [3.4] No Lane exits at bottom ---------------
	else:
		#TODO: This is just a copy of two lanes --> Needed to be fixed!
		# Identify window boundaries in x and y (and right and left)
		### Detect direction: Upwards
		# y_top = binary_warped.shape[0] - window * window_height
		# y_bottom = binary_warped.shape[0] - (window + 1) * window_height

		# if window == 0:
		# 	y_top = y_top - window_height//10

		# leftX_L = leftx_current - margin			# ----- Left Half
		# leftX_R = leftx_current + margin
		# rightX_L = rightx_current - margin			# ----- Right Half
		# rightX_R = rightx_current + margin
		# ####
		# # Draw the windows on the visualization image using cv2.rectangle()
		# cv2.rectangle(out_img,(leftX_L,y_top), (leftX_R, y_bottom), (0,255,0), 3)			# ----- Left Half
		# cv2.rectangle(out_img,(rightX_L,y_top), (rightX_R, y_bottom), (255,0,0), 3)			# ----- Right Half
		# ####
		# # Identify the nonzero pixels in x and y within the window
		# nonzeroL = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) & (nonzerox >= leftX_L) & (nonzerox < leftX_R)).nonzero()[0] #--- Left Half
		# nonzeroR = ((nonzeroy >= y_bottom) & (nonzeroy < y_top) & (nonzerox >= rightX_L) & (nonzerox < rightX_R)).nonzero()[0] #--- Right Half
		# ####
		# # Append these indices to the lists
		# left_lane_inds.append(nonzeroL)			# ----- Left Half
		# right_lane_inds.append(nonzeroR)		# ----- Right Half
		# ####
		# # If you found > minpix pixels, recenter next window on their mean 
		# if len(nonzeroL) > minpix:
		# 	leftx_current = int(np.mean(nonzerox[nonzeroL]))
		# if len(nonzeroR) > minpix:
		# 	rightx_current = int(np.mean(nonzerox[nonzeroR]))
		####
		pass

	# ===== [4. Detect Lane Points] ===== 
	left_fit = None
	right_fit = None
	if lane_case == "Both":
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
			left_fit = np.polyfit(lefty, leftx, 2)
			right_fit = np.polyfit(righty, rightx, 2)
		####
		except TypeError:
			print("Unable to detect lanes - BOTH")
			return None
	elif lane_case == "Left":
		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]

		# Fit a second order polynomial to each using np.polyfit()
		# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
		# the second order polynomial is unable to be sovled.
		# Thus, it is unable to detect edges.
		try:
			left_fit = np.polyfit(lefty, leftx, 2)
		####
		except TypeError:
			print("Unable to detect lanes - Left")
			return None
	elif lane_case == "Right":
		# Concatenate the arrays of indices
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		# Fit a second order polynomial to each using np.polyfit()
		# If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
		# the second order polynomial is unable to be sovled.
		# Thus, it is unable to detect edges.
		try:
			right_fit = np.polyfit(righty, rightx, 2)
		####
		except TypeError:
			print("Unable to detect lanes - Right")
			return None
	else:
		# 11/22
		# #TODO: This is just a copy of two lanes --> Needed to be fixed!
		# # Concatenate the arrays of indices
		# left_lane_inds = np.concatenate(left_lane_inds)
		# right_lane_inds = np.concatenate(right_lane_inds)

		# # Extract left and right line pixel positions
		# leftx = nonzerox[left_lane_inds]
		# lefty = nonzeroy[left_lane_inds]
		# rightx = nonzerox[right_lane_inds]
		# righty = nonzeroy[right_lane_inds]

		# # Fit a second order polynomial to each using np.polyfit()
		# # If there isn't a good fit, meaning any of leftx, lefty, rightx, and righty are empty,
		# # the second order polynomial is unable to be sovled.
		# # Thus, it is unable to detect edges.
		# try:
		# ##TODO
		# 	left_fit = np.polyfit(lefty, leftx, 2)
		# 	right_fit = np.polyfit(righty, rightx, 2)
		# ####
		# except TypeError:
		# 	print("Unable to detect lanes - None")
		# 	return None
		pass
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






##### Visualization Function #####
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

	# ======================
	if left_fit is not None:
		left_detected = True
	else:
		left_detected = False
	
	if right_fit is not None:
		right_detected = True
	else:
		right_detected = False
	# ======================
	
	# Create an image to draw on and an image to show the selection window
	out_img = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')
	window_img = np.zeros_like(out_img)
	# Color in left and right line pixels
	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

	# Generate x and y values for plotting
	ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
	if left_detected:
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	if right_detected:
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Generate a polygon to illustrate the search window area
	# And recast the x and y points into usable format for cv2.fillPoly()
	# margin = 100  # NOTE: Keep this in sync with *_fit()
	margin = 50  # NOTE: Keep this in sync with *_fit()
	
	if left_detected:
		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
	if right_detected:
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

	# Draw the lane onto the warped blank image
	# cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
	if left_detected:
		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 255))
	# cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
	if right_detected:
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 255))




######
	# middle_fitx = (left_fitx + right_fitx) / 2

	# # 將中間線的 x 和 y 坐標合併為一個 Nx2 的陣列
	# pts_middle = np.array([np.transpose(np.vstack([middle_fitx, ploty]))], dtype=np.int32)

	# # Draw the lane onto the warped blank image
	# cv2.polylines(window_img, [pts_middle], isClosed=False, color=(255, 0, 0), thickness=15)
	# waypoint_idx = [0,int(len(left_fitx)/2),len(left_fitx)-1]
	# #print(f'waypoint={waypoint_idx}')
	# waypoint=[]
	# for i in waypoint_idx:
	# 	cv2.circle(window_img, (int(middle_fitx[i]), int(ploty[i])), 25, (0, 0, 255), -1)
	# 	waypoint.append((int(middle_fitx[i]), int(ploty[i])))
	# #print(f'waypoint={waypoint}')
######
	result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

	# plt.imshow(result)
	# plt.plot(left_fitx, ploty, color='yellow')
	# plt.plot(right_fitx, ploty, color='yellow')
	# plt.xlim(0, 1280)
	# plt.ylim(720, 0)

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


def final_viz(undist, left_fit, right_fit, m_inv, detect_stopsign=False, lane_records=[0,0]):

	"""
	Final lane line prediction visualized and overlayed on top of original image
	"""
	# ======================
	if left_fit is not None:
		left_detected = True
	else:
		left_detected = False
	
	if right_fit is not None:
		right_detected = True
	else:
		right_detected = False
	# ======================
	# Generate x and y values for plotting
	ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])
	if left_detected:
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	if right_detected:
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	# Create an image to draw the lines on
	#warp_zero = np.zeros_like(warped).astype(np.uint8)
	#color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
	color_warp = np.zeros((image_h, image_w, 3), dtype='uint8')  # NOTE: Hard-coded image dimensions

	# Recast the x and y points into usable format for cv2.fillPoly()
	pts = None
	if left_detected:
		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts = pts_left
	if right_detected:
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = pts_right
	if left_detected and right_detected:
		pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	# if pts is not None:
		# cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

##########
	# middle_fitx = (left_fitx + right_fitx) / 2

	# # 將中間線的 x 和 y 坐標合併為一個 Nx2 的陣列
	# pts_middle = np.array([np.transpose(np.vstack([middle_fitx, ploty]))], dtype=np.int32)

	# # 假設 'image' 是目標圖像，繪製中間線
	# #cv2.polylines(color_warp, [pts_middle], isClosed=False, color=(255, 0, 0), thickness=5)  # 使用藍色繪製中間線

	# # Draw the lane onto the warped blank image
	# cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	# cv2.polylines(color_warp, [pts_middle], isClosed=False, color=(255, 0, 0), thickness=15)
	# waypoint_idx = [0,int(len(left_fitx)/2),len(left_fitx)-1]
	# #print(f'waypoint={waypoint_idx}')
	# waypoint=[]
	# for i in waypoint_idx:
	# 	#cv2.circle(color_warp, (int(middle_fitx[i]), int(middle_fitx[i])), 25, (0, 0, 255), -1)
	# 	waypoint.append((int(middle_fitx[i]), int(ploty[i])))
	# # 	text = f"({int(middle_fitx[i])}, {int(middle_fitx[i])})"  # 定義要顯示的文字
	# # 	font = cv2.FONT_HERSHEY_SIMPLEX  # 字體
	# # 	font_scale = 0.6  # 字體大小
	# # 	color = (0, 0, 0)  # 白色字體
	# # 	thickness = 2  # 字體厚度
    
	# # 	# 在圖像上指定位置寫文字（可以調整位置，避免與圓點重疊）
	# # 	cv2.putText(color_warp, text, (int(middle_fitx[i]) + 10, int(ploty[i]) - 10), font, font_scale, color, thickness)
	# print(f'waypoint={waypoint}')
	# #transfer waypoint from birdview to real image
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

			ratio = lane_h_px / lane_h_m * lane_w_m / lane_w_px

			offset_point = curr_point + [offset * normal[0], offset * normal[1] * ratio]
			offset_arc_points.append(tuple(offset_point))

		return offset_arc_points

	def draw_offset_arc(image, right_fitx, ploty, offset):
		offset_arc_points = get_arc_points(right_fitx, ploty, offset)
		pts = np.array(offset_arc_points, dtype=np.int32)
		# cv2.polylines(image, [pts], isClosed=False, color=(0, 0, 0), thickness=20)
		return pts

	offset = lane_w_px/2 # for right lane: left offset from original lane

	# If both lanes are detected, choose the one with less variation
	waypoint=[]



	# Generate waypoints for detour (left waypoints from left lane) if stop sign detected
	if detect_stopsign:
		# lane_records = [Left detected previously, Lane changed]
		if lane_records == [0,0]: # First time detecting stop sign
			# Generate waypoints from right lane with 3x offset (left wpts from left lane)
			if right_detected:
				pts_newway = draw_offset_arc(color_warp, right_fitx, ploty, offset*3)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [0,0]
				print("[LF] Changing lane: right")
			# Generate waypoints from left lane with -1x offset (left wpts from left lane) 
			elif left_detected:
				pts_newway = draw_offset_arc(color_warp, left_fitx, ploty, offset)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [1,0]
				print("[LF] Changing lane: left")
		elif lane_records == [1,0]: # Left lane detected previously
			# Generate waypoints for normal driving
			if right_detected:
				pts_newway = draw_offset_arc(color_warp, right_fitx, ploty, offset)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [1,1]
				print("[LF] Finish lane change")
			elif left_detected:
				pts_newway = draw_offset_arc(color_warp, left_fitx, ploty, offset)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [1,0]
				print("[LF] Changing lane: left")
		elif lane_records == [2,2]: # Return to original lane when reveive [3,3]
			if right_detected:
				pts_newway = draw_offset_arc(color_warp, right_fitx, ploty, offset)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [2,2]
				print("[LF] Following switched lane")
			elif left_detected:
				pts_newway = draw_offset_arc(color_warp, left_fitx, ploty, -offset)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [2,2]
				print("[LF] Following switched lane")
		elif lane_records == [3,3]: # Return to original lane
			if left_detected:
				pts_newway = draw_offset_arc(color_warp, left_fitx, ploty, -3*offset)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [3,3]
				print("[LF] Return to original lane: left")
			elif right_detected:
				pts_newway = draw_offset_arc(color_warp, right_fitx, ploty, -offset)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [4,3]
				print("[LF] Return to original lane: right")
		elif lane_records == [4,3]:
			if right_detected:
				pts_newway = draw_offset_arc(color_warp, right_fitx, ploty, -offset)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [4,3]
				print("[LF] Return to original lane: right")
			elif left_detected:
				pts_newway = draw_offset_arc(color_warp, left_fitx, ploty, -offset)
				waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
				for i in waypoint_idx:
					waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
				lane_records = [4,4]
				print("[LF] Finish lane change")


		print("[LF] Stop sign detected", lane_records)


	# Generate waypoints for normal driving
	else:
		if right_detected and left_detected:

			# right_variation = np.std(right_fitx)
			# left_variation = np.std(left_fitx)

			# if right_variation < left_variation:
			# 	left_detected = False
			# else:
			# 	right_detected = False

			if len(right_fitx) < 100:
				right_detected = False
			elif len(left_fitx) < 100:
				left_detected = False
			else:
				right_variation_s2m = np.std(right_fitx[:int(len(right_fitx)/2)])
				right_variation_m2e = np.std(right_fitx[int(len(right_fitx)/2):])
				dif_right_variation = abs(right_variation_s2m - right_variation_m2e)
				left_variation_s2m = np.std(left_fitx[:int(len(left_fitx)/2)])
				left_variation_m2e = np.std(left_fitx[int(len(left_fitx)/2):])
				dif_left_variation = abs(left_variation_s2m - left_variation_m2e)
				if dif_right_variation < dif_left_variation:
					left_detected = False
				else:
					right_detected = False
		

		# Generate waypoints from right lane
		if right_detected:
			pts_newway = draw_offset_arc(color_warp, right_fitx, ploty, offset)
			# waypoint_idx = [0,int(len(pts_newway)/2),len(pts_newway)-1]
			waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
			for i in waypoint_idx:
				waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
			# print(f'waypoint_black={waypoint}')

		# Generate waypoints from left lane
		elif left_detected:
			offset = -offset # for left lane: right offset from original lane

			pts_newway = draw_offset_arc(color_warp, left_fitx, ploty, offset)
			# waypoint_idx = [0,int(len(pts_newway)/2),len(pts_newway)-1]
			waypoint_idx = np.linspace(0, len(pts_newway)-1, 5, dtype=int)
			for i in waypoint_idx:
				waypoint.append((int(pts_newway[i][0]), int(pts_newway[i][1])))
			# print(f'waypoint_black={waypoint}')





	# Conversion from px to m
	# id=0: farest point from gem
	waypoints_m = [] 
	if waypoint:
		for (x, y) in waypoint:
			x_m = x * lane_w_m / lane_w_px
			y_m = y * lane_h_m / lane_h_px
			waypoints_m.append((float(x_m), float(y_m)))

		# id=3: gem position
		birdeye_w_m = lane_w_m * image_w / lane_w_px
		waypoints_m.append((birdeye_w_m/2, lane_h_m + camera_to_rearwheel))

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
		cv2.circle(newwarp, (x,y), 5, (0, 255, 255), -1)
		# text = f"({x}, {y})"  # 定義要顯示的文字
		# font = cv2.FONT_HERSHEY_SIMPLEX  # 字體
		# font_scale = 1  # 字體大小
		# color = (255, 0, 0)  # 白色字體
		# thickness = 2  # 字體厚度
    
		# 在圖像上指定位置寫文字（可以調整位置，避免與圓點重疊）
		# cv2.putText(newwarp, text, (x + 10, y - 10), font, font_scale, color, thickness)
###########


	undist = np.array(undist, dtype=np.uint8)
	newwarp = np.array(newwarp, dtype=np.uint8)
	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

	# return result
	return result, waypoints_m, lane_records
