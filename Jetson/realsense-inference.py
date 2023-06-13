import cv2
import numpy as np
import pyrealsense2 as rs
import time
import mqtt

# Constants.
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
SCORE_THRESHOLD = 0.75
NMS_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.5

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

# Configure realsense stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# used to record the time when we processed last frame
prev_frame_time = 0
new_frame_time = 0

def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle. 
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs


def post_process(input_image, depth_image ,outputs):
	# Lists to hold respective values while unwrapping.
	class_ids = []
	confidences = []
	boxes = []

	image_height, image_width = input_image.shape[:2]

	# Resizing factor.
	x_factor = image_width
	y_factor =  image_height

	# Iterate through 25200 detections.
	for out in outputs:
		for detection in out:
			classes_scores = detection[5:]
			class_id = np.argmax(classes_scores)
			confidence = classes_scores[class_id]
			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = detection[0], detection[1], detection[2], detection[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
			  
				box = np.array([left, top, width, height])
				boxes.append(box)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cv2.rectangle(input_image, (left, top), (left + width, top + height), RED, 3)
		label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
		draw_label(input_image, label, left, top)
		
		#distance
		cx = left + (width//2)
		cy = top + (height//2)
		# cx = 320
		# cy = 240
		#cy = top + (height//2+height//8)
		distance = depth_image[cy,cx]/10
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		cv2.imshow('depth', depth_colormap)	
		cv2.circle(input_image, (cx, cy), 3, (0, 255, 0), -1)
		cv2.circle(depth_image, (cx, cy), 3, (0, 255, 0), -1)
		cv2.putText(input_image, "{}cm".format(distance), (left, top - 5), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

		#potition
		# print(distance,cy,cx)
		if distance>0:
			if cx < 230 and distance<500:
				posisi = "A"
			elif cx >= 220 and cx < 500 and distance<350:
				posisi = "B"
			elif cx >= 400 and cx < 500 and distance>=350:
				posisi = "C"
			elif cx >= 500 and distance<330:
				posisi = "B"
			elif cx >= 500 and distance>=300:
				posisi = "C"
			elif cx > 200 and cx < 440 and  distance>=420:
				posisi = "D"
			else:
				posisi = ""
			cv2.putText(input_image, "Posisi: {}".format(posisi), (520, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
			#mqtt send
			status = mqtt.subscribe(client,"robot/docking")
			print(status,posisi,cx,distance)
			if status == "dock":
				mqtt.publish(client,classes[class_ids[i]],posisi)
		else:
			posisi = ""
			cv2.putText(input_image, "Posisi:{}".format(posisi), (520, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)

	cv2.putText(input_image, "x:{}".format(cx), (520, 60), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
	return input_image


if __name__ == '__main__':
	#mqtt
	global client 
	client = mqtt.connect_mqtt()
	client.loop_start()

	# Load class names.
	classesFile = "yolo/obj.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')

	# Load image.
	frame = pipeline.wait_for_frames()
	color_frame = frame.get_color_frame()
	color_image = np.asanyarray(color_frame.get_data())

	# Give the weight files to the model and load the network using them.
	net = cv2.dnn.readNet(model='training/rs-full_tiny/yolov4-tiny-custom_best.weights', config='cfg/yolov4-tiny-custom.cfg')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
	# Process image.
	while True:
		#ret, frame = cap.read()
		frame = pipeline.wait_for_frames()
		color_frame = frame.get_color_frame()
		depth_frame = frame.get_depth_frame()
		
		if not color_frame:
			continue
		
		# Convert image to numpy array
		color_image = np.asanyarray(color_frame.get_data())      
		depth_image = np.asanyarray(depth_frame.get_data())
		depth_image = depth_image[80:390, 105:515] #y,x
		depth_image= cv2.resize(depth_image,(640,480))
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		
		detections = pre_process(color_image, net)
		img = post_process(color_image.copy(), depth_image, detections)
		
		#count fps
		new_frame_time = time.time()
		fps = 1/(new_frame_time-prev_frame_time)
		prev_frame_time = new_frame_time
		fps = int(fps)
		# print(fps)
		
		# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		t, _ = net.getPerfProfile()
		label = 'Inference time: %.2f ms, FPS: %i' % (t * 1000.0 / cv2.getTickFrequency(), fps)
		#print(label)
		cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
		stack = np.hstack((img, depth_colormap))
		
		#cv2.imshow('Output', stack)	
		cv2.imshow('Output', img)	

		if cv2.waitKey(10) & 0xFF == ord('q'):	
			break
	# cap.release()
	cv2.destroyAllWindows()
