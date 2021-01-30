from PIL import ImageDraw,Image
import socket
import sys
import cv2
import struct ## new
import pickle,os
import argparse
import time, datetime
import numpy as np
import threading
# import matplotlib.pyplot as plt
CLASSES = UCF24_CLASSES = (  # always index 0
        'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling', 'Diving', 'Fencing',
        'FloorGymnastics', 'GolfSwing', 'HorseRiding', 'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing',
        'SalsaSpin','SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling',
        'Surfing', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Demo Server')
parser.add_argument('--showImage', default=True, type=str2bool, help='Show event detection image')
parser.add_argument('--streamPort', default=9875, type=int, help='Port for streaming')
parser.add_argument('--actionPort', default=8488, type=int, help='Port for action localization result')
# parser.add_argument('--cloudAddr', default="127.0.0.1", type=str, help='Cloud ip address')
parser.add_argument('--cloudAddr', default="192.168.32.152", type=str, help='Cloud ip address')
# parser.add_argument('--edgeAddr', default="127.0.0.1", type=str, help='Edge ip address')
parser.add_argument('--edgeAddr', default="192.168.251.195", type=str, help='Edge ip address')

args = parser.parse_args()

def listen_for_action(req_time):
	s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
	# print('Socket created')

	s.bind((args.edgeAddr,args.actionPort))
	# print('Socket bind complete')
	s.listen(10)
	print('Listening for actions...')

	conn,addr=s.accept()

	data = b""
	payload_size = struct.calcsize(">L")

	# output_dir = '/home/bo/research/dataset/ucf24/detections'
	output_dir = '/home/uiuc/realtime-action-detection/detections'
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	dtind = 0

	total_time = 0
	total_size = 0
	total_event_time = 0
	total_dec_time = 0
	time_cnt = 0
	t1 = time.perf_counter()

	while True:
		# receive timestamp
		while len(data) < 26:
			data += conn.recv(4096)
		recv_time0 = datetime.datetime.now()
		send_time = datetime.datetime.strptime(data[:26].decode(), "%Y-%m-%d %H:%M:%S.%f")
		data = data[26:]

		# receive number of detections
		while len(data) < payload_size:
			data += conn.recv(4096)

		num_detection = struct.unpack(">L", data[:payload_size])[0]
		data = data[payload_size:]

		res_str = ''
		for i in range(num_detection):
			# receive img size
			while len(data) < payload_size:
				data += conn.recv(4096)

			img_size = struct.unpack(">L", data[:payload_size])[0]
			data = data[payload_size:]

			# receive class label
			while len(data) < payload_size:
				data += conn.recv(4096)

			class_label = struct.unpack(">L", data[:payload_size])[0]
			data = data[payload_size:]

			# receive gt label
			while len(data) < payload_size:
				data += conn.recv(4096)

			packed_gt_label = data[:payload_size]
			data = data[payload_size:]
			gt_label = struct.unpack(">L", packed_gt_label)[0]

			# receive image data
			while len(data) < img_size:
				data += conn.recv(4096)
			frame_data = data[:img_size]
			data = data[img_size:]

			# decode and save image
			frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
			frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

			output_tube_dir = output_dir + '/event-' + \
	        				str(dtind) + '-' + \
	        				CLASSES[class_label] + '(' + CLASSES[gt_label] + ').png'
			img = Image.fromarray(frame.astype(np.uint8))
			img.save(output_tube_dir)

			total_size += img_size + 3*payload_size
			res_str += '[RESULT] Action: '+CLASSES[class_label]+'('+CLASSES[gt_label]+')--->'+('Success' if class_label==gt_label else 'Failure')+'\n'
			dtind += 1
		total_size += 26 + payload_size

		recv_time = datetime.datetime.now()
		tf = time.perf_counter()

		total_time += (recv_time-send_time).total_seconds()
		total_event_time += (tf - req_time.popleft())
		latency = (recv_time0 - send_time).total_seconds()
		time_cnt += 1
		avg_time = total_time/time_cnt
		avg_size = total_size/time_cnt
		avg_event_time = total_event_time/time_cnt
		bw = avg_size/avg_time/1000000

		print(res_str+'[RESULT] Service time:{:1.3f}s, network latency:{:.6f}s, action throughput:{:1.3f}MB/s'.format(avg_event_time,latency,bw))


if __name__ == '__main__':
	# read video
	print('Loading dataset')
	# save np.load
	np_load_old = np.load

	# modify the default parameters of np.load
	np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

	label_file,index_file = 'label_file.npy', 'index_file.npy'
	with open(label_file, 'rb') as f:
		labels = np.load(f)
	with open(index_file, 'rb') as f:
		indexes = np.load(f)
	d=np.load("dataset.npy").item()
	image_ids = d['ids']
	video_list = d['video_list']

	# restore np.load for future normal usage
	np.load = np_load_old
	print(labels.shape,indexes.shape)

	# start listen for action
	from collections import deque
	req_time = deque()
	action_thread = threading.Thread(target=listen_for_action, args=(req_time,))
	action_thread.start()

	# connect to server
	client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	ADDR = (args.cloudAddr,args.streamPort)
	client.connect(ADDR)

	# display/send video
	means = (104, 117, 123)
	chunk_size = 20
	cur_idx = 0
	frame_buffer = []
	total_enc_time = 0
	enc_cnt = 0
	pre_videoname = ""
	pre_videoid = -1
	streaming_start_t = None
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	videofile = 'images/output.avi'
	out = cv2.VideoWriter(videofile,fourcc, 20.0, (1024,512))
	streaming_start_t = time.perf_counter()
	for label,index in zip(labels,indexes):
		# load img
		img_name = 'dataset/{:05d}.png'.format(index)
		pil_img = Image.open(img_name)
		img = np.array(pil_img)
		# get video info
		annot_info = image_ids[index]
		video_id = annot_info[0]
		videoname = video_list[video_id]
    	# transform for displaying
		for ch in range(0,3):
			img[:,:,ch] += means[2-ch]
		# whether to show image
		if args.showImage:
			b,g,r = cv2.split(img)
			frame_rgb = cv2.merge((r,g,b))
			cv2.imshow('Edge sending window',frame_rgb)
			cv2.waitKey(1)
		frame_buffer += [img]
		# streaming
		if (index+1)%chunk_size==0:
			# write to file
			enc_start = time.perf_counter()
			out = cv2.VideoWriter(videofile,fourcc, 20.0, (1024,512))
			for frame in frame_buffer:
				out.write(frame)
			frame_buffer = []
			out.release()
			total_enc_time += (time.perf_counter() - enc_start)
			enc_cnt += 1
			# send file
			data = open(videofile,'rb').read()
			data = struct.pack(">L", len(data))+data
			time_str = str(datetime.datetime.now())
			client.send(str.encode(time_str)+data)
			# check whether a video finishes
			if videoname != pre_videoname and pre_videoname != '':
				req_time.append(time.perf_counter())
				streaming_time = time.perf_counter() - streaming_start_t
				print('[STATUS]', pre_videoid, pre_videoname,'streaming:{:0.1f}s, avg encoding:{:0.3f}s'.format(streaming_time, 0 if enc_cnt==0 else total_enc_time/enc_cnt))
				time.sleep(10)
				streaming_start_t = time.perf_counter()
			pre_videoname = videoname
			pre_videoid = video_id
	# clear remaining frames in buffer as the last segment
	if frame_buffer:
		# write to file
		enc_start = time.perf_counter()
		out = cv2.VideoWriter(videofile,fourcc, 20.0, (1024,512))
		for frame in frame_buffer:
			out.write(frame)
		frame_buffer = []
		out.release()
		total_enc_time += (time.perf_counter() - enc_start)
		enc_cnt += 1
		# send file
		data = open(videofile,'rb').read()
		time_str = str(datetime.datetime.now())
		client.send(str.encode(time_str)+struct.pack(">L", len(data))+data)
		streaming_time = time.perf_counter() - streaming_start_t
		print('[STATUS]',videoname,'streaming:{:0.1f}s, avg encoding:{:0.3f}s'.format(streaming_time, 0 if enc_cnt==0 else total_enc_time/enc_cnt))
	# request time of the last segment
	req_time.append(time.perf_counter())
	client.close()
	action_thread.join()
