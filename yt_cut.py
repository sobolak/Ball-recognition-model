import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
from pymysql import NULL
from PIL import Image

video = "football_slow_motion.mp4" #nazwa filmiku
count = 0
cuts = [[1,3],[30,33],[60*1 + 58 , 60*2]] #to co chcemy wyciąć minutki poprostu *60
os.makedirs("captures/"+video[:-4])
spacing = 10 #co która klatka
size = 1920, 1080

for x in cuts:
	ffmpeg_extract_subclip(video, x[0], x[1], targetname=video[:-4]+"_"+str(x)+".mp4")
	buffor = cv2.VideoCapture(video[:-4]+"_"+str(x)+".mp4")
	success = True
	while success:
		success,image = buffor.read()
		if count%spacing == 0:
			num = count//spacing
			name = "captures/"+video[:-4]+"/"+video[:-4]+"_frame%d.jpg" % num
			try:
				cv2.imwrite(name, image)
				im = Image.open(name)
				im_resized = im.resize(size, Image.ANTIALIAS)
				im_resized.save("captures/"+video[:-4]+"/"+video[:-4]+"_frame%d_R.jpg" % num)
				im.close
				os.remove(name)
				print('Frame update: ', num)
			except:
				pass
		count += 1
	buffor = NULL
	os.remove(video[:-4]+"_"+str(x)+".mp4")