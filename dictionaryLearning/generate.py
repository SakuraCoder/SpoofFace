import os, sys
import numpy as np
import pywt
import cv2

#p = 'train'
#read face pos for Replay-Attack database
f_train = open(sys.argv[1] + '_'+sys.argv[2] + '.txt', 'w')
def getFaceLocation(filename):
	f = open(filename, 'rt')
	lines = [k.strip() for k in f.readlines() if k.strip()]
	retval = np.zeros((len(lines), 5), dtype='int16')
	for i, line in enumerate(lines):
	  s = line.split()
	  for j in range(5): 
	    retval[i,j] = int(s[j])
	  if(retval[i,1] == 0 and retval[i,2] ==0 and retval[i,3] == 0 and retval[i,4] == 0):
	  	if(i!=0):
	  		if(retval[i-1,1] == 0 and retval[i-1,2] ==0 and retval[i-1,3] == 0 and retval[i-1,4] == 0):
	  			continue
	  		else:
	  			retval[i,1] = retval[i-1,1]
	  			retval[i,2] = retval[i-1,2]
	  			retval[i,3] = retval[i-1,3]
	  			retval[i,4] = retval[i-1,4]
	return retval
def getFileLength(filename):
	f = open(filename, 'rt')
	lines = f.readlines()
	return len(lines)
def traverse(root, path, size, normalize):
	global f_train
	parents = os.listdir(path)
	for parent in parents:
		child = os.path.join(path,parent)
		if(os.path.isdir(child)):
			if(normalize):
				newdir = os.path.join(root, child)
			else:
				newdir = os.path.join('non-frames', child)
			if(not os.path.exists(newdir)):
				os.makedirs(newdir)
			traverse(root, child, size, normalize)
		elif(os.path.isfile(child) and child[len(child)-4:] == ".mov"):
			if(normalize):
				newdir = os.path.join(root, child).replace(".mov", "")
			else:
				newdir = os.path.join('non-frames', child).replace(".mov", "")
			if(not os.path.exists(newdir)):
				os.makedirs(newdir)
			facefile = os.path.join('face-locations', child).replace(".mov", ".face")
			faceloc = getFaceLocation(facefile)
			#print newdir
			#print facefile
			video = cv2.VideoCapture(child)
			length = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
			
			assert(not (length - getFileLength(facefile)))
			assert(video.isOpened())

			i = 0
			while True:
				ret, frame = video.read()
				#print newdir
				#print i
				#print ret
				if(not ret):
					#print "reach"
					break
				if(faceloc[i,1] == 0 and faceloc[i,2] == 0  and faceloc[i,3] == 0  and faceloc[i,4] == 0 ):
					#print "reach"
					i = i + 1
					continue
				#print i
				#print frame.shape
				#print faceloc[i]
				frame = frame[faceloc[i,2]:(faceloc[i,2] + faceloc[i,4]), faceloc[i,1]:(faceloc[i,1] + faceloc[i,3])]
				

				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				#res = cv2.equalizeHist(frame)
				if(normalize):
					(cA, cD) = pywt.dwt(frame, 'db2', 'sp1')
					cA = cv2.convertScaleAbs(cA)
					cA = cv2.equalizeHist(cA)
					cD = cv2.convertScaleAbs(cD)
					cD = 1.5*cD
					res = pywt.idwt(cA, cD, 'db2', 'sp1')
					res = cv2.convertScaleAbs(res)
				else:
					res = frame
				res = cv2.resize(res,(size,size))
				fname = root + str(i) + '.jpg'
				cv2.imwrite(os.path.join(newdir, fname), res)
				f_train.write(os.path.join(newdir, fname) + " ")
				if(os.path.join(newdir, fname).find("attack") >= 0):
					f_train.write("0\n")
				else:
					f_train.write("1\n")
				i = i + 1
				#print "fuck1"
				
				#print "fuck"
				#res = cv2.equalizeHist(frame)
				#res = np.hstack((frame,equ)) 
             	#res = cv2.resize(res, (64,64))
            	

			    #cv2.imshow('Video', frame)
			
p = sys.argv[1]
r= sys.argv[2]
size = int(sys.argv[3])
normalize = int(sys.argv[4])

traverse(r, p, size, normalize)
#getFaceLocation("face-locations/train/attack/hand/attack_highdef_client027_session01_highdef_photo_adverse.face")