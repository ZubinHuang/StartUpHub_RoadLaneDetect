import cv2
def img_resize( img ):
    img = cv2.resize( img, (1280, 720) )

cam = cv2.VideoCapture( 'D:/touching fish/CarND-Advanced-Lane-Lines-master/园区道路线视频/14s.mp4' )
i = 0
ret = True
while( ret ):
    ret, img = cam.read()
    if ret:
        # img = img_resize( img )
        cv2.imwrite( './14s_single_picture/' + str( i ) + '.jpg', img )
        print( i )
        i += 1

