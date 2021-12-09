import numpy as np
import cv2


img1 = cv2.imread(r'D:\folders_put_in_desktop\Taipei_Tech\Programming\tensorflow_kares\gesture_detect\1000.jpg', -1)
g = img1.copy()
g.fill( 0 )
nr, nc = img1.shape[:2]
ycrcb = cv2.cvtColor( img1, cv2.COLOR_BGR2YCrCb )
for x in range( nr ):
    for y in range( nc ):
        Cr = int( ycrcb[x,y,1] )
        Cb = int( ycrcb[x,y,2] )
        if ( Cb >= 77 and Cb <= 127 and \
                Cr >= 133 and Cr <= 173 ):
            g[x,y,0] = g[x,y,1] = g[x,y,2] = 255
img2 = g
cv2.imshow( "Original Image", img1 )
cv2.imshow( "Skin Color Detection", img2 )
#cv2.imwrite(..., img2)
cv2.waitKey( 0 )