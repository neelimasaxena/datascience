import cv2
import label_image

size = 4


# We load the xml file
classifier = cv2.CascadeClassifier('F:/NeelimaProjects/FaceDetectionVideoRec/haarcascade_frontalface_alt.xml')
img_counter = 0
webcam = cv2.VideoCapture(0) #Using default WebCam connected to the PC.
cv2.namedWindow("WebCamera")

while True:
    if webcam.isOpened():
        rval, im = webcam.read()
        cv2.imshow("WebCamera", im)
    else:
        webcam.release()
    # (rval, im) = webcam.read()
    # im=cv2.flip(im,1,0) #Flip to act as a mirror

    # # Resize the image to speed up detection
    # mini = cv2.resize(im, (int(im.shape[1]/size), int(im.shape[0]/size)))
    if not rval:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed 
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # detect faces and draw bounding boxes
        minisize = (int(im.shape[1]/size),int(im.shape[0]/size))
        miniframe = cv2.resize(im, minisize)
        faces = classifier.detectMultiScale(miniframe)

        # detect MultiScale / faces 
        faces = classifier.detectMultiScale(miniframe)

        # Draw rectangles around each face
        for f in faces:
            (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
            cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0), 4)
        
            #Save just the rectangle faces in SubRecFaces
            sub_face = im[y:y+h, x:x+w]

            FaceFileName = "pic.jpg" #Saving the current image from the webcam for testing.
            cv2.imwrite(FaceFileName, sub_face)
        
            text = label_image.main(FaceFileName)# Getting the Result from the label_image file, i.e., Classification Result.
            text = text.title()# Title Case looks Stunning.
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(im, text,(x+w,y), font, 1, (0,0,255), 2)

        # Show the image
        cv2.imshow('WebCamera',   im)
        key = cv2.waitKey(10)
        webcam.release()
cv2.destroyAllWindows()



