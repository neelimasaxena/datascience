import flask
from flask import request, jsonify

import numpy as np
import cv2
import time
import os
import label_image
import webbrowser
import glob
import imutils
import uuid
import base64
size = 4
from mimetypes import guess_extension

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Create some test data for our catalog in the form of a list of dictionaries.
books = [{'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}]

# We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
## directory where the images to be saved:
LABLED_PATH = "images_labled/"
VIDEO_PATH = "video.mp4" # Change this
IMAGE_PATH = "images/" # ...and this
       
# Define the duration (in seconds) of the video capture here
capture_duration = 21

# # if os.path.isfile('output.avi'):
# # os.remove('output.avi')
#cap = cv2.VideoCapture(0)

# Base64 works on multiples of 4 characters..
# ..Sometimes we get 3/2/1 characters and it might be midway through another.
def relaxed_decode_base64(data):

 # If there is already padding we strim it as we calculate padding ourselves.
 if '=' in data:
  data = data[:data.index('=')]

 # We need to add padding, how many bytes are missing.
 missing_padding = len(data) % 4

 # We would be mid-way through a byte.
 if missing_padding == 1:
  data += 'A=='
 # Jut add on the correct length of padding.
 elif missing_padding == 2:
  data += '=='
 elif missing_padding == 3:
  data += '='

 # Actually perform the Base64 decode.
 return base64.b64decode(data)

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Distant Reading Archive</h1>
<p>A prototype API for distant reading of science fiction novels.</p>'''

## A route to return all of the available entries in our catalog.
#@app.route('/api/books/all', methods=['GET'])
#def api_all():
#    return jsonify(books)
@app.route('/api/analysis', methods=['POST'])
def api_src():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    
    #if 'src' in request.args:
    #    #imgstring = request.args['src']
    #    imgstring ='data:image/jpeg;base64,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'
    #else:
    #    return "Error: No src field provided. Please specify an image."
    req_data = request.get_json(force=True) # force=True will make sure this works even if a client does not specify application/json
    imgstring = req_data['src']
    #imgstring=request.data
    #print(imgstring)
    # Create an empty list for our results
    results = []
    current_frame = 0
    ## Loop through the data and match results that fit the requested ID.
    ## IDs are unique, but other fields might return many results
    #for book in books:
    #    if book['id'] == id:
    #        results.append(book)
    i = 0
    KPS = 0.2 # Target Keyframes Per Second
    EXTENSION = ".png"
    fname = str(uuid.uuid1())
    #print(guess_extension(imgstring))  
    #print(imgstring.partition(",")[2])  
    #imgs= imgstring.partition(",")[2]
    #imgs=imgstring.split(",")[1]

    block = imgstring.split(";")
    #Get the content type of the image
    #contentType = block[0].split(":")[1];// In this case "image/gif"
    #get the real base64 content of the file
    realData = block[1].split(",")[1]

   
    #pad = len(imgstring)%4
    #imgstring += b"="*pad
    #imgdata= base64.b64decode(imgstring.encode('utf-8'))
    #imgdata= base64.b64decode(imgstring) 
    imgdata=relaxed_decode_base64(realData)

    #filenm, m = urlretrieve(imgstring)
    #print(filenm, guess_extension(m.get_content_type()))

    #imgdata = base64.b64decode(imgstring)
    filename = IMAGE_PATH+fname+EXTENSION  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)

    file = filename
    ## Reading the given Image with OpenCV
    im = cv2.imread(file)
    minisize = (int(im.shape[1] / size),int(im.shape[0] / size))
    miniframe = cv2.resize(im, minisize)
    faces = classifier.detectMultiScale(miniframe)

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(miniframe)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x,y), (x + w,y + h), (0,255,0), 4)
        
        #Save just the rectangle faces in SubRecFaces
        sub_face = im[y:y + h, x:x + w]

        FaceFileName = "pic.jpg" #Saving the current image from the webcam for testing.
        cv2.imwrite(FaceFileName, sub_face)
        
        text = label_image.main(FaceFileName)# Getting the Result from the label_image file, i.e., Classification Result.
        text = text.title()# Title Case looks Stunning.
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(im, text,(x + w,y), font, 1, (0,0,255), 2)
        
        filename = LABLED_PATH + fname + EXTENSION
        cv2.imwrite(filename, im)

        #Encoding image
        with open(filename, "rb") as image_file:
            encodedStrByte = base64.b64encode(image_file.read())

        encodedStr= str(encodedStrByte)
        encodedStr=encodedStr.replace("b'","")
        encodedStr=encodedStr.replace("'","")

        #encodedStr = base64.b64encode(str(encodedStr, "utf-8") )
        print(encodedStr);
        res= {'id': filename,'origional_img': imgstring,'analysed_img':"data:image/jpeg;base64,"+str(encodedStr),'expression':text }
        results.append(res)

        #fname+=0.05
      
    current_frame += 1   
    i += 1
    #cap.release()

    cv2.destroyAllWindows()

    
    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)

#app.run()
#app.run(host='0.0.0.0', port=91)
app.run(host="192.168.7.124",port=5010,debug=False)