const STATE = require("./State");
const cv = require("opencv4nodejs");
const GREEN = new cv.Vec3(0,255,0);
const path = require("path");
const EventEmitter = require('events');

class DrowsinessDetector
{
    
    constructor()
    {
        // Thresholds
        this.EARThreshold = 0.3;
        this.warningsThreshold = 3;
        this.WarningMiliSecondsThreshold = 2000;
        this.WarningMiliSecondsWaiting = 2000;
        this.SafeModeMiliSecondsThreshold = 5000;
        //Number of warnings to enter in safe mode
        this.warnings = 0;
        this.state = STATE.INITIAL; 
        // Initialice face detector
        const faceClassifierOpts = {
            minSize: new cv.Size(30, 30),
            scaleFactor: 1.126,
            minNeighbors: 1,
        }
        this.classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);
        // Initialice Landmark detector
        this.loadLandmarkDetector();
        this.onChangeState = new EventEmitter();
        this.startTime = null;
        this.netFaceDetector = this.loadFaceDetector();
    }

    loadFaceDetector()
    {
        const prototxt = path.resolve("data/face", "deploy.prototxt");
        const modelFile = path.resolve("data/face", "face_detector.caffemodel");
        // Load the face detector
        return cv.readNetFromCaffe(prototxt, modelFile);
    }

    loadLandmarkDetector()
    {
        const modelLandmarkFile = path.resolve("data/face", "lbfmodel.yaml");

        this.facemark = new cv.FacemarkLBF();
        this.facemark.loadModel(modelLandmarkFile);
    }

    process(image)
    {
        const result = this.detectFaceAndLandmarks(image);

        // Face not detected or lost
        if(result.landmarks == null)
        {
            switch(this.state)
            {
                case STATE.FACE_DETECTED:
                case STATE.EYES_CLOSED_DETECTED:
                        this.changeState(STATE.INITIAL);
                    break;
            }

            this.startTime = null;
        }
        // Face detected
        else
        {
            switch(this.state)
            {
                case STATE.INITIAL:
                        this.changeState(STATE.FACE_DETECTED);
                    break;

                case STATE.FACE_DETECTED:
                    if(this.checkEyesClosed(result.landmarks))
                    {
                        // Start counting the time user keeps eyes closed
                        this.startTime = Date.now();
                        this.changeState(STATE.EYES_CLOSED_DETECTED);
                    }
                    break;

                case STATE.EYES_CLOSED_DETECTED:
                    if(this.checkEyesClosed(result.landmarks))
                    {
                        const milisecondsEyesClosed = Math.abs(this.startTime - Date.now());

                        if(milisecondsEyesClosed > this.WarningMiliSecondsThreshold)
                        {
                            // Start counting miliseconds alarm 
                            this.startTime = Date.now();
                            this.warnings ++;
                            this.changeState(STATE.WARN_USER);
                        }
                    }
                    else
                    {
                        this.changeState(STATE.FACE_DETECTED)
                    }
                    break;

                case STATE.WARN_USER:
                    // When the number of warnings approach the threshold enter in safe mode
                    if(this.warnings >= this.warningsThreshold)
                    {
                        this.changeState(STATE.SAFE_MODE);
                    }
                    // Return to eyes closed state
                    else
                    {
                        const milisecondsAlarmSound = Math.abs(this.startTime - Date.now());
                        if(milisecondsAlarmSound > this.WarningMiliSecondsWaiting)
                        {
                            this.changeState(STATE.FACE_DETECTED);
                            this.startTime = null;
                        }
                    }
                    break;
                case STATE.SAFE_MODE:
                        //image = this.drawEasterEgg(result.landmarks, image);
                    break;
            }

            image = this.drawEyes(result.landmarks,image);
        }

        if(result.faceRects)
        {
            image.drawRectangle(result.faceRects[0], GREEN, 2);
        }

        return image;

    }

    changeState(newState)
    {
        this.state = newState;
        this.onChangeState.emit(newState);
    }

    detectFaceAndLandmarks(image)
    {
        let result = {face: null, landmarks: null, originalDimensions: {cols: -1, rows: -1}}; 
        // Detect faces
        //const faces = this.classifier.detectMultiScale(grayScale).objects;
        // Get the original dimensions of the image
        const rows = image.rows;
        const cols = image.cols;

        const resizedForNet = image.resize(300,300);

        const blob = cv.blobFromImage(resizedForNet);

        this.netFaceDetector.setInput(blob);

        let outputBlob = this.netFaceDetector.forward();

        outputBlob = outputBlob.flattenFloat(outputBlob.sizes[2], outputBlob.sizes[3]);

        let faces = Array(outputBlob.rows).fill(0)
        .map((res, i) => {
          const className = "face";
          const confidence = outputBlob.at(i, 2);
          const topLeft = new cv.Point(
            outputBlob.at(i, 3) * image.cols,
            outputBlob.at(i, 4) * image.rows
          );
          const bottomRight = new cv.Point(
            outputBlob.at(i, 5) * image.cols,
            outputBlob.at(i, 6) * image.rows
          );
    
          return ({
            className,
            confidence,
            topLeft,
            bottomRight
          })
        });

        // Filter for faces with confidence more than threshold
        faces = faces.filter((face) => face.confidence > 0.7);

         if(faces.length > 0)
         {

            // Creates face rects for landmark detection
            // Reescale the geometries
            let faceRects = faces.map((face) => {return new cv.Rect(face.topLeft.x, 
                                                                        face.topLeft.y, 
                                                                        (face.bottomRight.x - face.topLeft.x), 
                                                                        (face.bottomRight.y - face.topLeft.y) 
                                                                    )
                                                });

            let grayScale = image.bgrToGray();

            grayScale = grayScale.equalizeHist();

            result.faceRects = faceRects;

            const facelandmarks = this.facemark.fit(grayScale, faceRects);

            if(facelandmarks.length > 0)
            {
                result = {face: faces[0], landmarks: facelandmarks[0]};
            }
        }

        return result;
    }

    calculateEAR(landmarks)
    {
        // Calculate EAR of left eye
        const EARLeft = (this.euclidean(landmarks[37], landmarks[41]) + 
                         this.euclidean(landmarks[38], landmarks[40])) /
                         //////////////////////////////////////////////
                         (2 * this.euclidean(landmarks[36], landmarks[39])); 

        // Calculate EAR of right eye
        const EARRight = (this.euclidean(landmarks[43], landmarks[47]) + 
                          this.euclidean(landmarks[44], landmarks[46])) / 
                          //////////////////////////////////////////////
                          (2 * this.euclidean(landmarks[42], landmarks[45])); 

        // The total aspect ratio will be the maximum beetwen them
        // due to the user could only have one eye closed

        return Math.max(EARLeft, EARRight);
    }

    euclidean(p1, p2)
    {
        return Math.sqrt(Math.abs(Math.pow(p2.x - p1.x, 2) + 
                        Math.pow(p2.y - p1.y, 2)));
    }

    checkEyesClosed(landmarks)
    {
        // Compute the eye aspect ratio
        const EAR = this.calculateEAR(landmarks);
        // When EAR is lower than EAR Threshold change the state
        return EAR < this.EARThreshold;
    }

    drawLandmarks(points, image)
    {
        for(let indPoint = 0; indPoint < points.length; indPoint ++)
        {
            image.drawCircle(points[indPoint], 2, GREEN, 2);
        }

        return image;
    }

    drawEasterEgg(points, image)
    {
        const destinationPoint = points[66];

        const imageDraw = cv.imread(path.resolve("./images", "pene.png",), cv.IMREAD_COLOR);

        const width = imageDraw.cols;
        const height = imageDraw.rows;
        const imageCopied = imageDraw.copyTo(image.getRegion(new cv.Rect(destinationPoint.x - 20, 
                                                                          destinationPoint.y - 10, width, height)));
                                             
        return image;
    }

    drawEyes(landmarks, image)
    {
        // Right eye
        image.drawFillConvexPoly(landmarks.slice(36, 42), GREEN);
        // Left eye
        image.drawFillConvexPoly(landmarks.slice(42, 48), GREEN);

        return image;
    }
}

module.exports = DrowsinessDetector;