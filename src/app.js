const STATE = require("./State");
const cv = require("opencv4nodejs");
const path = require("path");
const DrowsinesDetector = require("./DrowsinessDetector");
 
// Initialice Drownsiness detector
const drowsinessDetector = new DrowsinesDetector();

const videoCapture = new cv.VideoCapture(0);

let message = "OK!";

//drowsinessDetector.onChangeState.on(STATE.INITIAL, () => console.log("Face Lost"));

drowsinessDetector.onChangeState.on(STATE.FACE_DETECTED, () => message = "OK!");

drowsinessDetector.onChangeState.on(STATE.EYES_CLOSED_DETECTED, () => message = "Ojos cerrados!");

drowsinessDetector.onChangeState.on(STATE.WARN_USER, () => message = "Alarma !");

drowsinessDetector.onChangeState.on(STATE.SAFE_MODE, () => message = "Automatico ON...");

while(true)
{
    // Get frame from video Capture
    const frame = videoCapture.read();

    const image = frame.gaussianBlur(new cv.Size(3,3), 1);

    let result = drowsinessDetector.process(image);

    if(drowsinessDetector.state == STATE.WARN_USER)
    {
        result.putText('Status: ' + message, new cv.Point2(result.cols - result.cols / 2.5 , result.rows - result.rows / 14), cv.FONT_HERSHEY_PLAIN, 1.25, new cv.Vec3(0,0,255), 2, cv.LINE_4);
    }
    else if(drowsinessDetector.state == STATE.EYES_CLOSED_DETECTED)
    {
        result.putText('Status: ' + message, new cv.Point2(result.cols - result.cols / 2.5 , result.rows - result.rows / 14), cv.FONT_HERSHEY_PLAIN, 1.25, new cv.Vec3(66, 158, 244), 2, cv.LINE_4);
    }
    else
    {
        result.putText('Status: ' + message, new cv.Point2(result.cols - result.cols / 2.5 , result.rows - result.rows / 14), cv.FONT_HERSHEY_PLAIN, 1.25, new cv.Vec3(0, 255, 0), 2, cv.LINE_4);
    }

    // Show original frame
    cv.imshow("camera", result);

    if(cv.waitKey(1) == 113)
        return;

}

videoCapture.release();
cv.destroyAllWindows();

