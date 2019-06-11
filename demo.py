
import cv2

from facerec import FaceRecognition

def main():
    # Face detection and recognition class
    facerec = FaceRecognition(cpudet=False)
    

    print("Starting capture")

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()

    # Obtain and classify faces from webcam
    process_this_frame = True

    names = []
    dists = []
    while ret:
        img = frame

        # Only process every other frame of video to save time
        if process_this_frame:
            face_info = facerec.evaluate(img)
        
        if face_info:
            names, dists, bbs = face_info

            # Draw and identify bounding boxes
            # for (left, top, right, bottom) in bbs:
            for (left, top, right, bottom), name, dist in zip(bbs, names, dists):
                # Padding to 
                left, top = max(left, 0), max(top, 0)
                right, bottom = min(right, img.shape[1]), min(bottom, img.shape[0])

                if name == "Unknown":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name+" %0.2f"%dist, (left, bottom +20), font, 1.0, color, 1)
            
            # Show
            cv2.imshow('Faces', frame)
            k = cv2.waitKey(1) & 0xFF 
            if k == ord('q'):
                break
            elif k == ord('s'):
                cv2.imwrite("sample.png", img)

        process_this_frame = not process_this_frame
        ret, frame = video_capture.read()

    # Release handle to the webcam
    video_capture.release()

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
