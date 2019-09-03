
import cv2

from facerec import FaceRecognition

import argparse

from time import time

def parse_args():
    parser = argparse.ArgumentParser(description='Train nn for seam carving detection.')
    parser.add_argument("--dist-thresh", action='store', type=float, help="Distance threshold for classification.", default=0.6)
    parser.add_argument("--det-ctx", action='store', type=str, help="Detection context (cpu, gpu).", choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument("--det-thresh", action='store', type=str, help="Detection thresholds (default: '0.7,0.8,0.8').", default='0.7,0.8,0.8')
    parser.add_argument("--det-factor", action='store', type=float, help="Detection pyramid scaling factor (default: 0.709).", default=0.709)
    parser.add_argument("--det-minsize", action='store', type=int, help="Detection minimum size (default: 40).", default=40)
    parser.add_argument("--feat-ctx", action='store', type=str, help="Feature extraction context (cpu, gpu).", choices=['cpu', 'gpu'], default='cpu')

    return parser.parse_args()

def main(dist_thresh, det_ctx, det_thresh, det_factor, det_minsize, feat_ctx):
    # Param parsing
    cpudet = True if det_ctx == 'cpu' else False
    cpufeat = True if feat_ctx == 'cpu' else False
    det_thresh = [float(t) for t in det_thresh.split(',')]

    font = cv2.FONT_HERSHEY_DUPLEX

    # Face detection and recognition class
    facerec = FaceRecognition(dist_threshold=dist_thresh, cpudet=cpudet, det_threshold=det_thresh, det_factor=det_factor, det_minsize=det_minsize, cpufeat=cpufeat)
    

    print("Starting capture")

    # Get a reference to webcam #0 (the default one)
    video_capture = cv2.VideoCapture(0)
    ret, frame = video_capture.read()

    # Obtain and classify faces from webcam
    process_this_frame = True

    names = []
    dists = []
    prev_time = time()
    fps = 1

    while ret:
        img = frame

        # Only process every other frame of video to save time
        if process_this_frame:
            face_info = facerec.evaluate(img)
            # Measuring frames per second
            cur_time = time()
            fps = 1. / (cur_time - prev_time)
            prev_time = cur_time
        
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
                cv2.putText(frame, name+" %0.2f"%dist, (left, bottom +20), font, 1.0, color, 1)
        
        # Show
        cv2.putText(frame, "%.1f"%fps, (20, 20), font, 0.6, (0, 0, 255), 1)
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
    args = parse_args()
    main(args.dist_thresh, args.det_ctx, args.det_thresh, args.det_factor, args.det_minsize, args.feat_ctx)
