from ultralytics import YOLO
import cv2
import numpy as np
import requests
from io import BytesIO


model = YOLO("yolo11n-pose.pt")  # load pose detection model

# results = model("images/random/boy-trying-stop-car-among-260nw-1059519287.webp", conf=0.25)  # predict on an image
def get_pose(model, frame):
    """
    Gets pose points from a given YOLO model and frame.
    Args:
        model: YOLO model.
        frame: Frame from the camera.
    Returns:
        bounding_box: bounding box of person with highest confidence
        pose_points: List of pose points. None if low visibility or confidence
    """
    results = model(frame)
    # Access the results
    if len(results) == 0:
        print("No results found.")
        return None
        
    #TODO: change 0 to be index of person with highest confidence
    result = results[0]  # first result
    xywh = result.boxes.xywh  # center-x, center-y, width, height
    xywhn = result.boxes.xywhn  # normalized
    xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    xyxyn = result.boxes.xyxyn  # normalized
    names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    confs = result.boxes.conf  # confidence score of each box
        
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    
    xyxy = xyxy[0] if len(xyxy) > 0 else None
    return xyxy, xy


cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    # cv2.imshow('frame', frame)
    
    #detect pose
    
    boxes, xy = get_pose(model, frame)

    if xy is None or boxes is None:
        continue
    x1, y1, x2, y2 = map(int, boxes)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.putText(frame, f"{names[i]} {confs[i]:.2f}", (x1, y1 - 10),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for i in range(len(xy)):
        for j in range(len(xy[i])):
            x, y = map(int, xy[i][j])
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Draw keypoints
            # cv2.putText(result.orig_img, f"{names[i]} {confs[i]:.2f}", (x + 10, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # Show the image with boxes
    cv2.imshow("Image with Boxes", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
exit()


