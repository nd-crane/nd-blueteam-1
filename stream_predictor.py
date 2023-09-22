from threading import Thread
import time
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
import cv2
import numpy as np
from numpy.typing import NDArray

FONT = cv2.FONT_HERSHEY_SIMPLEX

def write_to_csv(frame_number, score, filename, score_threshold=0.0):    
    with open(filename, 'a') as csvfile:
        line = f"{frame_number},{score:.3f}\n"
        csvfile.write(line)        


def transform_image(image, im_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([im_size, im_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transformed_image = transform(image)[0:3, :, :].unsqueeze(0)
    return transformed_image

def predict(model, image_data, im_size):
    transformed_image = transform_image(image_data, im_size)
    with torch.no_grad():
        output = model(transformed_image)
        SMScore = nn.Softmax(dim=1)(output).detach().cpu().numpy()[:, 1]
    return SMScore[0]


def predict_impath(model, image_path, im_size):
    
    transform = transforms.Compose([
        transforms.Resize([im_size, im_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image)[0:3, :, :].unsqueeze(0)
    image.close()

    with torch.no_grad():
        output = model(transformed_image)
        SMScore = nn.Softmax(dim=1)(output).detach().cpu().numpy()[:, 1]
    return SMScore[0]


def draw_boxes(image, boxes, scores, score_threshold=0.0):
    # Iterate over each box and draw it on the image
    for box, score in zip(boxes, scores):
        if score > score_threshold:  # Only draw boxes with scores higher than the threshold
            # Convert box coordinates to integers
            x1, y1, x2, y2 = map(int, box)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the score above the bounding box
            cv2.putText(image, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def get_model_and_size(network, nClasses):
    if network == "resnet":
        im_size = 224
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, nClasses)
    elif network == "inception":
        im_size = 299
        model = models.inception_v3(pretrained=True, aux_logits=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, nClasses)
    else:  # default is DenseNet
        im_size = 224
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, nClasses)
    return model, im_size


def overlay_score(frame: int, image_np, score, frame_display_width, frame_display_height):

    # Resize the frame
    #image_np = cv2.resize(image_np, (frame_display_width, frame_display_height))
    
    # Format text and calculate position
    text = f'Score: {score:.2f}'
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, 1, 2)

    # Position calculations
    text_x = image_np.shape[1] - text_width - 10
    text_y = 30  # Keeping this constant to have the text on the top

    # Choose color based on score
    color = (0, 0, 255) if score < 0.5 else (0, 255, 0)

    # Overlay the score on the image
    cv2.putText(image_np, text, (text_x, text_y), FONT, 1, color, 2, cv2.LINE_AA)

    return image_np

    # Display the frame
    # TODO: refactory frame name 
    cv2.imwrite(f'ND BlueTeam 1 - Model 1 - {frame}.png', image_np)
    #cv2.imshow('ND BlueTeam 1 - Model 1', image_np)

class StreamInput(Thread):
    _stream: cv2.VideoCapture
    _should_exit: bool = False

    # Latest Information
    _latest_timestamp = 0.0

    def __init__(self, location: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._latest_frame = None

        # Initialize Stream
        self._stream = cv2.VideoCapture(location) 
        if not self._stream.isOpened():
            raise Exception("can't open video writer")

        # Reduce buffer size if supported
        self._stream.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    def isOpened(self) -> bool:
        return self._stream.isOpened()
    
    def width(self) -> int:
        return int(self._stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    def height(self) -> int:
        return int(self._stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def latest(self):
        if self._latest_frame is None:
            return (None, None)
        return (self._latest_timestamp, self._latest_frame.copy())

    def run(self) -> None:
        while not self._should_exit:
            ret, frame = self._stream.read()
            if not ret:
                break
            self._latest_frame = frame
            self._latest_timestamp = self._stream.get(cv2.CAP_PROP_POS_MSEC)

        self._stream.close()
        self._stream.release()

    def stop(self):
        self._should_exit = True

class RTSPOutput(Thread):
    def __init__(self, width: int, height: int, fps: int, location: str, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._should_exit = False
        self._latest_frame = None
        self._timestep = 1.0 / fps
        self._stream = cv2.VideoWriter('appsrc ! videoconvert' + \
            ' ! video/x-raw,format=I420' + \
            ' ! x264enc speed-preset=ultrafast key-int-max=' + str(fps * 2) + \
            ' ! video/x-h264,profile=baseline' + \
            f' ! rtspclientsink protocols=tcp location={location}',
            cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
        if not self._stream.isOpened():
            raise Exception("can't open video writer")
    
    def update(self, frame):
        self._latest_frame = frame.copy()
        
    def run(self) -> None:
        while not self._should_exit:
            start = time.time()

            if self._latest_frame is not None:
                self._stream.write(self._latest_frame)
            
            diff = time.time() - start
            if diff > self._timestep:
                time.sleep(diff)

        self._stream.release()
        print("Output finished")

    def stop(self):
        self._should_exit = True


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Read and display video from multiple RTSP streams using OpenCV')
    parser.add_argument('--stream_label', default="Stream_0", help='Stream Label')
    parser.add_argument('--stream_url', required=True, help='Stream RTSP URL')
    parser.add_argument("--weights_path", required=True, help="Path to the model weights file")
    parser.add_argument('--output_folder', default='output', help='Main output folder to save results')
    parser.add_argument('--frame_save_width', type=int, default=960, help='Width to save frames (optional)')
    parser.add_argument('--frame_save_height', type=int, default=720, help='Height to save frames (optional)')
    parser.add_argument('--file_format', default='png', help='File format for saving frames (e.g., jpg, png)')
    parser.add_argument('--time_limit', type=int, default=30, help='Time limit in seconds to capture frames (optional)')
    parser.add_argument('--frame_display_width', type=int, default=640, help='Width to display frames (optional)')
    parser.add_argument('--frame_display_height', type=int, default=480, help='Height to display frames (optional)')
    parser.add_argument('--score_threshold', type=float, default=0.0, help='score_threshold')
    parser.add_argument('--network', default="densenet", type=str)
    parser.add_argument('--nClasses', default=2, type=int)

    args = parser.parse_args()

    #stream_label = 'Stream_0'
    #stream_url = 'rtsp://admin:Marialufy2@192.168.0.101:65534'
    #stream_url = 'rtsp://192.168.0.103:554/H264Video'
    #stream_url = '/home/pmoreira/tai-raite/raite-stream_recorder/video_1.mkv'

    stream_folder = os.path.join(args.output_folder, args.stream_label)
    if not os.path.exists(stream_folder):
        os.makedirs(stream_folder)

    # TODO: refactory this    
    raite_output_path = os.path.join(stream_folder, 'ndblueteam_1-model_1.csv')   

    # Load weights of model
    device = torch.device('cpu')
    weights = torch.load(args.weights_path, map_location=device)
    model, im_size = get_model_and_size(args.network, args.nClasses)
    model.load_state_dict(weights['state_dict'])
    model = model.to(device)
    model.eval()

    #cv2.imshow('ND-BT-1 M1', np.zeros((frame_display_height, frame_display_width, 3), dtype=np.uint8))
    time.sleep(2)
    
    # Open the stream 
    cap_time = time.time()

    input_rtsp = StreamInput(args.stream_url)
    input_rtsp.start()

    output_rtsp = RTSPOutput(input_rtsp.width(), input_rtsp.height(), 30, "rtsp://localhost:8554/blue-team-output")
    output_rtsp.start()

    print("cap time", time.time()-cap_time)
    frame_count = 0
    last_timestamp = 0.0
    while input_rtsp.isOpened():
        frame_time = time.time()

        timestamp, frame = input_rtsp.latest()
        if frame is None or last_timestamp == timestamp:
            time.sleep(0.01)
            continue

        last_timestamp = timestamp

        # Make prediction
        frame_count += 1
        
        score = predict(model, frame, im_size)
        #score = 0

        # Write predictions for RAITE ouput format 
        # write_to_csv(frame_count, score, raite_output_path)        

        # Display the frame with boxes
        overlay_score(frame_count, frame, score, args.frame_display_width, args.frame_display_height)

        output_rtsp.update(frame)

        print("frame time", time.time()-frame_time, " timestamp", timestamp)

    input_rtsp.stop()
    input_rtsp.join()
    
    output_rtsp.stop()
    output_rtsp.join()

if __name__ == "__main__":
    main()
