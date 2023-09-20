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


def overlay_score_and_display(image_np, score, frame_display_width, frame_display_height):

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

    # Display the frame
    # TODO: refactory frame name 
    cv2.imshow('ND BlueTeam 1 - Model 1', image_np)


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

    _ = predict_impath(model, 'sample_img.png', im_size)
    
    #cv2.imshow('ND-BT-1 M1', np.zeros((frame_display_height, frame_display_width, 3), dtype=np.uint8))
    time.sleep(2)
    
    # Open the stream 
    cap_time = time.time()
    cap = cv2.VideoCapture(args.stream_url) 

    if not cap.isOpened():
        print("Error: Could not open stream.")
        exit()

    print("cap time", time.time()-cap_time)
    frame_count = 0
    while True:

        frame_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't read frame.")
            break        

        # Make prediction
        frame_count += 1
        
        score = predict(model, frame, im_size)
        #score = 0

        # Write predictions for RAITE ouput format 
        write_to_csv(frame_count, score, raite_output_path)        

        # Display the frame with boxes
        overlay_score_and_display(frame, score, args.frame_display_width, args.frame_display_height)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

        print("frame time", time.time()-frame_time)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
