# RAITE 2023
TEAM_ID=Notre-Dame-1
MATCH_DURATION=7m20s
TIMESTAMP=$(date +'%Y-%m-%d_%H-%M')

# Configure before each match
#-------------------------------
MATCH_ID=01
CAM_LABEL=camera-20
RSTP_URL=rtsp://localhost:8554
#-------------------------------


TIMESTAMP=$(date +'%Y-%m-%d_%H-%M')

MODEL_NAME=droid
python stream_predictor.py \
    --input_rtsp ${RSTP_URL}/${CAM_LABEL}  \
    --output_rtsp ${RSTP_URL}/output__${MATCH_ID}__${TEAM_ID}__${CAM_LABEL}__${MODEL_NAME}__${TIMESTAMP}  \
    --weights_path ${MODEL_NAME}_model.pth \
    --output_fname output/${MATCH_ID}__${TEAM_ID}__${CAM_LABEL}__${MODEL_NAME}__${TIMESTAMP}.csv &

PID1=$!

MODEL_NAME=ngebm
python stream_predictor.py \
    --input_rtsp ${RSTP_URL}/${CAM_LABEL}  \
    --output_rtsp ${RSTP_URL}/output__${MATCH_ID}__${TEAM_ID}__${CAM_LABEL}__${MODEL_NAME}__${TIMESTAMP}  \
    --weights_path ${MODEL_NAME}_model.pth \
    --output_fname output/${MATCH_ID}__${TEAM_ID}__${CAM_LABEL}__${MODEL_NAME}__${TIMESTAMP}.csv & 

PID2=$!

# Sleep for match duration
sleep ${MATCH_DURATION}

# Kill the processes
kill -9 $PID1 $PID2