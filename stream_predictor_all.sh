MATCHID=01
CAMLABEL=camera-20
TEAMID=Notre-Dame-1
RSTP_URL=rtsp://localhost:8554

TIMESTAMP=$(date +'%Y-%m-%d_%H-%M')

MODELNAME=droid
python stream_predictor.py \
    --input_rtsp ${RSTP_URL}/${CAMLABEL}  \
    --output_rtsp ${RSTP_URL}/output__${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}  \
    --weights_path ${MODELNAME}_model.pth \
    --output_fname output/${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}.csv &

PID1=$!

MODELNAME=ngebm
python stream_predictor.py \
    --input_rtsp ${RSTP_URL}/${CAMLABEL}  \
    --output_rtsp ${RSTP_URL}/output__${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}  \
    --weights_path ${MODELNAME}_model.pth \
    --output_fname output/${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}.csv & 

PID2=$!

# Sleep for match duration of 7 minutes (420 seconds) + 14 seconds
#sleep 420s
sleep 74s

# Kill the processes
kill -9 $PID1 $PID2