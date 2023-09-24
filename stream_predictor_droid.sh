MATCHID=01
CAMLABEL=camera-20
TEAMID=Notre-Dame-1

TIMESTAMP=$(date +'%Y-%m-%d_%H-%M')
RSTP_URL=rtsp://localhost:8554

MODELNAME=droid

INPUT_RSTP=${RSTP_URL}/${CAMLABEL}
OUTPUT_RSTP=${RSTP_URL}/output__${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}
OUTPUT_FNAME=${MATCHID}__${TEAMID}__${CAMLABEL}__${MODELNAME}__${TIMESTAMP}.csv

python stream_predictor.py \
    --input_rtsp ${INPUT_RSTP}  \
    --output_rtsp ${OUTPUT_RSTP}  \
    --weights_path ${MODELNAME}_model.pth \
    --time_limit 2 \
    --output_fname output/${OUTPUT_FNAME}