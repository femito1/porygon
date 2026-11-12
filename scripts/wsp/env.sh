# source inside a job: gloria's venv (read-only) with this stream's src first
export WSP=/vol/bitbucket/fr1226/porygon-ws-wsp
export GLORIA=/vol/bitbucket/fr1226/gloria
export PYTHONPATH=$WSP/src:$WSP/scripts/wsp${PYTHONPATH:+:$PYTHONPATH}
export PATH=$GLORIA/.venv/bin:$GLORIA/.node/bin:$PATH
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
cd $WSP
