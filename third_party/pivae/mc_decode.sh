#!/bin/bash

jobname=${1}
jobid=${2}
sampling_num=${3}

echo "Make list of python script for mc decoding"
launch-mc-decode configure ${jobname} ${jobid} ${sampling_num}
chmod +w -R logs/${jobname}/${jobid}
echo 'Start mc decoding'
launch-mc-decode start ${jobname} ${jobid}
