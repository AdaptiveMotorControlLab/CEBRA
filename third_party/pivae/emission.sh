#!/bin/bash

jobname=${1}
jobid=${2}
jobtype=${3}
echo "Make list of python script for emission"
launch-emission configure ${jobname} ${jobid} ${jobtype}
chmod +w -R logs/${jobname}/${jobid}
echo 'Start job'
launch-emission start ${jobname} ${jobid}
