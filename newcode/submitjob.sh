#!/bin/bash 
#$ -cwd
#$ -o qsubout
#$ -e qsuberr
#$ -V
#$ -r y
echo JOB_ID=$JOB_ID
echo "Running job = ${JOB_ID} on `hostname` at `date`"
#echo $maxiter $taul $etal $taub $etab $prep
echo $maxiter $taul $etal $prep
#python combo.py $maxiter $taul $etal $taub $etab $prep >> job_${maxiter}_${taul}_${etal}_${taub}_${etab}_${prep}.txt 2>&1 
python billogistic.py $maxiter $taul $etal $prep >> job_${maxiter}_${taul}_${etal}_${prep}.txt 2>&1


