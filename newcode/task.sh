#!/bin/bash 

name=prep_${4}.`date "+%d%H%M%S"`
iter=${1}
tl=${2}
el=${3}
prp=${4}

#use qacct -j jobid to know about the job - memory+cpu-usage.

qsub -q medium -l h_vmem=4G -r y -R y -N ${name} -v maxiter=${iter},taul=${tl},etal=${el},prep=${prp} submitjob.sh 

