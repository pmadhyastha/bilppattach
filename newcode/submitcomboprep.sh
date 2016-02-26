#!/bin/bash
#$ -cwd
#$ -l h_vmem=4G
#$ -q short
#$ -o /home/usuaris/pranava/acl2016/shorts/ppattach/newcode/qsublogs
#$ -e /home/usuaris/pranava/acl2016/shorts/ppattach/newcode/qsublogs 
#$ -V 

maxiter=${1}
taul=${2}
etal=${3}
taub=${4}
etab=${5}
prep=${6}

python -u /home/usuaris/pranava/acl2016/shorts/ppattach/newcode/comboprep.py ${maxiter} ${taul} ${etal} ${taub} ${etab} ${prep} > output.txt 2>&1 

