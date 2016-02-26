#!/bin/bash

maxiter=${1}
taul=${2}
etal=${3}
taub=${4}
etab=${5}
prep=${6}
mdir='combo_'${prep}'_'${taul}'_'${etal}'_'${taub}'_'${etab}'_'${maxiter}
mkdir ${mdir}
cp submitcomboprep.sh ${mdir} 
cd ${mdir} 

#rec=${qfile}' '${cfile}' '${trtfile}' '${vqfile}' '${vcfile}' '${vtfile}' '${lc}' '${tau}' '${maxiter}' '${reg}' '${regtype}' '${st}
rec=${maxiter}' '${taul}' '${etal}' '${taub}' '${etab}' '${prep}
qsub submitcomboprep.sh ${rec} > temp
#sleep 5
#prnt=$(tail temp)
#pid=$(tail temp | awk '{print $3}') 
#cmdpid=${pid}' = '${rec}' = '${mdir}
#echo $cmdpid >> ../processlog.txt 
