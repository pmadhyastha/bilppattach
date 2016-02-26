#!/bin/bash

maxiter=${1}
tau=${2}
eta=${3}
prep=${4}
mdir='bil_'${prep}'_'${tau}'_'${eta}'_'${maxiter}
mkdir ${mdir}
cp submitbil.sh ${mdir} 
cd ${mdir} 

#rec=${qfile}' '${cfile}' '${trtfile}' '${vqfile}' '${vcfile}' '${vtfile}' '${lc}' '${tau}' '${maxiter}' '${reg}' '${regtype}' '${st}
rec=${maxiter}' '${tau}' '${eta}' '${prep}
qsub submitbil.sh ${rec} > temp
#sleep 7
#prnt=$(tail temp)
#pid=$(tail temp | awk '{print $3}') 
#cmdpid=${pid}' = '${rec}' = '${mdir}
#echo $cmdpid >> ../processlog.txt 
