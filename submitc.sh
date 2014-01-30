#!/bin/sh

#$ -l h_vmem=50G
#$ -V
#$ -S /bin/bash
#$ -o /home/usuaris/pranava/bmaps/new/experiments/qsublogs
#$ -e /home/usuaris/pranava/bmaps/new/experiments/qsublogs 
#$ -cwd

~/bin/python rccombonew.py $1 $2 $3 $4 $5 $6 $7 $8 

