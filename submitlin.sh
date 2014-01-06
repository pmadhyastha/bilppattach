#!/bin/sh

#$ -l h_vmem=20G
#$ -V
#$ -S /bin/bash
#$ -o /home/usuaris/pranava/bmaps/new/experiments/qsublogs
#$ -e /home/usuaris/pranava/bmaps/new/experiments/qsublogs 
#$ -cwd

~/bin/python rcmaxent.py $1 $2 $3 $4 $5 $6 

