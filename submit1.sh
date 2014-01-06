#!/bin/sh

#$ -l h_vmem=30G
#$ -V
#$ -S /bin/bash
#$ -o /home/usuaris/pranava/bmaps/new/experiments/qsublogs
#$ -e /home/usuaris/pranava/bmaps/new/experiments/qsublogs 
#$ -cwd

~/bin/python rccombo.py $1 'l2pnn'
