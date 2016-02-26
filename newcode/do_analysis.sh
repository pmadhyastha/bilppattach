for f in fobos_nn_*; do echo -ne $f' = ' ;cat $f/output.txt |awk 'BEGIN{max=0}{if($13>max)max=$13}END{print max}'; done > flsl2onlys.txt
cat flsl2onlys.txt | sort -s -n -r -k3 > flsnnonly.txt
for f in fobos_l2_*; do echo -ne $f' = ' ;cat $f/output.txt |awk 'BEGIN{max=0}{if($13>max)max=$13}END{print max}'; done > flsl2onlys.txt
cat flsl2onlys.txt | sort -s -n -r -k3 > flsl2only.txt
for f in fobos_l1_*; do echo -ne $f' = ' ;cat $f/output.txt |awk 'BEGIN{max=0}{if($13>max)max=$13}END{print max}'; done > flsl2onlys.txt
cat flsl2onlys.txt | sort -s -n -r -k3 > flsl1only.txt
echo 'nn' >> bestfile.txt
nnfle=`head -n 1 flsnnonly.txt | awk '{print $1}'`
cat /home/usuaris/pranava/bmaps/nnmin/python/processlog.txt | grep $nnfle >> bestfile.txt
echo 'l2' >> bestfile.txt
l2fle=`head -n 1 flsl2only.txt | awk '{print $1}'`
cat /home/usuaris/pranava/bmaps/nnmin/python/processlog.txt | grep $l2fle >> bestfile.txt
echo 'l1' >> bestfile.txt
l1fle=`head -n 1 flsl1only.txt | awk '{print $1}'`
cat /home/usuaris/pranava/bmaps/nnmin/python/processlog.txt | grep $l1fle >> bestfile.txt
