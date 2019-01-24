#!/bin/bash 
echo '==> Count samples of each category:'$1;
for i in `seq $2 $3`;do
  echo -n $i,
  cat $1|grep " $i$"|wc -l;    # last dollar to grep \n
done
echo '==> ...Done'
