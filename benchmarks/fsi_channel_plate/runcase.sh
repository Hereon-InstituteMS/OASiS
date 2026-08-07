#!/bin/bash
# $1 = case tag
SD=/tmp/claude-1001/-home-alexander-4C/b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/fourc_fsi
T=$1
rm -rf $SD/run_$T; mkdir -p $SD/run_$T; cd $SD/run_$T
S=$(date +%s)
LD_LIBRARY_PATH=/opt/4C-dependencies/lib stdbuf -oL /home/alexander/4C/build/4C $SD/$T.4C.yaml $SD/run_$T/$T > $SD/run_$T/$T.log 2>&1
RC=$?
E=$(date +%s)
echo "RUN_EXIT=$RC WALL=$((E-S))s" > $SD/run_$T/status.txt
grep -c "processor 0 finished normally" $SD/run_$T/$T.log >> $SD/run_$T/status.txt
if [ $RC -eq 0 ]; then
  LD_LIBRARY_PATH=/opt/4C-dependencies/lib stdbuf -oL /home/alexander/4C/build/post_processor --filter=vtu \
     --postprocessor_deprecation_warning_off --file=$SD/run_$T/$T --output=$SD/run_$T/pp </dev/null > $SD/run_$T/pp.log 2>&1
  echo "PP_EXIT=$?" >> $SD/run_$T/status.txt
  /home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python $SD/extract.py --meta $SD/${T}_meta.json \
     --rundir $SD/run_$T --prefix $T --pp pp --json $SD/ref_$T.json > $SD/run_$T/extract.txt 2>&1
  echo "EX_EXIT=$?" >> $SD/run_$T/status.txt
fi
