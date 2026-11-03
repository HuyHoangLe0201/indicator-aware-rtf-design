#!/bin/bash
# campaigns on five further public cell sets, both channels (about half an hour in parallel)
cd "$(dirname "$0")"
run() { CH=$2 NSPLIT=16 NDRAW=8 python bl_campaign.py $1 $3 2>&1 | grep -v -i warn > bl_campaign_$1_$2.txt; }
run HUST dq 6,10,16,24 & run HUST capacity 6,10,16,24 &
run Tongji dq 6,10,16,24 & run Tongji capacity 6,10,16,24 &
run RWTH dq 4,6,10,14 & run RWTH capacity 4,6,10,14 &
run Stanford dq 4,6,10,12 & run Stanford capacity 4,6,10,12 &
run MICH dq 4,6,10,12 & run MICH capacity 4,6,10,12 &
wait
