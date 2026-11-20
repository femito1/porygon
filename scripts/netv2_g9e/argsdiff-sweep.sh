#!/bin/bash
# argsdiff sweep v20 -> v21 over every mode and both trees (current, now)
cd /home/nand/Documents/personal-projects/ShowdownAgent/.claude/worktrees/ws-netv2
export NV2_PARK_HOLD_ROUNDS=1 NV2_G9T_HOLD_ROUNDS=1  # refused first starts in the fake tree exit 3 instead of spinning
O=scripts/netv2_g9q/netv2.env.v20; N=scripts/netv2_g9e/netv2.env.v21; R=/home/nand/.cache/nv2s/g9e/argsdiff
for what in current now; do
  for m in g9 g9p g9s g9q g9p2 g9r g9t g9q2 g9qbf g9qent g9p4 g9e; do
    r=$(ARGSDIFF_NO_AS=1 ARGSDIFF_ROOT=$R ARGSDIFF_G9_MODE=$m bash scripts/netv2_swap/argsdiff.sh $O $N $what 2>&1 | head -n 1)
    echo "$what g9=$m: $r"
  done
  for m in r1b g9 g9s g9p g9q g9t; do
    r=$(ARGSDIFF_NO_AS=1 ARGSDIFF_ROOT=$R ARGSDIFF_R1_MODE=$m bash scripts/netv2_swap/argsdiff.sh $O $N $what 2>&1 | head -n 1)
    echo "$what r1=$m: $r"
  done
done
ARGSDIFF_NO_AS=1 ARGSDIFF_ROOT=$R ARGSDIFF_G9_MODE=g9e bash scripts/netv2_swap/argsdiff.sh $O $N now > /home/nand/.cache/nv2s/g9e/argsdiff-g9e-now.txt 2>&1
