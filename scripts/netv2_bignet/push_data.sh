#!/bin/bash
# Cluster side of the BIGNET staging (runs on the gpucluster2 login node under nohup/setsid; started by
# stage_vm.sh): rsync P4's training + val shards, BC-P4 and the BC code of record straight to the GPU Dojo VM, then
# write ~/bignet/data/PUSH_DONE there (run_vm.sh's data stage waits for it and checks every md5).
# Direct route because the laptop's uplink measured 0.19 MB/s (11-17), i.e. ~11 h for 7.6 GB through a laptop pipe.
#   bash push_data.sh VM_HOST        (key ~/.ssh/porygon_vm_push, authorised on the VM by stage_vm.sh)
set -uo pipefail
VM=${1:?azureuser@host}
KEY=$HOME/.ssh/porygon_vm_push
D=/vol/bitbucket/fr1226/porygon-ws-data/datasets/frozen/p4-gen9-20261116-1502
C=/vol/bitbucket/fr1226/porygon-ws-netv2
SSH="ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30"
L=$C/bignet/push
mkdir -p "$L"
grep -E '  v2pre-human-shuf/(shard-(val-)?[0-9]+\.npz|meta\.json|heldout-games-0\.06\.txt|verify\.json)$' "$D/MD5SUMS" > "$L/MD5SUMS.shuf"
[ "$(grep -c shard- "$L/MD5SUMS.shuf")" = 533 ] || { echo "MD5SUMS.shuf has not 533 shards"; exit 1; }
cut -c35- "$L/MD5SUMS.shuf" > "$L/files.txt"
t0=$(date +%s)
echo "$(date -u +%FT%TZ) push start to $VM: $(wc -l < "$L/files.txt") files"
$SSH "$VM" 'mkdir -p ~/bignet/data/p4 ~/bignet/nets'
ok=0
for try in 1 2 3 4 5 6; do
  rsync -a --partial --timeout=300 -e "$SSH" "$C/src-p2b" "$VM:bignet/" \
  && rsync -a --partial --timeout=300 -e "$SSH" "$C/runs/pretrain/P4-g9-s2.pt" "$VM:bignet/nets/" \
  && rsync -a --partial --timeout=300 -e "$SSH" "$L/MD5SUMS.shuf" "$VM:bignet/data/p4/" \
  && rsync -a --partial --timeout=300 --info=stats1 -e "$SSH" --files-from="$L/files.txt" "$D/" "$VM:bignet/data/p4/" \
  && { ok=1; break; }
  echo "$(date -u +%FT%TZ) rsync try $try failed; retry in 60 s"; sleep 60
done
[ $ok = 1 ] || { echo "$(date -u +%FT%TZ) push FAILED"; exit 1; }
dt=$(( $(date +%s) - t0 ))
bytes=$(cd "$D" && xargs du -cb < "$L/files.txt" | tail -1 | cut -f1)
msg="$(date -u +%FT%TZ) $bytes bytes in $dt s from $(hostname)"
$SSH "$VM" "echo '$msg' > ~/bignet/data/PUSH_DONE"
echo "$msg: PUSH_DONE written"
