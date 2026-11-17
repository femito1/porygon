#!/bin/bash
# Laptop side of BIGNET (run once, right after the user starts the VM from the GPU Dojo page; ~5 min of laptop time,
# nothing large stored or relayed by the laptop):
#  1. copy the three VM scripts (~40 KB) to ~/bignet on the VM;
#  2. authorise a cluster-side push key on the VM (created on the cluster if missing; `restrict` = no tty/forwarding);
#  3. start push_data.sh on the gpucluster2 login node (nohup setsid): cluster -> VM rsync of the data, BC-P4, src;
#  4. install + start the systemd unit (run_vm.sh run), which waits for PUSH_DONE, then trains, evaluates and shuts
#     the VM down. After a stop/start the unit resumes by itself.
#   bash scripts/netv2_bignet/stage_vm.sh [VM]
set -euo pipefail
VM=${1:-azureuser@gpudojo-fernando-rimoli26-88caa466d2ef.polandcentral.cloudapp.azure.com}
HERE=$(cd "$(dirname "$0")" && pwd)
C=/vol/bitbucket/fr1226/porygon-ws-netv2
ssh -o BatchMode=yes -o ConnectTimeout=20 "$VM" 'nvidia-smi --query-gpu=name --format=csv,noheader; df -h / | tail -1'
ssh -o BatchMode=yes "$VM" 'mkdir -p ~/bignet/logs'
scp -q "$HERE/run_vm.sh" "$HERE/pretrain_resumable.py" "$HERE/microbench.py" "$VM:bignet/"
ssh -o BatchMode=yes gpucluster2 "mkdir -p $C/bignet && cd $C/bignet && cat > push_data.sh" < "$HERE/push_data.sh"
PUB=$(ssh -o BatchMode=yes gpucluster2 '[ -f ~/.ssh/porygon_vm_push ] || ssh-keygen -q -t ed25519 -N "" -C porygon-vm-push -f ~/.ssh/porygon_vm_push; cat ~/.ssh/porygon_vm_push.pub')
ssh -o BatchMode=yes "$VM" "grep -qF '${PUB#* }' ~/.ssh/authorized_keys 2>/dev/null || echo 'restrict $PUB' >> ~/.ssh/authorized_keys"
ssh -o BatchMode=yes gpucluster2 "cd $C/bignet && nohup setsid bash push_data.sh $VM > push.log 2>&1 < /dev/null & sleep 20; tail -3 $C/bignet/push.log"
ssh -o BatchMode=yes "$VM" 'bash ~/bignet/run_vm.sh install && sudo systemctl start porygon-bignet && sleep 5 && bash ~/bignet/run_vm.sh status'
echo "staged. Push log: ssh gpucluster2 tail $C/bignet/push.log ; VM: ssh $VM bash ~/bignet/run_vm.sh status"
