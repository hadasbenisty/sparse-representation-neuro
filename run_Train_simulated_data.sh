#!/usr/bin/bash
# Set the location of the anaconda python
export PATH="/Local/md_benisty/anaconda3/bin:$PATH"
# Navigate to the script directory
cd /srv01/technion/hadasbe/scripts/sparse_representation/sparse-representation-neuro
# Run the python script with the specified parameters
echo "this is the shell script" echo running with "$@"
/Local/md_benisty/anaconda3/bin/python spike_detection_by_CRASE_on_simulated.py "$@"

