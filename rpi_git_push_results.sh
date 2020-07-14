!/bin/sh

REPO_DIR=/home/pi/Documents/GitHub/EMMA_Chapelle_large_encrypted

cd ${REPO_DIR}

sudo git pull

sudo git add Data_online_mode/EMS_logs_Chapelle_Moudon_large_30sec.mat
sudo git add Data_online_mode/exit_loop_flag_Chapelle_Moudon_large.txt

sudo sudo git commit -m "From raspberry at Chapelle (large): results updated"
sudo git push