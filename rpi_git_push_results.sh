!/bin/sh

REPO_DIR=/home/pi/Documents/GitHub/EMMA_Chapelle_encrypted

cd ${REPO_DIR}

git pull
git add Data_online_mode/EMS_logs_Zurich_30sec.mat

git add Data_online_mode/EMS_logs_Chapelle_Moudon_large_30sec.mat
git add Data_online_mode/exit_loop_flag_Chapelle_Moudon_large.txt

git add Data_online_mode/EMS_logs_Chapelle_Moudon_small_30sec.mat
git add Data_online_mode/exit_loop_flag_Chapelle_Moudon_small.txt

git commit -m "From raspberry: results updated"
git push