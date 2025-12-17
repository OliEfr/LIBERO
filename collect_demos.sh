sudo env PYTHONPATH="/home/admin_07/project_repos/LIBERO" /home/admin_07/miniconda3/envs/libero/bin/python /home/admin_07/project_repos/LIBERO/scripts/collect_demonstration.py \
	--controller OSC_POSE \
	--camera agentview \
	--robots Panda \
	--num-demonstration 10 \
	--rot-sensitivity 1.5 \
	--bddl-file ./libero/libero/bddl_files/libero_10/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl
