UBUNTU

Follow the install instructions here: https://github.com/NVlabs/spacemouse-extension (incl pip install hidapi)
Debugging, for me this helped: https://github.com/JakubAndrysek/PySpaceMouse and https://github.com/JakubAndrysek/PySpaceMouse/blob/master/troubleshooting.md

Also, you need to run the collect_demonstration.py script with sudo, otherwise SpaceMouse is not recognized: sudo $(which python) scripts/collect_demonstrations.py ...

Example:
sudo $(which python) scripts/collect_demonstration.py \
                       --controller OSC_POSE \
                       --camera agentview --robots Panda \
                       --num-demonstration 50 \
                       --rot-sensitivity 1.5 \
                       --bddl-file ./libero/libero/bddl_files/libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl


Also, search in the code base for "vendor" and "product", and replace the IDs with your spacemouse ids (lsusb shows them!)