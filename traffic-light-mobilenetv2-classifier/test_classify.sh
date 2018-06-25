#!/bin/bash
echo 'Classifying real_pic/0/left1575.jpg'
python classify_traffic_light_image.py --image ./left1575.jpg
echo 'Classifying sim_pic/0/sim_img_04238.jpg'
python classify_traffic_light_image.py --image ./sim_img_04238.jpg
