#！/bin/bash
# bash save_cross_modality_images.sh

for ((i=1;i<50;i++))
do
        python save_cross_modality_images.py --model_prefix Flow2Flow-no_activation --epoch $i
done
