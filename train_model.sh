python train.py --data_path ../data/sysu \
                --model_path ./save_model \
                --log_path ./log \
                --total_epoch 50 \
                --save_epoch 2 \
                --gpu 1 \
                --batch_size 1 \
                --img_h 384 \
                --img_w 128 \
                --model_prefix Flow2Flow-384x128 \
                --log_file log-Flow2Flow-384x128.txt


