# CUDA_VISIBLE_DEVICES=0 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/base_all_real -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 39 --reliability 1.0 --lambda_view 0.1  --start_epoch 100

CUDA_VISIBLE_DEVICES=0 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/base_0.9reliability_ours -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 50 --reliability 0.9 --lambda_view 0.1  --start_epoch 100
CUDA_VISIBLE_DEVICES=0 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/base_0.9reliability_0.1viewloss -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 50 --reliability 0.9 --lambda_view 0.1  --start_epoch 0
# CUDA_VISIBLE_DEVICES=3 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/test_not_update_tw_0.9_proxy_0.5weights -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 50 --reliability 0.8 --lambda_view 0.1  --start_epoch 0
# CUDA_VISIBLE_DEVICES=3 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/test_not_update_tw_0.9_proxy_0.5weights -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 50 --reliability 0.8 --lambda_view 0.1  --start_epoch 0
CUDA_VISIBLE_DEVICES=2 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/base_0.9reliability_update_all -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 50 --reliability 0.9 --lambda_view 0.1  --start_epoch 100


# CUDA_VISIBLE_DEVICES=2 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/base_all_real -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 39 --reliability 1.0 --lambda_view 0.1  --start_epoch 100
# CUDA_VISIBLE_DEVICES=2 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/base_0.9reliability -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 50 --reliability 0.9 --lambda_view 0.1  --start_epoch 100
# CUDA_VISIBLE_DEVICES=2 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/base_0.9reliability_0.1viewloss -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 50 --reliability 0.9 --lambda_view 0.1  --start_epoch 0
# # CUDA_VISIBLE_DEVICES=3 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/test_not_update_tw_0.9_proxy_0.5weights -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 50 --reliability 0.8 --lambda_view 0.1  --start_epoch 0
# # CUDA_VISIBLE_DEVICES=3 python examples/vdg_train_usl_aug2_proxy_debug.py  -d Market_view --logs-dir logs/test_not_update_tw_0.9_proxy_0.5weights -b 64 --num-instance 8 --eps 0.6 --lr 0.00035 --epochs 50 --reliability 0.8 --lambda_view 0.1  --start_epoch 0
