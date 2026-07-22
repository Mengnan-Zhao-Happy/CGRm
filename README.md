# Confusion-Geometry Rebalancing for Long-Tailed AT

## Dataset building:

> python make_cifar10_lt.py --data_dir .\data --output_dir .\data\CIFAR10-LT-IR50 --ir 50 --seed 1
> python make_tinyimagenet_lt.py --data_dir .\data\tiny-imagenet-200 --output_dir .\data\TinyImageNet-LT-IR10 --ir 10 --seed 1 

## Partial Comparative Methods

> python RoBal_LT.py  --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\RoBal_TRADES_LT" --base_algorithm trades --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

> python AT-BSL_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\AT_BSL_LT" --base_algorithm at --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

> python REAT_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\REAT_LT" --base_algorithm at --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

> python TAET_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\TAET_LT" --base_algorithm at --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

## Base Learner: AT, on CIFAR10-LT

> python AT_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\AT_LT" --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

> python UDR_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\UDR_AT_LT" --base_algorithm at --lamda_init 1.0 --lamda_lr 0.02 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

> python CFA_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\CFA_AT_LT" --base_algorithm at --cfa_begin 10 --cfa_lambda1 0.5 --cfa_lambda2 0.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

> python DAFA_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\DAFA_AT_LT" --base_algorithm at --dafa_lambda 1.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

> python RobustLT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\RobustLT_AT" --base_algorithm at --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

> python CGR_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\CGR_LT_AT" --base_algorithm at --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --feedback_start 10  --weight_lambda 1.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 0.05 --margin_m 0.5 --graph_topk 3 --overwrite

## Evaluation 

> python EVAL_CW_AA_LT.py --data_root ./data/TinyImageNet-LT-IR10 --checkpoint ./model_output/tiny-AWP-wide/AWP_LT/best.pt --model wrn-28-10 --attack both


