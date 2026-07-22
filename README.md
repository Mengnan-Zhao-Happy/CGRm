## Confusion-Geometry Rebalancing for Long-Tailed Adversarial Training

# Dataset building:

python make_cifar10_lt.py --data_dir .\data --output_dir .\data\CIFAR10-LT-IR50 --ir 50 --seed 1 

python make_cifar100_lt.py `
  --data_dir ".\data\cifar-data100" `
  --output_dir ".\data\CIFAR100-LT-IR10" `
  --ir 10 `
  --seed 1


  python make_tinyimagenet_lt.py --data_dir .\data\tiny-imagenet-200 --output_dir .\data\TinyImageNet-LT-IR10 --ir 10 --seed 1 



AT

CIFAR 10 LT 上的测试

python AT_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\AT_LT" --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python UDR_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\UDR_AT_LT" --base_algorithm at --lamda_init 1.0 --lamda_lr 0.02 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python CFA_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\CFA_AT_LT" --base_algorithm at --cfa_begin 10 --cfa_lambda1 0.5 --cfa_lambda2 0.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python DAFA_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\DAFA_AT_LT" --base_algorithm at --dafa_lambda 1.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python RobustLT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\RobustLT_AT" --base_algorithm at --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python CGR_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\CGR_LT_AT" --base_algorithm at --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --feedback_start 10  --weight_lambda 1.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 0.05 --margin_m 0.5 --graph_topk 3 --overwrite


CIFAR 100 LT 上的测试

python AT_LT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\AT_LT" --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python UDR_LT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\UDR_AT_LT" --base_algorithm at --lamda_init 1.0 --lamda_lr 0.02 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python CFA_LT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\CFA_AT_LT" --base_algorithm at --cfa_begin 10 --cfa_lambda1 0.5 --cfa_lambda2 0.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python DAFA_LT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\DAFA_AT_LT" --base_algorithm at --dafa_lambda 1.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python RobustLT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\RobustLT_AT" --base_algorithm at --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python CGR_LT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\CGR_LT_AT" --base_algorithm at --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --feedback_start 10  --weight_lambda 1.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 0.05 --margin_m 0.5 --graph_topk 3 --overwrite



TinyImageNet LT 上的测试

python AT_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\TinyImageNet\AT_LT" --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python UDR_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\TinyImageNet\UDR_AT_LT" --base_algorithm at --lamda_init 1.0 --lamda_lr 0.02 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python CFA_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\TinyImageNet\CFA_AT_LT" --base_algorithm at --cfa_begin 10 --cfa_lambda1 0.5 --cfa_lambda2 0.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python DAFA_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\TinyImageNet\DAFA_AT_LT" --base_algorithm at --dafa_lambda 1.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python RobustLT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\TinyImageNet\RobustLT_AT" --base_algorithm at --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python CGR_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\TinyImageNet\CGR_LT_AT" --base_algorithm at --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --feedback_start 10  --weight_lambda 1.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 0.05 --margin_m 0.5 --graph_topk 3 --overwrite






python AWP_LT.py `
  --data_root ".\data\CIFAR10-LT-IR50" `
  --dataset auto `
  --model wrn-28-10 `
  --model_dir ".\model_output\cifar10-AWP-wide\AWP_LT" `
  --base_algorithm trades `
  --epochs 110 `
  --eval_freq 10 `
  --pgd_num_steps 10 `
  --test_pgd_num_steps 20 `
  --batch_size 128 `
  --test_batch_size 200 `
  --awp_gamma 0.01 `
  --awp_warmup 10 `
  --overwrite



AWP

CIFAR 10 LT 上的测试

python UDR_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model wrn-28-10 --model_dir ".\model_output\cifar10-AWP-wide\UDR_LT" --base_algorithm AWP --lamda_init 1.0 --lamda_lr 0.02 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite 

python CFA_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model wrn-28-10 --model_dir ".\model_output\cifar10-AWP-wide\CFA_LT" --base_algorithm AWP --cfa_begin 10 --cfa_lambda1 0.5 --cfa_lambda2 0.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python DAFA_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model wrn-28-10 --model_dir ".\model_output\cifar10-AWP-wide\DAF_LT" --base_algorithm AWP --dafa_lambda 1.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python RobustLT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model wrn-28-10 --model_dir ".\model_output\cifar10-AWP-wide\RobustLT_AT" --base_algorithm AWP --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python CGR_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model wrn-28-10 --model_dir ".\model_output\cifar10-AWP-wide\CGR_AT" --base_algorithm AWP --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --feedback_start 10  --weight_lambda 1.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 2 --margin_m 0.5 --graph_topk 3 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite


CIFAR 100 LT 上的测试

python AWP_LT.py `
  --data_root ".\data\CIFAR100-LT-IR10" `
  --dataset auto `
  --model resnet `
  --model_dir ".\model_output\cifar100\AWP_LT" `
  --base_algorithm AWP `
  --epochs 110 `
  --eval_freq 10 `
  --pgd_num_steps 10 `
  --test_pgd_num_steps 20 `
  --batch_size 128 `
  --test_batch_size 200 `
  --awp_gamma 0.01 `
  --awp_warmup 10 `
  --overwrite



python UDR_LT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\UDR_LT" --base_algorithm AWP --lamda_init 1.0 --lamda_lr 0.02 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite 

python CFA_LT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\CFA_LT" --base_algorithm AWP --cfa_begin 10 --cfa_lambda1 0.5 --cfa_lambda2 0.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python DAFA_LT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\DAFA_LT" --base_algorithm AWP --dafa_lambda 1.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python RobustLT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\RobustLT" --base_algorithm AWP --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python CGR_LT.py --data_root ".\data\CIFAR100-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\cifar100\CGR_LT" --base_algorithm AWP --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --feedback_start 10  --weight_lambda 1.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 0.05 --margin_m 0.5 --graph_topk 3 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite




Tiny LT 上的测试

python AWP_LT.py `
  --data_root ".\data\TinyImageNet-LT-IR10" `
  --dataset auto `
  --model resnet `
  --model_dir ".\model_output\tiny\AWP_LT" `
  --base_algorithm at `
  --epochs 110 `
  --eval_freq 10 `
  --pgd_num_steps 10 `
  --test_pgd_num_steps 20 `
  --batch_size 128 `
  --test_batch_size 200 `
  --awp_gamma 0.01 `
  --awp_warmup 10 `
  --overwrite


python UDR_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\tiny\UDR_LT" --base_algorithm AWP --lamda_init 1.0 --lamda_lr 0.02 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite 

python CFA_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\tiny\CFA_LT" --base_algorithm AWP --cfa_begin 10 --cfa_lambda1 0.5 --cfa_lambda2 0.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python DAFA_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\tiny\DAF_LT" --base_algorithm AWP --dafa_lambda 1.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python RobustLT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model wrn-28-10 --model_dir ".\model_output\tiny-wide\RobustLT_AT" --base_algorithm AWP --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python CGR_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\tiny\CGR_AT" --base_algorithm AWP --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --feedback_start 10  --weight_lambda 1.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 0.05 --margin_m 0.5 --graph_topk 3 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python AWP_LT.py `
  --data_root ".\data\TinyImageNet-LT-IR10" `
  --dataset auto `
  --model resnet `
  --model_dir ".\model_output\tiny\AWP_LT" `
  --base_algorithm at `
  --epochs 110 `
  --eval_freq 10 `
  --pgd_num_steps 10 `
  --test_pgd_num_steps 20 `
  --batch_size 128 `
  --test_batch_size 200 `
  --awp_gamma 0.01 `
  --awp_warmup 10 `
  --overwrite


Robal

python RoBal_LT.py `
  --data_root ".\data\CIFAR10-LT-IR50" `
  --dataset auto `
  --model resnet `
  --model_dir ".\model_output\cifar10\RoBal_TRADES_LT" `
  --base_algorithm trades `
  --epochs 110 `
  --eval_freq 10 `
  --pgd_num_steps 10 `
  --test_pgd_num_steps 20 `
  --batch_size 128 `
  --test_batch_size 200 `
  --overwrite


  CIFAR 10 LT 上的测试

python UDR_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\UDR_LT" --base_algorithm RoBal --lamda_init 1.0 --lamda_lr 0.02 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite 

python CFA_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\CFA_LT" --base_algorithm RoBal --cfa_begin 10 --cfa_lambda1 0.5 --cfa_lambda2 0.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python DAFA_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\tiny\DAF_LT" --base_algorithm RoBal --dafa_lambda 1.5 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python RobustLT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\tiny\RobustLT_AT" --base_algorithm RoBal --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite

python CGR_LT.py --data_root ".\data\TinyImageNet-LT-IR10" --dataset auto --model resnet --model_dir ".\model_output\tiny\CGR_AT" --base_algorithm RoBal --robustlt_alpha 0.3 --robustlt_beta 0.8 --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --feedback_start 10  --weight_lambda 1.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 0.05 --margin_m 0.5 --graph_topk 3 --awp_gamma 0.01 --awp_warmup 10 --awp_lr 0.01 --overwrite


cd C:\Users\Administrator\Desktop\FAT\DAFA-master\DAFA-master

python AT-BSL_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\AT_BSL_LT" --base_algorithm at --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python REAT_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\REAT_LT" --base_algorithm at --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite

python TAET_LT.py --data_root ".\data\CIFAR10-LT-IR50" --dataset auto --model resnet --model_dir ".\model_output\cifar10\TAET_LT" --base_algorithm at --epochs 110 --eval_freq 10 --pgd_num_steps 10 --test_pgd_num_steps 20 --batch_size 128 --test_batch_size 200 --overwrite











tail_vicinal_lambda 会严重损害尾部类的鲁棒性
python CARE_V2_LT.py --data_root ./data/CIFAR100-LT-IR10 --model_dir ./model_output/care_v2_D1_tv02 --overwrite --tail_vicinal_lambda 0.2
TEST: Clean(all) 50.10%, Robust(all) 18.98%, Clean(tail) 47.30%, Robust(tail) 17.86%
python CARE_V2_LT.py --data_root ./data/CIFAR100-LT-IR10 --model_dir ./model_output/care_v2_D2_notv --overwrite --tail_vicinal_lambda 0.0
 TEST: Clean(all) 49.95%, Robust(all) 20.57%, Clean(tail) 47.19%, Robust(tail) 19.25%

Cifar10
python CARE_V2_LT.py --data_root ./data/CIFAR10-LT-IR50 --model_dir ./model_output/care_v2_D2_notv --overwrite --tail_vicinal_lambda 0.0
 TEST: Clean(all) 66.08%, Robust(all) 30.74%, Clean(tail) 60.69%, Robust(tail) 21.64%


python CARE_V2_LT.py --data_root ./data/CIFAR10-LT-IR50 --model_dir ./model_output/care_v2_robust --epochs 110 --eval_freq 10  --beta 4.0  --adv_ramp_beta 0.8 --care_start 10 --feedback_start 10 --tail_vicinal_start 5  --tail_vicinal_adv_start 20  --tail_vicinal_lambda 0.0 --overwrite
TEST: Clean(all) 68.56%, Robust(all) 32.95%, Clean(tail) 62.28%, Robust(tail) 23.23%

python CARE_V2_LT.py --data_root ./data/CIFAR10-LT-IR50 --model_dir ./model_output/care_v2_robust --epochs 110 --eval_freq 10  --beta 2.0  --adv_ramp_beta 0.8 --care_start 10 --feedback_start 10 --tail_vicinal_start 5  --tail_vicinal_adv_start 20  --tail_vicinal_lambda 0.0 --overwrite --eta_robust_eps 1.0
TEST: Clean(all) 70.96%, Robust(all) 31.14%, Clean(tail) 64.95%, Robust(tail) 20.96%


python CARE_V2_LT.py --data_root ./data/CIFAR10-LT-IR50 --model_dir ./model_output/care_v2_robust --epochs 110 --eval_freq 10  --beta 6.0  --adv_ramp_beta 0.8 --care_start 10 --feedback_start 10 --tail_vicinal_start 5  --tail_vicinal_adv_start 20  --tail_vicinal_lambda 0.0 --overwrite --eta_robust_eps 1.0
TEST: Clean(all) 64.06%, Robust(all) 31.89%, Clean(tail) 57.56%, Robust(tail) 22.76%

python CARE_V2_LT.py --data_root ./data/CIFAR100-LT-IR10 --model_dir ./model_output/care_v2_robust --epochs 110 --eval_freq 10  --beta 6.0  --adv_ramp_beta 0.8 --care_start 10 --feedback_start 10 --tail_vicinal_start 5  --tail_vicinal_adv_start 20  --tail_vicinal_lambda 0.0 --overwrite --eta_robust_eps 1.0
TEST: Clean(all) 48.46%, Robust(all) 21.38%, Clean(tail) 45.61%, Robust(tail) 20.16%
python CARE_V2_LT.py --data_root ./data/CIFAR100-LT-IR10 --model_dir ./model_output/care_v2_robust --epochs 110 --eval_freq 10  --beta 4.0  --adv_ramp_beta 0.8 --care_start 10 --feedback_start 10 --tail_vicinal_start 5  --tail_vicinal_adv_start 20  --tail_vicinal_lambda 0.0 --overwrite --eta_robust_eps 1.0
TEST: Clean(all) 49.53%, Robust(all) 20.70%, Clean(tail) 46.70%, Robust(tail) 19.44%





68.29% 62.39% 34.92% 26.36%
wo feedback weight
python CGR_LT.py --data_root ".\data\CIFAR10-LT-IR50" --model_dir ".\model_output\ablation\CGR_wo_feedback_weight" --model resnet --base_algorithm trades --robustlt_alpha 0.3 --robustlt_beta 0.8 --feedback_start 10 --eval_freq 10 --weight_lambda 0.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 0.05 --margin_m 0.5 --graph_topk 3 --overwrite
TEST: Clean(all) 68.27%, Robust(all) 34.78%, Clean(tail) 62.01%, Robust(tail) 25.04%
w/o class-wise beta
TEST: Clean(all) 67.78%, Robust(all) 34.47%, Clean(tail) 61.51%, Robust(tail) 24.96%
w/o confusion-geometry margin
TEST: Clean(all) 67.92%, Robust(all) 35.41%, Clean(tail) 61.67%, Robust(tail) 25.11%
wo balanced CE
TEST: Clean(all) 54.10%, Robust(all) 31.38%, Clean(tail) 43.56%, Robust(tail) 18.49%


python CGR_LT.py --data_root ".\data\CIFAR10-LT-IR50" --model_dir ".\model_output\ablation\CGR_wo_feedback_weight" --model resnet --base_algorithm trades --robustlt_alpha 0.3 --robustlt_beta 0.8 --feedback_start 10 --eval_freq 10 --weight_lambda 0.0 --conf_lambda 0.5 --beta_lambda 0.5 --margin_lambda 0.05 --margin_m 0.5 --graph_topk 3 --no_classwise_beta --no_cgr_margin --overwrite
TEST: Clean(all) 65.60%, Robust(all) 32.96%, Clean(tail) 58.79%, Robust(tail) 22.31%





48.14%, Robust(all) 21.31%, Clean(tail) 45.94%, Robust(tail) 20.55%
48.81% 46.80% 21.54% 20.73%

0.05 

0.1
TEST: Clean(all) 68.69%, Robust(all) 34.65%, Clean(tail) 62.69%, Robust(tail) 25.51%
0.5
TEST: Clean(all) 69.35%, Robust(all) 34.67%, Clean(tail) 63.59%, Robust(tail) 25.64%
1.0
TEST: Clean(all) 70.48%, Robust(all) 35.33%, Clean(tail) 65.05%, Robust(tail) 26.59%
1.5
TEST: Clean(all) 70.94%, Robust(all) 35.04%, Clean(tail) 65.60%, Robust(tail) 26.44%
2.0
TEST: Clean(all) 70.91%, Robust(all) 35.34%, Clean(tail) 65.80%, Robust(tail) 27.04%
2.5
TEST: Clean(all) 71.36%, Robust(all) 35.23%, Clean(tail) 66.20%, Robust(tail) 27.01%
3.0
TEST: Clean(all) 70.36%, Robust(all) 35.15%, Clean(tail) 65.34%, Robust(tail) 27.39%
4.0
TEST: Clean(all) 71.85%, Robust(all) 35.29%, Clean(tail) 67.25%, Robust(tail) 27.84%
5.0
TEST: Clean(all) 72.09%, Robust(all) 34.81%, Clean(tail) 67.51%, Robust(tail) 27.70%
6.0
TEST: Clean(all) 71.75%, Robust(all) 34.44%, Clean(tail) 67.10%, Robust(tail) 26.92%

cifar100
48.81% 46.80% 21.54% 20.73%
0.1 49.03\% & 46.67\% & 21.10\% & 20.10\% 
.5 48.33\% & 45.80\% & 20.65\% & 19.54\%
1 47.95\% & 45.66\% & 20.73\% & 19.97\%
3 47.08\% & 45.27\% & 19.66\% & 19.25\%
4 46.99\% & 45.15\% & 19.13\% & 18.95\%
5 46.88\% & 45.04\% & 18.61\% & 18.35\%




python EVAL_CW_AA_LT.py --data_root ./data/TinyImageNet-LT-IR10 --checkpoint ./model_output/tiny-AWP-wide/AWP_LT/best.pt --model wrn-28-10 --attack aa


python EVAL_CW_AA_LT.py --data_root ./data/TinyImageNet-LT-IR10 --checkpoint ./model_output/tiny-AWP-wide/AWP_LT/best.pt --model wrn-28-10 --attack cw


