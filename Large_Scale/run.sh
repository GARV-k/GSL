
python -u main_large.py --type_GCN 'structured' --dataset 'arxiv' --loader_params 100 --activation 'relu' --splits 5 --lr 0.01 --seed 0 --hidden_layers 64 --hidden_dim 256 --train_iter 3 --test_iter 10 --use_saved_model False --alpha 0.5 --theta 2 --dropout 0.6 --w_decay 0.0005 --device 'cuda:0' | tee test
