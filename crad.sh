conda activate CRAD
cd experiments/
bash train_torch.sh config.yaml 1 1 1111 1
bash eval_torch.sh config.yaml 1 1 1111 1