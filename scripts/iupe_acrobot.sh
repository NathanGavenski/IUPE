xvfb-run -a -s "-screen 1 1400x900x24" python train.py \
--gpu 2 \
--pretrained \
--encoder vector \
--run_name alpha_mlp_acrobot_7_2 \
--data_path ./dataset/acrobot/IDM_VECTOR/ \
--expert_path ./dataset/acrobot/POLICY_VECTOR/ \
--alpha ./dataset/acrobot/ALPHA/ \
--domain acrobot_vector \
\
--lr 7e-2 \
--lr_decay_rate 1 \
--batch_size 128 \
--idm_epochs 100 \
\
--policy_lr 7e-2 \
--policy_lr_decay_rate 1 \
--policy_batch_size 128 \
\
--verbose
