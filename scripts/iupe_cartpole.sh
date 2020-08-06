xvfb-run -a -s "-screen 1 1400x900x24" python train.py \
--gpu 3 \
--pretrained \
--encoder vector \
--run_name alpha_mlp_cartpole_1 \
--data_path ./dataset/cartpole/IDM_VECTOR/ \
--expert_path ./dataset/cartpole/POLICY_VECTOR/ \
--alpha ./dataset/cartpole/ALPHA/ \
--domain cartpole_vector \
\
--lr 1e-1 \
--lr_decay_rate 1 \
--batch_size 128 \
\
--policy_lr 1e-1 \
--policy_lr_decay_rate 1 \
--policy_batch_size 128 \
\
--verbose