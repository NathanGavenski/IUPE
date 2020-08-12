xvfb-run -a -s "-screen 1 1400x900x24" python train.py \
--gpu 2 \
--pretrained \
--encoder attention \
--run_name alpha_attention_3x3_1 \
--data_path ./dataset/maze/IDM/maze3/ \
--alpha ./dataset/maze/POLICY/alpha_3/ \
--maze_type all \
--maze_size 3 \
--idm_epochs 100 \
--domain maze \
\
--lr 5e-3 \
--lr_decay_rate 0.95 \
--batch_size 30 \
\
--policy_lr 1e-2 \
--policy_lr_decay_rate 0.99 \
--policy_batch_size 30 \
\
--verbose \
--no_hit
