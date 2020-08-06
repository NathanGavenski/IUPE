xvfb-run -a -s "-screen 2 1400x900x24" python train.py \
--gpu 1 \
--pretrained \
--encoder attention \
--run_name alpha_attention_10x10_2 \
--data_path ./dataset/maze/IDM/maze10/ \
--alpha ./dataset/maze/POLICY/alpha_10/ \
--maze_type all \
--maze_size 10 \
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
