xvfb-run -a -s "-screen 1 1400x900x24" python train.py \
--gpu 3 \
--pretrained \
--encoder vector \
--run_name alpha_mlp_mountaincar_1 \
--data_path ./dataset/mountaincar/IDM_VECTOR/ \
--expert_path ./dataset/mountaincar/POLICY_VECTOR/ \
--alpha ./dataset/mountaincar/ALPHA/ \
--domain mountaincar_vector \
\
--lr 5e-1 \
--lr_decay_rate 1 \
--batch_size 128 \
--idm_epochs 100 \
\
--policy_lr 8e-1 \
--policy_lr_decay_rate 1 \
--policy_batch_size 128 \
\
--verbose
