# Imitating Unknown Policies via Exploration (IUPE)
Official Pytorch implementation of [Imitating Unknown Policies via Exploration](https://arxiv.org/pdf/2008.05660.pdf)

---

Behavioral cloning is an imitation learning technique that teaches an agent how to behave through expert demonstrations.
Recent approaches use self-supervision of fully-observable unlabeled snapshots of the states to decode state-pairs into actions.
However, the iterative learning scheme from these techniques are prone to getting stuck into bad local minima.
We address these limitations incorporating a two-phase model into the original framework, which learns from unlabeled observations via exploration, substantially improving traditional behavioral cloning by exploiting (i) a sampling mechanism to prevent bad local minima, (ii) a sampling mechanism to improve exploration, and (iii) self-attention modules to capture global features.

Imitating Unknown Policies via Exploration (IUPE) combines both an *Inverse Dynamics Model* (IDM) to infer actions in a self-supervised fashion, and a *Policy Model* (PM), which is a function that tells the agent what to do in each possible state of the environment. IUPE further augments the Behavioral Cloning from Observations framework with two strategies for avoiding local minima, sampling and exploration, and with self-attention modules for improving the learning of global features and, hence, generalization.

<p align="center">
  <img src="https://github.com/NathanGavenski/IUPE/blob/master/images/iupe_flow_diagram.svg" width="75%" />
</p>

<br><br>

## Downloading the data
You can download the data we used to train our models [here](https://drive.google.com/file/d/1_wnrfv1OEM_EuPaF5tMF2l2ZJjr9lJVh/view?usp=sharing).

## Training IUPE

After downloading the expert demonstration, you can then train IUPE. There are several training scripts in the directory. 
```
./scripts/iupe_3  # Maze 3x3
./scripts/iupe_5  # Maze 5x5
./scripts/iupe_10  # Maze 10x10
./scripts/iupe_acrobot  # Acrobot
./scripts/iupe_cartpole  # Cartpole
./scripts/iupe_mountaincar  # Mountaincar
```
**We ran IUPE on a server, if you are running locally you might want to remove** ```xvfb-run -a -s "-screen 0 1400x900x24"``` **from the scripts**.

## Results
Performance and Average Episode Reward for our approach and related work:

| Models | Metrics | <a href="https://gym.openai.com/envs/CartPole-v0/">CartPole</a> | <a href="https://gym.openai.com/envs/Acrobot-v1/">Acrobot</a> | <a href="https://gym.openai.com/envs/MountainCar-v0/">MountainCar</a> | <a href="https://github.com/MattChanTK/gym-maze">Maze 3x3</a> | <a href="https://github.com/MattChanTK/gym-maze">Maze 5x5</a> | <a href="https://github.com/MattChanTK/gym-maze">Maze 10x10</a> |
| :-----: | :------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: | :------------------------------------------------------: |
| <b>Expert</b> | <i>P</i><br> <i>AER</i> | 1.000<br>442.628 | 1.000<br>-110.109 | 1.000<br>-147.265 | 1.000<br>0.963 | 1.000<br>0.970 | 1.000<br>0.981 |
| <b>Random</b> | <i>P</i><br> <i>AER</i> | 0.000<br>18.700 | 0.000<br>-482.600 | 0.000<br>-200.000 | 0.000<br>0.557 | 0.000<br>0.166 | 0.000<br>-0.415 | 
| <b>BC</b> | <i>P</i><br> <i>AER</i> | 1.135<br>500.000 | 1.071<br>-83.590 | 1.560<br>-117.720 | -1.207<br>0.180 | -0.921<br>-0.507 | -0.470<br>-1.000 | 
| <b><a href="https://arxiv.org/abs/1805.01954">BCO</a></b> | <i>P</i><br> <i>AER</i> | <b>1.135</b><br><b>500.000</b> | 0.980<br>-117.600 | 0.948<br>-150.000 | 0.883<br><b>0.927</b> | -0.112<br>0.104 | -0.416<br>-0.941 | 
| <b><a href="https://arxiv.org/abs/1805.07914">ILPO</a></b> | <i>P</i><br> <i>AER</i> | <b>1.135</b><br><b>500.000</b> | 1.067<br>-85.300 | 0.626<br>-167.000 | -1.711<br>-0.026 | -0.398<br>-0.059 | 0.257<br>-0.020 | 
  | <b><a href="https://arxiv.org/abs/2008.05660">IUPE</a></b> | <i>P</i><br> <i>AER</i> | <b>1.135</b><br><b>500.000</b> | <b>1.086</b><br><b>-78.100</b> | <b>1.314</b><br><b>-130.700</b> | <b>1.361</b><br><b>0.927</b> | <b>1.000</b><br><b>0.971</b> | <b>1.000</b><br><b>0.981</b> | 


## Citation
```
@inproceedings{GavenskiEtAl2020bmvc,
  author    = {Gavenski, Nathan and
               Monteiro, Juarez and 
               Granada, Roger and 
               Meneguzzi, Felipe and 
               Barros, Rodrigo C.},
  title     = {Imitating Unknown Policies via Exploration},
  booktitle = {Proceedings of the 31st British Machine Vision Conference},
  series    = {BMVC 2020},
  location  = {Manchester, UK},
  pages     = {1--8},
  url       = {},
  month     = {September},
  year      = {2020},
  publisher = {BMVA Press}
}
```
