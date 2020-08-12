# Imitating Unknown Policies via Exploration (IUPE)
Official Pytorch implementation of Imitating Unknown Policies via Exploration

---

Behavioral cloning is an imitation learning technique that teaches an agent how to behave through expert demonstrations.
Recent approaches use self-supervision of fully-observable unlabeled snapshots of the states to decode state-pairs into actions.
However, the iterative learning scheme from these techniques are prone to getting stuck into bad local minima.
We address these limitations incorporating a two-phase model into the original framework, which learns from unlabeled observations via exploration, substantially improving traditional behavioral cloning by exploiting (i) a sampling mechanism to prevent bad local minima, (ii) a sampling mechanism to improve exploration, and (iii) self-attention modules to capture global features.

Imitating Unknown Policies via Exploration (IUPE) combines both an *Inverse Dynamics Model* (IDM) to infer actions in a self-supervised fashion, and a *Policy Model* (PM), which is a function that tells the agent what to do in each possible state of the environment. IUPE further augments the Behavioral Cloning from Observations framework with two strategies for avoiding local minima, sampling and exploration, and with self-attention modules for improving the learning of global features and, hence, generalization.

<p align="center">
  <img src="https://github.com/NathanGavenski/IUPE/blob/code/images/iupe_flow.svg" width="75%" />
</p>

<br><br>

## Downloading the data
You can download the data we used to train our models [here](https://drive.google.com/file/d/1_wnrfv1OEM_EuPaF5tMF2l2ZJjr9lJVh/view?usp=sharing)

## Training IUPE

After downloading the expert demonstration, you can then train ABCO. There are several training scripts in the directory. 
```
./scripts/iupe_3  # Maze 3x3
./scripts/iupe_5  # Maze 5x5
./scripts/iupe_10  # Maze 10x10
./scripts/iupe_acrobot  # Acrobot
./scripts/iupe_cartpole  # Cartpole
./scripts/iupe_mountaincar  # Mountaincar
```
**We ran IUPE on a server, if you are running locally you might want to remove** ```xvfb-run -a -s "-screen 0 1400x900x24"``` **from the scripts**

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
