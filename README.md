# jikken2019-autumn
2019年度　主専攻実験（秋）強化学習

## 環境
#### Pendulum-v0
- 状態空間の次元数: 3
- 行動空間の次元数: 1
#### BipedalWalker-v2
- 状態空間の次元数: 24
- 行動空間の次元数: 4
#### RoboschoolHumanoid-v1
- 状態空間の次元数: 44
- 行動空間の次元数: 17

## Docker
```
$ sudo docker build -t="<user>/jikken-autumn:latest" ./
$ docker run -it --rm --gpus all --name reinforce_learning --shm-size 16G -v $PWD:/home/ -w /home/ <user>/jikken-autumn:latest /bin/bash
```

## 経験再生を使用しないテーブルQ学習
```shell
$ python table_q_learning.py --max-step 128000000 --save-step 128000 --eval-step 25600 --seeds 2
```

## 経験再生を使用するテーブルQ学習
```shell
$ python table_q_learning.py --er --seeds 2
```
## 経験再生とランダム行動の確率をアニーリングするテーブルQ学習
```shell
$ python table_q_learning.py --seeds 2 --er --eps-annealing --eps-gamma 0.995
```

## Actor-Criticの学習
```
$ python actor_critic.py --eval-episodes 50 --seed 2 --device 0
```

## TD3の学習
```
" use all improvement
$ python td3.py --eval-episodes 50 --seed 2 --device 0
" drop Target Actor & Target Critic
$ python td3.py --eval-episodes 50 --seed 2 --device 0 --target-ac
" drop Target Policy Smoothing Regularization
$ python td3.py --eval-episodes 50 --seed 2 --device 0 --smooth-reg
" drop Delayed Policy Update
$ python td3.py --eval-episodes 50 --seed 2 --device 0 --delay-update
" drop Clipped Double Q-Learning 
$ python td3.py --eval-episodes 50 --seed 2 --device 0 --clip-double
```

## 任意の環境での学習
```shell
$ python actor_critic.py --device 0 --eval-episodes 50  --seed 2 --env BipedalWalker-v2 --save-step 300 --div-step
$ python td3.py --device 0 --eval-episodes 50  --seed 2 --env BipedalWalker-v2 --save-step 300 --div-step
```