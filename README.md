# jikken2019-autumn
主専攻実験（秋）　強化学習

## Env
### [Pendulum-v0](https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L7)

- Action

  - 次元数 : 1
    - 方向+強さ
  - 空間
    - 方向+強さ : [-2, 2]

- State

  - 次元数 : 3

    - [cos, sin, 角速度]

  - 空間

    - cos : [-1, 1]

    - sin : [-1, 1]

    - 角速度 $\dot{\theta}$ : [-8, 8]


## Docker
```
$ sudo docker build -t="<user>/jikken-autumn:latest" ./
$ docker run -it --rm --gpus all --name reinforce_learning --shm-size 16G -v $PWD:/home/ <user>/jikken-autumn:latest /bin/bash

# cd /home/
# pip install -r requirements.txt
```

## 課題1
```
$ python table_q_learning.py --max-step 128000000 --save-step 128000 --eval-step 25600 --seeds 2
```

## 課題2
```
$ python table_q_learning.py --er --seeds 2
```

## 課題3
```
$ python 
```

## 課題4
```
$ python
```