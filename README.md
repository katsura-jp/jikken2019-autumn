# jikken2019-autumn
主専攻実験（秋）　強化学習

## Agent
### テーブルQ学習
### Actor-Critic

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

      

## Memo
- 空間は`gym.spaces`の[Box](https://github.com/openai/gym/blob/master/gym/spaces/box.py)を使うといい
  - `from gym.spaces import Box`