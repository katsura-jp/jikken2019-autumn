# jikken2019-autumn
主専攻実験（秋）　強化学習

## Agent
### テーブルQ学習
### Actor-Critic

## Env
### [Pendulum-v0](https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py#L7)
- actionは[-2, 2]からサンプル
- stateは[角度,角速度]

## Memo
- 空間は`gym.spaces`の[Box](https://github.com/openai/gym/blob/master/gym/spaces/box.py)を使うといい