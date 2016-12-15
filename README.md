# Asynchronous_RL

This is an implementation of asynchronous reinforcement learning algorithms as described in https://arxiv.org/pdf/1602.01783.pdf. 
This implementation is for gym's doom and atari environment.

There are 4 algorithms
* asynchronous one-step Q-learning
* asynchronous n-steps Q-learning
* asynchronous advanced actor critic
* asynchronous LSTM advanced actor critic

## Requirements
* [tensorflow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)
* [gym](https://github.com/openai/gym#installation)
* [gym(Atari)](https://gym.openai.com/envs#atari)
* [gym_pull](https://github.com/ppaquette/gym-pull)
* [gym_doom](https://github.com/ppaquette/gym-doom)
* [skimage](http://scikit-image.org/)
* [Keras](https://keras.io/)


## Execution
###Training
In order to execute one of the above algorithms, and train an agent in one of
the available environment, just run:
```
python dqn.py --game "ppaquette/DoomDefendCenter-v0"
```

This way the algorithm will use the default parameteres, running the environment in
8 actor learner threads.

If you want to train in an atari game run the following command
```
python a3c_lstm.py --game "Breakout-v0" --game_type "Atari"
```
###Testing
In order to test an already trained agent just run
```
python dqn.py --game "ppaquette/DoomDefendCenter-v0" --testing True --checkpoint_path "path/to/parameters/"
```

## Results
Below there are some evaluation from gym openai

* Using one step dqn:

https://gym.openai.com/evaluations/eval_MGqu9wbTxS0fVFlz2puow

https://gym.openai.com/evaluations/eval_YB4PBRMQRWmWDW9eCXmV6g

* Using n steps dqn:

https://gym.openai.com/evaluations/eval_f8hCpqhQnqJEJCn3uiOWg

* Using a3c:

https://gym.openai.com/evaluations/eval_bxAN82ZRQe07kgfJTV5jA 
This evaluation is only for 10M steps. It can become a lot better with 80M steps training

## Resources
https://github.com/coreylynch/async-rl

https://github.com/muupan/async-rl/wiki

https://github.com/tatsuyaokubo/async-rl

https://arxiv.org/pdf/1602.01783.pdf

https://webdocs.cs.ualberta.ca/~sutton/book/the-book-2nd.html
