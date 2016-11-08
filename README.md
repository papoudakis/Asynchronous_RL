## Asynchronous_RL

This is an implementation of asynchronous reinforcement alogithms as described in https://arxiv.org/pdf/1602.01783.pdf. 
This implementation is for gym's doom environment.

There are 3 algorithms
* asynchronous one-step Q-learning
* asynchronous n-steps Q-learning
* ansynchronous advanced actor critic

## Requirements
* [tensorflow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)
* [gym](https://github.com/openai/gym#installation)
* [gym_pull](https://github.com/ppaquette/gym-pull)
* [gym_doom](https://github.com/ppaquette/gym-doom)
* [skimage](http://scikit-image.org/)
* [Keras](https://keras.io/)


## Execution
#Training
In order to execute one of the above algorithms, and train an agent in one of
the available environment, just run:
```
python dqn.py --game "ppaquette/DoomDefendCenter-v0"
```
This way the algorithm will use the default parameteres, running the environment in
8 different processes.
#Testing
In order to test an already trained agent just run
```
python dqn.py --game "ppaquette/DoomDefendCenter-v0" --testing True --checkpoint_path "path/to/parameters/"
```
## Resources
https://arxiv.org/pdf/1602.01783.pdf
The code of this repo is completely based on the code of this repo:
https://github.com/coreylynch/async-rl
https://webdocs.cs.ualberta.ca/~sutton/book/the-book-2nd.html
