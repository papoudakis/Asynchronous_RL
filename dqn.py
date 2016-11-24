import sys
import gym
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from gym.spaces import Box, Discrete
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import RMSprop
import numpy as np
import random
import copy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import sqrt
from keras.initializations import zero
import threading
import gym_pull
from environment import DoomEnv
import time
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('game', 'ppaquette/DoomBasic-v0', 'Name of the Doom game to play.')
flags.DEFINE_integer('num_concurrent', 8, 'Number of concurrent actor-learner threads to use during training.')
flags.DEFINE_integer('tmax', 80000000, 'Number of training timesteps.')
flags.DEFINE_integer('width', 84, 'Scale screen to this width.')
flags.DEFINE_integer('height', 84, 'Scale screen to this height.')
flags.DEFINE_integer('history_length', 4, 'Use this number of recent screens as the environment state.')
flags.DEFINE_integer('network_update_frequency', 32, 'Frequency with which each actor learner thread does an async gradient update')
flags.DEFINE_integer('target_network_update_frequency', 10000, 'Reset the target network every n timesteps')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_integer('anneal_epsilon_timesteps', 1000000, 'Number of timesteps to anneal epsilon.')
flags.DEFINE_string('checkpoint_dir', '/tmp/checkpoints/', 'Directory for storing model checkpoints')
flags.DEFINE_boolean('show_training', True, 'If true, have gym render evironments during training')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_string('checkpoint_path', 'path/to/recent.ckpt', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_integer('num_eval_episodes', 100, 'Number of episodes to run gym evaluation.')
flags.DEFINE_integer('checkpoint_interval', 600,'Checkpoint the model (i.e. save the parameters) every n ')
FLAGS = flags.FLAGS


T = 0
TMAX = FLAGS.tmax

def create_model(num_actions, agent_history_length, resized_width, resized_height):
    with tf.device("/cpu:0"):
        state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
        inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
        model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs)
        model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(model)
        model = Flatten()(model)
        model = Dense(output_dim=256, activation='relu')(model)
        q_values = Dense(output_dim=num_actions, activation='linear')(model)
        m = Model(input=inputs, output=q_values)
    return state, m

class DQN:
    
    def __init__(self, num_actions):
        
        self.TMAX = TMAX
        self.T = 0
        print "start object"
        g = tf.Graph()
        with g.as_default(), tf.Session() as self.session:
            
            K.set_session(self.session)
            self.create_operations(num_actions)
            self.saver = tf.train.Saver()
        


        # episodes global counters
        
            if FLAGS.testing:
                self.test(num_actions)
            else:
                self.train(num_actions)
                
    def create_operations(self, num_actions):

        # create model and state
        self.state, self.model = create_model(num_actions, FLAGS.history_length, FLAGS.width, FLAGS.height)

        # parameters of the model
        self.model_params = self.model.trainable_weights

        # create target network
        self.new_state, self.target_model = create_model(num_actions,  FLAGS.history_length, FLAGS.width, FLAGS.height)

        # parameters of the target model
        self.target_model_params = self.target_model.trainable_weights

        # operation for q values
        self.q_values = self.model(self.state)
         
        # operation for q values of target mdoel
        self.target_q_values = self.target_model(self.new_state)

        # operation for updating target's parameters
        self.update_target = [self.target_model_params[i].assign(self.model_params[i]) for i in range(len(self.target_model_params))]

        print "creating operations"
        # operations for training model

        # placeholder for actions
        self.actions = tf.placeholder("float", [None, num_actions])

        # placeholder for targets
        self.targets = tf.placeholder("float", [None])

        # multiple q values with actions. actions is an array with all zeros
        # except the value of the action which executed.
        # so action_q_values has only the qvalue with the same index
        # as the action that executed
        action_q_values = tf.reduce_sum(tf.mul(self.q_values, self.actions), reduction_indices=1)

        # define cost
        cost = tf.reduce_mean(tf.square(self.targets - action_q_values))

        # define variable learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        # define optimazation method
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # define traininf function
        self.grad_update = optimizer.minimize(cost, var_list=self.model_params)

    def sample_final_epsilon(self):
        possible_epsilon = [0.1]*4 + [0.5]*3 + [0.01]*3
        return random.choice(possible_epsilon)
    
    def actor_learner_thread(self, env, thread_id, num_actions):

        # create instance of Doom environment
        env = DoomEnv(env, FLAGS.width, FLAGS.height, FLAGS.history_length)

        # Initialize network gradients
        states = []
        actions = []
        targets = []

        initial_epsilon = 1
        epsilon = 1
        final_epsilon = self.sample_final_epsilon()    
        print 'Starting thread ' + str(thread_id) + ' with final epsilon ' + str(final_epsilon)

        time.sleep(3*thread_id)
        t = 0

        while self.T < self.TMAX:
        
            # Get initial game observation
            state = env.get_initial_state()
            done = False

            # episode's counter
            episode_reward = 0
            mean_q = 0
            frames = 0

            while not done:
                # forward pass of network. Get Q(s,a)
                q_values = self.q_values.eval(session = self.session, feed_dict = {self.state : [state]})

                # define list of actions. All values are zeros except , the
                # value of action that is executed
                action_list = np.zeros([num_actions])

                action_index = 0

                # chose action based on current policy
                if random.random() <= epsilon:
                    action_index = random.randrange(num_actions)
                else:
                    action_index = np.argmax(q_values)
                action_list[action_index] = 1

                # reduce epsilon
                if epsilon > final_epsilon:
                    epsilon -= (initial_epsilon - final_epsilon) / FLAGS.anneal_epsilon_timesteps

                # decrease learning rate
                if self.lr > 0:
                    self.lr -= FLAGS.learning_rate / self.TMAX
                    
                # Gym excecutes action in game environment on behalf of actor-learner
                new_state, reward, done = env.step(action_index)

                # forward pass of target network. Get Q(s',a)
                target_q_values = self.target_q_values.eval(session = self.session, feed_dict = {self.new_state : [new_state]})

                # clip reward to -1, 1
                clipped_reward = np.clip(reward, -1, 1)

                #compute targets based on Q-learning update rule
                # targets = r + gamma*max(Q(s',a))                
                if done:
                    targets.append(clipped_reward)
                else:
                    targets.append(clipped_reward + FLAGS.gamma * np.max(target_q_values))
    
                actions.append(action_list)
                states.append(state)
    
                # Update the state and global counters
                state = new_state
                self.T += 1
                t += 1

                # update episode's counter
                frames += 1
                episode_reward += reward
                mean_q += np.max(q_values)

                
                # update_target_network
                if self.T % FLAGS.target_network_update_frequency == 0:
                    self.session.run(self.update_target)
    
                # train online network
                if t % FLAGS.network_update_frequency == 0 or done:
                    if states:
                        self.session.run(self.grad_update, feed_dict = {self.state : states,
                                                          self.actions : actions,
                                                          self.targets :targets,
                                                          self.learning_rate: self.lr})
                    # Clear gradients
                    states = []
                    actions = []
                    targets = []
    
                # Save model progress
                if t % FLAGS.checkpoint_interval == 0:
                    self.saver.save(self.session, FLAGS.checkpoint_dir+"/" + FLAGS.game.split("/")[1] + ".ckpt" , global_step = t)

                # Print end of episode stats
                if done:
                    print "THREAD:", thread_id, "/ TIME", self.T, "/ TIMESTEP", t, "/ EPSILON", epsilon, "/ REWARD", episode_reward, "/ Q_MAX %.4f" % (mean_q/float(frames)), "/ EPSILON PROGRESS", t/float(FLAGS.anneal_epsilon_timesteps)
                    break





    def train(self, num_actions):

        # Initialize target network weights
        self.session.run(self.update_target)

        # inititalize learning rate
        self.lr = FLAGS.learning_rate
        
        # Set up game environments (one per thread)
        envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)]

        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        # Initialize variables
        self.session.run(tf.initialize_all_variables())


        # Start num_concurrent actor-learner training threads
        actor_learner_threads = [threading.Thread(target=self.actor_learner_thread, args=( envs[thread_id], thread_id, num_actions)) for thread_id in range(FLAGS.num_concurrent)]
        for t in actor_learner_threads:
            t.start()

        # Show the agents training and write summary statistics
        while True:
            if FLAGS.show_training:
                for env in envs:
                    env.render()
        
        for t in actor_learner_threads:
            t.join() 
    
    def test(self, num_actions):
        self.saver.restore(self.session, FLAGS.checkpoint_path)
        print "Restored model weights from ", FLAGS.checkpoint_path
        monitor_env = gym.make(FLAGS.game)
        monitor_env.monitor.start("/tmp/" + FLAGS.game ,force=True)
        env = DoomEnv(monitor_env, FLAGS.width, FLAGS.height, FLAGS.history_length)
   


        for i_episode in xrange(FLAGS.num_eval_episodes):
            state = env.get_initial_state()
            episode_reward = 0
            done = False
            while not done:
                monitor_env.render()
                q_values = self.q_values.eval(session = self.session, feed_dict = {self.state : [state]})
                action_index = np.argmax(q_values)
                new_state, reward, done = env.step(action_index)
                state = new_state
                episode_reward += reward
            print "Finished episode " + str(i_episode + 1) + " with score " + str(episode_reward)
        
        monitor_env.monitor.close()

def get_num_actions():
    env = gym.make(FLAGS.game)
    env = DoomEnv(env, FLAGS.width, FLAGS.height, FLAGS.history_length)
    num_actions = len(env.gym_actions)
    return num_actions

def main():

    num_actions = get_num_actions()
    DQN(num_actions)

if __name__ == "__main__":
    main()
