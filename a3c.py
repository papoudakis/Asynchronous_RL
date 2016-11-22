import sys
import gym
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from gym.spaces import Box, Discrete
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Convolution2D, TimeDistributed
from keras.layers.recurrent import LSTM
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
flags.DEFINE_integer('target_network_update_frequency', 40000, 'Reset the target network every n timesteps')
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_float('BETA', 0.0, 'factor of regularazation.')
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

t_max = 32

def create_model(num_actions, agent_history_length, resized_width, resized_height):
    with tf.device("/cpu:0"):
        state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
        inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
        model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu', border_mode='same')(inputs)
        model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu', border_mode='same')(model)
        model = Flatten()(model)
        model = Dense(output_dim=256, activation='relu')(model)
        q_values = Dense(output_dim=1, activation='linear')(model)
        p_values = Dense(output_dim=num_actions, activation='softmax')(model)
        value_model = Model(input=inputs, output=q_values)
        policy_model = Model(input=inputs, output=p_values)
    return state, value_model, policy_model

def sample_policy_action(num_actions, probs):
    """
    Sample an action from an action probability distribution output by
    the policy network.
    """
    # Subtract a tiny value from probabilities in order to avoid
    # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
    probs = probs - np.finfo(np.float32).epsneg

    histogram = np.random.multinomial(1, probs)
    action_index = int(np.nonzero(histogram)[0])
    return action_index

class A3C:
    
    def __init__(self, num_actions):
        
        self.TMAX = TMAX
        self.T = 0
        g = tf.Graph()
        with g.as_default(), tf.Session() as self.session:
            K.set_session(self.session)
            self.create_operations(num_actions)
            self.saver = tf.train.Saver()
        
            if FLAGS.testing:
                self.test(num_actions)
            else:
                self.train(num_actions)
                
    def create_operations(self, num_actions):

        # create model and state
        self.state, self.value_model, self.policy_model = create_model(
            num_actions, FLAGS.history_length, FLAGS.width, FLAGS.height)

        # parameters of the model
        self.value_model_params = self.value_model.trainable_weights

        # parameters of the target model
        self.policy_model_params = self.policy_model.trainable_weights

        # operation for q values
        self.value = self.value_model(self.state)
         
        # operation for policies
        self.policy_values = self.policy_model(self.state)

        print "creating operations"
        # operations for training model

        # placeholder for actions
        self.actions = tf.placeholder("float", [None, num_actions])

        # placeholder for targets
        self.targets = tf.placeholder("float", [None])

        #compute advantage
        advantage = self.targets - self.value
        print self.targets.get_shape()
        print self.value.get_shape()
        print advantage.get_shape()
        # now we will compute the cost for policy networkc which is:
        # log(policy(a|s, theta) )*(R - value(s, theta')) + b*entropy

        # compute log probs
        log_probs = tf.log(tf.clip_by_value(self.policy_values, 1e-20, 1.0))
        print log_probs.get_shape()
        print self.policy_values.get_shape()
        # compute entropy
        entropy = tf.reduce_sum(self.policy_values * log_probs, reduction_indices=1)
        print entropy.get_shape()
        # policy network loss 
        p_loss = -(tf.reduce_sum(tf.mul(log_probs, self.actions), reduction_indices=1) * tf.stop_gradient(advantage) + FLAGS.BETA * entropy)
        print p_loss.get_shape()
        # value network loss
        v_loss = tf.square(advantage)
        print v_loss.get_shape()
        # total loss
        cost = tf.reduce_mean(p_loss + 0.5 * v_loss)
        print cost.get_shape()

        # define variable learning rate
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # define optimazation method
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # define traininf function
        self.grad_update = optimizer.minimize(cost)


    def sample_final_epsilon(self):
        possible_epsilon = [0.1]*4000 + [0.5]*3000 + [0.01]*3000
        return random.choice(possible_epsilon)



    
    def actor_learner_thread(self, env, thread_id, num_actions):


        # create instance of Doom environment
        env = DoomEnv(env, FLAGS.width, FLAGS.height, FLAGS.history_length)


        
        initial_epsilon = 1
        epsilon = 1
        final_epsilon = self.sample_final_epsilon()    
        print 'Starting thread ' + str(thread_id) + ' with final epsilon ' + str(final_epsilon)

        time.sleep(3*thread_id)

        state = env.get_initial_state()

        # episode's counter
        episode_reward = 0
        mean_q = 0
        frames = 0
        counter = 0

        while self.T < self.TMAX:
        
            # Get initial game observation
            
            done = False

            
            

            # clear gradients
            states = []
            actions = []
            targets = []
            prev_reward = []

            t = 0
            t_start = t
            
            while not (done or ((t - t_start)  == t_max)):
                # forward pass of network. Get Q(s,a)
                probs = self.session.run(self.policy_values, feed_dict={self.state: [state]})[0]

                # define list of actions. All values are zeros except , the
                # value of action that is executed
                action_list = np.zeros([num_actions])

                action_index = 0

                # chose action based on current policy
                #~ if random.random() <= epsilon:
                    #~ action_index = random.randrange(num_actions)
                #~ else:
                action_index = sample_policy_action(num_actions, probs)
                action_list[action_index] = 1

                # add state and action to list
                actions.append(action_list)
                states.append(state)
                
                # reduce epsilon
                if epsilon > final_epsilon:
                    epsilon -= (initial_epsilon - final_epsilon) / FLAGS.anneal_epsilon_timesteps

                # Gym excecutes action in game environment on behalf of actor-learner
                new_state, reward, done = env.step(action_index)

                # clip reward to -1, 1
                clipped_reward = np.clip(reward, -1, 1)
                prev_reward.append(clipped_reward)

                # decrease learning rate
                if self.lr > 0:
                    self.lr -= FLAGS.learning_rate / self.TMAX

                # Update the state and global counters
                state = new_state
                self.T += 1
                t += 1
                counter += 1
                # update episode's counter
                frames += 1
                episode_reward += reward
    
    
                # Save model progress
                if counter % FLAGS.checkpoint_interval == 0:
                    self.saver.save(self.session, FLAGS.checkpoint_dir+"/" + FLAGS.game.split("/")[1] + ".ckpt" , global_step = counter)
                    
          
            if done:
                R_t = 0
            else:
                R_t = self.value.eval(session = self.session, feed_dict = {self.state : [state]})[0][0]

            targets = np.zeros((t - t_start))
                
            for i in range(t - t_start -1 , -1, -1):
                R_t = prev_reward[i] + FLAGS.gamma * R_t
                targets[i] = R_t


            #update q value network
            self.session.run(self.grad_update, feed_dict = {self.state: states,
                                                          self.actions: actions,
                                                          self.targets: targets,
                                                          self.learning_rate: self.lr})
                
            
                
            if done:
                print "THREAD:", thread_id, "/ TIME", self.T, "/ TIMESTEP", counter, "/ EPSILON", epsilon, "/ REWARD", episode_reward, "/ EPSILON PROGRESS", counter/float(FLAGS.anneal_epsilon_timesteps)
                episode_reward = 0
                frames = 0
                state = env.get_initial_state()





    def train(self, num_actions):

        # Set up game environments (one per thread)
        envs = [gym.make(FLAGS.game) for i in range(FLAGS.num_concurrent)]

        if not os.path.exists(FLAGS.checkpoint_dir):
            os.makedirs(FLAGS.checkpoint_dir)
        # Initialize variables
        self.session.run(tf.initialize_all_variables())

        # inititalize learning rate
        self.lr = FLAGS.learning_rate
        
        # Start num_concurrent actor-learner training threads
        actor_learner_threads = [threading.Thread(target=self.actor_learner_thread, args=( envs[thread_id], thread_id, num_actions)) for thread_id in range(FLAGS.num_concurrent)]
        for t in actor_learner_threads:
            t.start()

        # Show the agents training and write summary statistics
        while True:
            if FLAGS.show_training:
                for env in envs:
                #~ print "paparas"
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


def main():

    #~ num_actions = get_num_actions()
    A3C(3)

if __name__ == "__main__":
    main()
