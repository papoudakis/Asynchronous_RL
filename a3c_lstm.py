import sys
import gym
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Convolution2D, TimeDistributed, LSTM
from keras import backend as K
import numpy as np
import random
import threading
import time
from environment import Env
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('game', 'ppaquette/DoomBasic-v0', 'Name of the Doom game to play.')
flags.DEFINE_integer('num_concurrent', 8, 'Number of concurrent actor-learner threads to use during training.')
flags.DEFINE_integer('tmax', 80000000, 'Number of training timesteps.')
flags.DEFINE_integer('width', 84, 'Scale screen to this width.')
flags.DEFINE_integer('height', 84, 'Scale screen to this height.')
flags.DEFINE_integer('history_length', 4, 'Use this number of recent screens as the environment state.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('gamma', 0.99, 'Reward discount rate.')
flags.DEFINE_float('decay', 0.99,'Decay of RMSProp Optimizer ')
flags.DEFINE_float('BETA', 0.01, 'factor of regularazation.')
flags.DEFINE_string('checkpoint_dir', '/tmp/checkpoints/', 'Directory for storing model checkpoints')
flags.DEFINE_boolean('show_training', True, 'If true, have gym render evironments during training')
flags.DEFINE_boolean('testing', False, 'If true, run gym evaluation')
flags.DEFINE_string('checkpoint_path', 'path/to/recent.ckpt', 'Path to recent checkpoint to use for evaluation')
flags.DEFINE_integer('num_eval_episodes', 100, 'Number of episodes to run gym evaluation.')
flags.DEFINE_integer('checkpoint_interval', 600,'Checkpoint the model (i.e. save the parameters) every n ')
flags.DEFINE_string('game_type', 'Doom','Doom or atari game')
FLAGS = flags.FLAGS


T = 0
TMAX = FLAGS.tmax

t_max = 5

def create_model(num_actions, agent_history_length, resized_width, resized_height):
    with tf.device("/cpu:0"):
        state = tf.placeholder("float", [None, t_max, agent_history_length, resized_width, resized_height])
        inputs = Input(shape=(t_max, agent_history_length, resized_width, resized_height))
        model = TimeDistributed(Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4,4), activation='relu'))(inputs)
        model = TimeDistributed(Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2,2), activation='relu'))(model)
        model = TimeDistributed(Flatten())(model)
        model = TimeDistributed(Dense(output_dim=256, activation='relu'))(model)
        model = LSTM(256, return_sequences= False)(model)
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

class A3C_LSTM:
    
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

        # create local networks and states and parameter updates
        self.local_states = []
        self.local_p_model = []
        self.local_v_model = []
        self.p_params = []
        self.v_params = []
        self.local_policy = []
        self.local_value = []
        self.update_policy = []
        self.update_value = []
        
        for i in range(FLAGS.num_concurrent):
            s, v, p = create_model(num_actions, FLAGS.history_length, FLAGS.width, FLAGS.height)
            self.local_states.append(s)
            self.local_v_model.append(v)
            self.local_p_model.append(p)
            self.p_params.append(p.trainable_weights)
            self.v_params.append(v.trainable_weights)
            self.local_policy.append(p(s))
            self.local_value.append(v(s))
            self.update_policy.append([self.p_params[i][j].assign(self.policy_model_params[j]) for j in range(len(self.p_params[i]))])
            self.update_value.append([self.v_params[i][j].assign(self.value_model_params[j]) for j in range(len(self.v_params[i]))])

        print "creating operations"
        # operations for training model

        # placeholder for actions
        self.actions = tf.placeholder("float", [None,  num_actions])

        # placeholder for targets
        self.targets = tf.placeholder("float", [None])

        #compute advantage
        advantage = self.targets - self.value

        # now we will compute the cost for policy networkc which is:
        # log(policy(a|s, theta) )*(R - value(s, theta')) + b*entropy

        # compute log probs
        log_probs = tf.log(tf.clip_by_value(self.policy_values, 1e-20, 1.0))

        # compute entropy
        entropy = -tf.reduce_sum(self.policy_values * log_probs)

        # policy network loss 
        p_loss = -tf.reduce_sum(tf.reduce_sum(log_probs * self.actions, [1]) * tf.stop_gradient(advantage)) - FLAGS.BETA * entropy

        # value network loss
        v_loss = tf.reduce_sum(tf.square(advantage))

        # total loss
        cost = p_loss + 0.5 * v_loss

        # define optimazation method
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

        # define traininf function
        self.grad_update = optimizer.minimize(cost)

    def actor_learner_thread(self, env, thread_id, num_actions):

        # create instance of Doom environment
        env = Env(env, FLAGS.width, FLAGS.height, FLAGS.history_length, FLAGS.game_type)
          
        print 'Starting thread ' + str(thread_id)
        time.sleep(3*thread_id)

        # get initial state
        state = env.get_initial_state()

        # episode's counter
        episode_reward = 0
        frames = 0
        counter = 0

        # create sequence of states
        state_sequence = np.zeros((t_max, FLAGS.history_length, FLAGS.width, FLAGS.height))
        state_sequence[t_max -1, :, :, :] = state
        
        # define args of grad_update function
        states = []
        actions = []
        while self.T < self.TMAX:
            
            done = False
            # clear previous rewards
            prev_reward = []
            
            t = 0
            t_start = t
            
            # synchronize policy and value network
            self.session.run(self.update_policy[thread_id])
            self.session.run(self.update_value[thread_id])
            
            while not (done or ((t - t_start)  == t_max)):
                
                # forward pass of network. Get probability of every action
                probs = self.session.run(self.local_policy[thread_id], feed_dict={self.local_states[thread_id]: [state_sequence]})[0]

                # define list of actions. All values are zeros except , the
                # value of action that is executed
                action_list = np.zeros([num_actions])

                # choose action index based on policy
                action_index = sample_policy_action(num_actions, probs)
                action_list[action_index] = 1

                # add states and actions to gradients
                actions.append(action_list)
                states.append(state_sequence)

                # Gym excecutes action in game environment on behalf of actor-learner
                new_state, reward, done = env.step(action_index)

                # clip reward to -1, 1
                clipped_reward = np.clip(reward, -1, 1)
                prev_reward.append(clipped_reward)

                # decrease learning rate
                if self.lr > 0:
                    self.lr -= FLAGS.learning_rate / self.TMAX

                # update state
                state = new_state
                
                # add state to sequence
                state_sequence = np.delete(state_sequence, 0, 0)
                state_sequence = np.insert(state_sequence, t_max-1, state, 0)
                
                # Update global counters
                self.T += 1
                t += 1
                counter += 1

                # update episode's counter
                frames += 1
                episode_reward += reward
                
                # Save model progress
                if counter % FLAGS.checkpoint_interval == 0:
                    if FLAGS.game_type == 'Doom':
                        self.saver.save(self.session, FLAGS.checkpoint_dir+"/" + FLAGS.game.split("/")[1] + ".ckpt" , global_step = counter)
                    else:
                        self.saver.save(self.session, FLAGS.checkpoint_dir+"/" + FLAGS.game + ".ckpt" , global_step = counter)

            if done:
                R_t = 0
            else:
                R_t = self.session.run(self.local_value[thread_id], feed_dict = {self.local_states[thread_id] : [state_sequence]})[0][0]

            targets = np.zeros((t - t_start))
            for i in range(t - t_start -1 , -1, -1):
                R_t = prev_reward[i] + FLAGS.gamma * R_t
                targets[i] = R_t


            #update q value network
            self.session.run(self.grad_update, feed_dict = {self.state: states,
                                                          self.actions: actions,
                                                          self.targets: targets})
            # clear gradients    
            actions = []
            states = []
                
            if done:
                print "THREAD:", thread_id, "/ TIME", self.T, "/ TIMESTEP", counter, "/ REWARD", episode_reward
                episode_reward = 0
                frames = 0
                state = env.get_initial_state()

                # clear state sequence
                state_sequence = np.zeros((t_max, FLAGS.history_length, FLAGS.width, FLAGS.height))
                state_sequence[t_max-1, :, :, :] = state

                

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

        # Show the agents training
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
        env = Env(env, FLAGS.width, FLAGS.height, FLAGS.history_length, FLAGS.game_type)
        
        for i_episode in xrange(FLAGS.num_eval_episodes):
            state = env.get_initial_state()
            episode_reward = 0
            done = False
            
            # create state sequence
            state_sequence = np.zeros((t_max, FLAGS.history_length, FLAGS.width, FLAGS.height))
            state_sequence[t_max -1, :, :, :] = state
            
            while not done:
                monitor_env.render()
                q_values = self.q_values.eval(session = self.session, feed_dict = {self.state : [state_sequence]})
                action_index = np.argmax(q_values)
                new_state, reward, done = env.step(action_index)
                state = new_state

                # update state sequence
                state_sequence = np.delete(state_sequence, 0, 0)
                state_sequence = np.insert(state_sequence, t_max-1, state, 0)
                episode_reward += reward
            print "Finished episode " + str(i_episode + 1) + " with score " + str(episode_reward)
        
        monitor_env.monitor.close()

def get_num_actions():
    env = gym.make(FLAGS.game)
    env = Env(env, FLAGS.width, FLAGS.height, FLAGS.history_length, FLAGS.game_type)
    num_actions = len(env.gym_actions)
    return num_actions


def main():

    num_actions = get_num_actions()
    A3C_LSTM(num_actions)

if __name__ == "__main__":
    main()
