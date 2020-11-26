import gym
import os
import numpy as np
import argparse
from skimage.color import rgb2gray
import tensorflow as tf
import random
import network
import matplotlib.pyplot as plt
from copy import copy
import cv2
import time


def setup_options():
	parser = argparse.ArgumentParser(description='training options')
	parser.add_argument('--batch_size', type=int, default=32, help="batct size")
	parser.add_argument('--width', type=int, default=84, help="input width")
	parser.add_argument('--height', type=int, default=84, help="input height")
	parser.add_argument('--num_stack', type=int, default=3, help="frames stack")
	parser.add_argument('--hit_ratio', type=float, default=0.2, help="ratio of hit state in memory")		
	parser.add_argument('--memory_capacity', type=int, default=2000, help="memory capacity")
	parser.add_argument('--num_step', type=int, default=5000, help="time step of a eposide")
	parser.add_argument('--num_eposides', type=int, default=99999, help="number of eposide")
	parser.add_argument('--num_repeat', type=int, default=1, help="repeat dataset n times")
	parser.add_argument('--target_interval', type=int, default=10, help="saving model every log interval")
	parser.add_argument('--log_interval', type=int, default=1, help="saving model every log interval")
	parser.add_argument('--ckpt_dir', type=str, default='./tmp/ckpt', help='set checkpoint path')
	parser.add_argument('--not_restore', action='store_true', default=False, help='Not restore checkpoint')
	parser.add_argument('--render', action='store_true', default=False, help='enable rendering')
	parser.add_argument('--plot_input', action='store_true', default=False, help='enable show input')


	args = parser.parse_args()
	return args

class Env():
	def __init__(self, environment):
		self.env = gym.make(environment)
		self.reward_threshold = self.env.spec.reward_threshold

	def reset(self):
		return self.env.reset()

	def render(self):
		self.env.render()

	def step(self, action, time_step):
		obs, reward, done, _ = self.env.step(action)
		reward_new = reward
		if done:
			reward_new -= 20
		elif reward_new == 0:
			reward_new = -(time_step/500)
		return obs, reward_new, done, reward

class Agent():
	def __init__(self, global_step, args, network, scope):
		self.scope = scope
		self.memory_capacity = args.memory_capacity
		self.memory = np.dtype([('s', np.float64, (args.width, args.height, args.num_stack)), ('s_', np.float64, (args.width, args.height, args.num_stack)), ('a', np.float64), ('r', np.float64), ('d', np.bool)])
		self.memories = np.empty(self.memory_capacity, dtype=self.memory)
		self.batch_size = args.batch_size
		self.memory_counter = 0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.epsilon = tf.get_variable(name=self.scope+'_epsilon', dtype=tf.float32, initializer=tf.constant_initializer(0.95), shape=1)
		self.epsilon_step = self.epsilon.assign(self.epsilon * self.epsilon_decay)
		self.start_epoch = tf.get_variable(name=self.scope+'_start_epoch', dtype=tf.int32, initializer=tf.constant_initializer(0), shape=1)
		self.start_epoch_inc = self.start_epoch.assign(self.start_epoch+1)
		
		self.init_lr = 0.001
		self.decay_steps = 10
		self.decay_factor = 0.999
		self.action_space = 6
		self.num_repeat = args.num_repeat
		self.gamma = 0.85
		self.num_hit = 1
		self.num_miss = 1
		self.global_step = global_step
		self.optimizer_name = 'adam'
		self.ckpt_dir = args.ckpt_dir

		with tf.name_scope(self.scope):
			self.states_placeholder = tf.placeholder(tf.float32, shape=(None, args.width, args.height, args.num_stack), name='states_placeholder')
			self.targets_placeholder = tf.placeholder(tf.float32, shape=(None,), name='targets_placeholder')
			self.actions_placeholder = tf.placeholder(tf.int32, shape=(None,), name='actions_placeholder')
			self.dqn = network.q_net(self.scope)

			with tf.name_scope('predict'):
				self.predict_op = self.dqn.base_CNN(self.states_placeholder)

			with tf.name_scope('action'):
				self.best_action = tf.argmax(self.predict_op, axis=-1, name='prediction')

			with tf.name_scope('q_value'):
				self.q_value = tf.reduce_max(self.predict_op, axis=-1)

			with tf.name_scope('loss'):
				action_bool = tf.one_hot(self.actions_placeholder, depth=self.action_space) > 0.5
				action_q = tf.boolean_mask(self.predict_op, action_bool)

				#self.loss_op = tf.reduce_mean(tf.pow(action_q - self.targets_placeholder,2))
				self.loss_op = tf.keras.losses.Huber(delta=10.0)(self.targets_placeholder, action_q)

			with tf.name_scope("learning_rate"):
				self.learning_rate = tf.train.exponential_decay(self.init_lr, self.global_step, self.decay_steps, self.decay_factor,staircase=False)

			with tf.name_scope('training'):
				if self.optimizer_name == "sgd":
					optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
				elif self.optimizer_name == "adam":
					optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)	
	            #elif self.optimizer_name == "momentum":
	                #optimizer = tf.train.MomentumOptimizer(learning_rate=self.init_learning_rate, momentum=args.momentum)
	            #elif self.optimizer_name == "nesterov_momentum":
	                #optimizer = tf.train.MomentumOptimizer(learning_rate=self.init_learning_rate, momentum=args.momentum, use_nesterov=True)
				else:
					sys.exit("Invalid optimizer")
				self.train_op = optimizer.minimize(loss=self.loss_op, global_step=self.global_step)
				update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				self.train_op = tf.group([self.train_op, update_ops])

	def get_sess_saver(self, sess, saver):
		self.sess = sess
		self.saver = saver


	def load_model(self):
		checkpoint_prefix = os.path.join(self.ckpt_dir ,"checkpoint")
		if os.path.exists(checkpoint_prefix+"-latest"):
			print("Last checkpoint epoch {}".format(self.start_epoch.eval()[0]))
			latest_checkpoint_path = tf.train.latest_checkpoint(self.ckpt_dir,latest_filename="checkpoint-latest")
			self.saver.restore(self.sess, latest_checkpoint_path)


	def save_model(self):
		print('Saving checkpoint...')
		self.saver.save(self.sess, os.path.join(self.ckpt_dir, 'checkpoint'),global_step=tf.train.global_step(self.sess, self.global_step), latest_filename="checkpoint-latest")


	def add_memory(self, experience):
		if experience[3] > 0:
			self.num_hit += 1
		else:
			self.num_miss += 1
		self.memories[self.memory_counter] = experience
		self.memory_counter += 1
		if self.memory_counter >= self.memory_capacity:
			self.memory_counter = 0
			self.num_miss = 1
			self.num_hit = 1
			return True
		else:
			return False

	def select_action(self, state):
		if random.random() <= self.epsilon.eval()[0]:
			action =  random.randrange(self.action_space)
		else:
			action = self.sess.run(self.best_action, feed_dict={self.states_placeholder:state, self.dqn.train_phase:False, self.dqn.keep_prob:1})
		return action

	def copy_model(self, new_model):
		old_variables = [v for v in tf.trainable_variables() if v.name.startswith(self.scope)]
		old_variables = sorted(old_variables, key = lambda x: x.name)
		new_variables = [v for v in tf.trainable_variables() if v.name.startswith(new_model.scope)]

		new_variables = sorted(old_variables, key = lambda x: x.name)

		ops = []
		for old_variable, new_variable in zip(old_variables, new_variables):
			new_variable_value = self.sess.run(new_variable)
			op = old_variable.assign(new_variable_value)
			ops.append(op)
		self.sess.run(ops)

	def update(self, target_model):
		with tf.device('/cpu:0'):
			np.random.shuffle(self.memories)

			s, s_, a, r, d = self.memories['s'], self.memories['s_'], self.memories['a'], self.memories['r'], self.memories['d']

			#training_data = tf.data.Dataset.from_tensor_slices((np.array(s), np.array(s_), np.array(a), np.array(r), np.array(d)))
			#training_data = training_data.shuffle(buffer_size=self.memory_capacity)
			#training_data = training_data.batch(batch_size=self.batch_size, drop_remainder=False)
		#train_iterator = training_data.make_initializable_iterator()
		#next_element_train = train_iterator.get_next()
		for repeat in range(self.num_repeat):
			epoch = self.start_epoch.eval()[0]
			print('      Epoch {}'.format(epoch))
			loss_mean = []
			#self.sess.run(train_iterator.initializer)
			for i in range(self.memory_capacity // self.batch_size):
				states, states_, actions, rewards, dones = [], [], [], [], []
				for j in range(self.batch_size):
					states.append(s[i*self.batch_size + j])
					states_.append(s_[i*self.batch_size + j])
					actions.append(a[i*self.batch_size + j])
					rewards.append(r[i*self.batch_size + j])
					dones.append(d[i*self.batch_size + j])
					#states, states_, actions, rewards, dones = self.sess.run(next_element_train)
				target = np.zeros_like(rewards)
				q_ = self.sess.run(target_model.q_value, feed_dict={target_model.states_placeholder:states_, target_model.dqn.train_phase:False, target_model.dqn.keep_prob:1})
				for index in range(len(states)):
					if not dones[index]:
						target[index] = rewards[index] + self.gamma * q_[index]
					else:
						target[index] = rewards[index]
				loss, train = self.sess.run([self.loss_op, self.train_op], feed_dict={self.states_placeholder:states_, self.targets_placeholder:target, self.actions_placeholder:actions , self.dqn.train_phase:True, self.dqn.keep_prob:0.95})
				loss_mean.append(loss)
			print('      average loss: {}'.format(np.mean(loss_mean)))
			if self.start_epoch.eval()[0] % args.log_interval == 0:
				print('Saving model at epoch: {}'.format(epoch))
				self.save_model()
			self.start_epoch_inc.op.run()
		if self.epsilon.eval()[0] > self.epsilon_min:
			print('      Epsilon:  {}'.format(self.epsilon.eval()[0]))
			self.epsilon_step.op.run()

def processed_state(state, width, length):
	state_processed = rgb2gray(state)
	state_processed = cv2.resize(state_processed, (width, length))[:,:,np.newaxis]
	return state_processed

if __name__=='__main__':
	args  = setup_options()


	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	with tf.Graph().as_default():
		global_step = tf.train.get_or_create_global_step()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		#states_placeholder = tf.placeholder(tf.float32, shape=input_shape, name='states_placeholder')
		#q_net = network.q_net()
		#logits = q_net.base_CNN(states_placeholder)
		#action_choice = tf.argmax(logits, axis=-1)
		env = Env('DemonAttack-v0')
		#start_epoch = tf.get_variable("start_epoch", shape=[1], initializer= tf.zeros_initializer,dtype=tf.int32)
        #start_epoch_inc = start_epoch.assign(start_epoch+1)
		agent = Agent(global_step, args, network, 'model')
		target_agent = Agent(global_step, args, network, 'target_model')
		saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=50)
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			agent.get_sess_saver(sess, saver)
			target_agent.get_sess_saver(sess, saver)
			if not args.not_restore:
				agent.load_model()
			for eposide in range(args.num_eposides):
				print('Eposide: {}'.format(eposide))
				total_reward = 0
				frame_pre1 = np.zeros((args.width, args.height, 1))
				frame_pre2 = np.zeros((args.width, args.height, 1))
				state = env.reset()
				frame_now = processed_state(state, args.width, args.height)
				model_input = np.concatenate((frame_pre1, frame_pre2, frame_now), axis=-1)
				if eposide % args.target_interval == 0:
					target_agent.copy_model(agent)
				for time_step in range(args.num_step):
					if args.render:
						env.render()
					action = agent.select_action(model_input[np.newaxis,:,:,:])
					state_, reward, done, reward_orig = env.step(action, time_step)
					total_reward += reward_orig
					frame_pre1 = frame_pre2[:]
					frame_pre2 = frame_now[:]
					frame_now = processed_state(state_, args.width, args.height)
					model_input_next = np.concatenate((frame_pre1, frame_pre2, frame_now), axis=-1)
					if (agent.num_hit / agent.num_miss) > args.hit_ratio or reward_orig > 0:
						if agent.add_memory((model_input, model_input_next, action, reward, done)):
							print('      Updating parmeters')
							agent.update(target_agent)
						if args.plot_input:
							plt.subplot(211)
							plt.imshow(np.concatenate((model_input[:,:,0], model_input[:,:,1], model_input[:,:,2]), axis=1), cmap='gray')
							plt.subplot(212)
							plt.imshow(np.concatenate((model_input_next[:,:,0], model_input_next[:,:,1], model_input_next[:,:,2]), axis=1), cmap='gray')
							plt.show(block=False)
							plt.pause(0.01)
							plt.draw()
					model_input = model_input_next
					if done:
						break
				print("      Total reward with time step {}:  {}".format(time_step, total_reward))


