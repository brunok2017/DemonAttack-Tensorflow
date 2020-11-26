import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from  skimage.color import rgb2gray
import tensorflow as tf
import argparse
import cv2
env = gym.make('DemonAttack-v0')
state = env.reset()
print(state.shape)
i = 0


def processed_state(state, width, length):
	state_processed = rgb2gray(state)
	state_processed = cv2.resize(state_processed, (width, length))[:,:,np.newaxis]
	return state_processed
	

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='test options')
	parser.add_argument("--model_path", type=str, default='./tmp/ckpt/checkpoint-5518.meta')
	parser.add_argument("--ckpt_path", type=str, default='./tmp/ckpt/checkpoint-5518')
	parser.add_argument('--width', type=int, default=84, help="input width")
	parser.add_argument('--height', type=int, default=84, help="input height")
	args = parser.parse_args()

	graph = tf.get_default_graph()
	imported_meta = tf.train.import_meta_graph(args.model_path)
	env = gym.make('DemonAttack-v0')
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	next_game = 'y'
	num_game = 0
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		imported_meta.restore(sess, args.ckpt_path)

		start_epoch = graph.get_tensor_by_name('model_start_epoch:0')
		epoch = sess.run(start_epoch)
		print("Restore checkpoint success of epoch {}".format(epoch))
		while next_game == 'y':
			print("Trial {}".format(str(num_game)))
			num_game += 1
			total_reward = 0
			state = env.reset()
			frame_pre1 = np.zeros((args.width, args.height, 1))
			frame_pre2 = np.zeros((args.width, args.height, 1))
			frame_now = processed_state(state, args.width, args.height)
			model_input = np.concatenate((frame_pre1, frame_pre2, frame_now), axis=-1)
			while True:
				env.render()
				action = sess.run('model/action/prediction:0', feed_dict={
					'model/states_placeholder:0': model_input[np.newaxis,:,:,:],
					'model/keep_prob:0':1.0,
					'model/train_phase_placeholder:0': False})
				state_, reward, done, _ = env.step(action)
				total_reward += reward
				frame_pre1 = frame_pre2[:]
				frame_pre2 = frame_now[:]
				frame_now = processed_state(state_, args.width, args.height)
				model_input = np.concatenate((frame_pre1, frame_pre2, frame_now), axis=-1)
				time.sleep(0.005)
				if done:
					break
			print("Total reward:  {}".format(total_reward))
			next_game = input("Next game (y/n)")
			if next_game != 'y':
				break
	sess.close()