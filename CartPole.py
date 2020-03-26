import gym, numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')
#print(env.observation_space)
ninputs, nlayers, noutputs, alpha = 4, 4, 1, 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()
placeholder = tf.placeholder(tf.float32, shape=[None, ninputs])
hiddenlayers = tf.layers.dense(placeholder, nlayers, activation=tf.nn.elu, kernel_initializer=initializer)

raw_outputs = tf.layers.dense(hiddenlayers, noutputs)
activated_outputs = tf.nn.sigmoid(raw_outputs)

action = tf.multinomial(tf.concat(axis=1, values=[activated_outputs, 1-activated_outputs]), num_samples=1)
crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=1.0-tf.to_float(action), logits=raw_outputs)
optimizer = tf.train.AdamOptimizer(alpha)
optimized_values = optimizer.compute_gradients(crossentropy)

_gradients, _placeholders, _inputs = [], [], []

for grad, var in optimized_values:
	_gradients.append(grad)
	_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
	_placeholders.append(_placeholder)
	_inputs.append((_placeholder, var))

train = optimizer.apply_gradients(_inputs)

save = tf.train.Saver()
init = tf.global_variables_initializer()
episodes, steps, moves, gamma = 10, 1000, 250, 0.95

def discount_rewards(rewards, gamma):
	discountrewards, net_reward = np.zeros(len(rewards)), 0
	#for _ in range(len(rewards)-1, -1, -1):
	for _ in reversed(range(len(rewards))):
		net_reward = rewards[_]+net_reward*gamma
		discountrewards[_] = net_reward
	return discountrewards

# with tf.Session() as sess:
# 	sess.run(init)
# 	print('Start')
# 	for move in range(moves):
# 		totalrewards, totalgradients = [], []
# 		for episode in range(episodes):
# 			currewards, curgradients = [], []
# 			observations = env.reset()
# 			for step in range(steps):
# 				_action, _gradient = sess.run([action, _gradients], feed_dict={placeholder: observations.reshape(1, ninputs)})
# 				observations, reward, done, info = env.step(_action[0][0])
# 				currewards.append(reward)
# 				curgradients.append(_gradient)
# 				if done: break
# 			totalrewards.append(currewards)
# 			totalgradients.append(curgradients)
# 		discountrewards = [discount_rewards(reward, gamma) for reward in totalrewards]
# 		_discountrewards = np.concatenate(discountrewards)
# 		mean, std = _discountrewards.mean(), _discountrewards.std()
# 		normalized_rewards = [(reward-mean)/std for reward in discountrewards]
# 		feed = {_ph : np.mean([reward*totalgradients[ind][stp][_vr] for ind, rewards in enumerate(normalized_rewards) for stp, reward in enumerate(rewards)], axis=0) for _vr, _ph  in enumerate(_placeholders)}
# 		sess.run(train, feed_dict=feed)
# 	metadata = tf.train.export_meta_graph(filename='metadata.meta')
# 	save.save(sess, 'metadata')
# 	print('Ended')

def main():
	observations = env.reset()
	with tf.Session() as sess:
		metadata = tf.train.import_meta_graph('metadata.meta')
		metadata.restore(sess, 'metadata')
		nofmoves = 0
		for _ in range(5000):
			_action, _gradient = sess.run([action, _gradients], feed_dict={placeholder: observations.reshape(1, ninputs)})
			observations, reward, done, info = env.step(_action[0][0])
			nofmoves += 1
			env.render()
			if done: break
		print("number of moves:", nofmoves)
main()