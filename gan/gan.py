import os
import math
import json
import logging
import argparse
from PIL import Image
from glob import glob
from collections import deque
from tqdm import trange
from itertools import chain
from datetime import datetime
import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, type=str, default='')
parser.add_argument('--mode', required=True, choices=['train','test'])
parser.add_argument('--load_path', type=str, default='') # must specify when test 
parser.add_argument('--test_num', type=int, default=3)

parser.add_argument('--max_step', type=int, default=100000)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--hidden_num', type=int, default=128, choices=[64,128])
parser.add_argument('--z_num', type=int, default=64, choices=[64, 128])
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--lr_update_step', type=int, default=75000, choices=[100000, 75000])
parser.add_argument('--d_lr', type=float, default=0.00004)
parser.add_argument('--g_lr', type=float, default=0.00004)
parser.add_argument('--lr_lower_boundary', type=float, default=0.00001)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--lambda_k', type=float, default=0.001)

parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--save_step', type=int, default=5000)
parser.add_argument('--num_log_samples', type=int, default=3)
parser.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--random_seed', type=int, default=123)

config = parser.parse_args()


# data_format = 'NHWC'

def make_dirs(config):
	formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
	logger = logging.getLogger()
	for hdlr in logger.handlers:
		logger.removeHandler(hdlr)
	handler = logging.StreamHandler()
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	if config.load_path:	
		config.model_dir = os.path.join(config.log_dir,config.load_path)
	else:
		config.model_name = "{}_{}".format(config.dataset, datetime.now().strftime("%y%m%d_%H%M%S"))
	if not hasattr(config,'model_dir'):
		config.model_dir = os.path.join(config.log_dir, config.model_name)
	config.data_path = os.path.join(config.data_dir, config.dataset)

	for path in [config.log_dir, config.data_dir, config.model_dir]:
		if not os.path.exists(path):
			os.makedirs(path)

def save_image(tensor, filename, column_num=8):
	def make_grid(tensor, column_num=8):
		image_num = tensor.shape[0]
		columns = min(column_num, image_num)
		rows = int(math.ceil(float(image_num) / columns))
		height, width = int(tensor.shape[1]), int(tensor.shape[2])
		grid = np.zeros([height * rows, width * columns, 3], dtype=np.uint8)
		i = 0
		for r in range(rows):
			for c in range(columns):
				if i >= image_num:
					break
				h = r * height
				w = c * width 
				grid[h:h+height, w:w+width] = tensor[i]
				i += 1
		return grid

	ndarr = make_grid(tensor, column_num=column_num)
	im = Image.fromarray(ndarr)
	im.save(filename)

def data_loader(root,batch_size):
	input_paths = glob(os.path.join(root, "*.jpg"))
	decode = tf.image.decode_jpeg
	if len(input_paths) == 0:
		input_paths = glob(os.path.join(root, "*.png"))
		decode = tf.image.decode_png

	if len(input_paths) == 0:
		raise Exception("dataset contains no image files")	

	with Image.open(input_paths[0]) as img:
		w, h = img.size
		shape = [h,w,3]

	path_queue = tf.train.string_input_producer(list(input_paths),shuffle=False)
	reader = tf.WholeFileReader()
	paths, contents = reader.read(path_queue)
	images = decode(contents,channels=3)
	images.set_shape(shape)

	min_after_dequeue = 5000
	capacity = min_after_dequeue + 3 * batch_size
	queue = tf.train.shuffle_batch([images],batch_size=batch_size,num_threads=4,capacity=capacity,min_after_dequeue=min_after_dequeue,name='synthetic_inputs')


	return tf.to_float(queue)



def upsample(x, scale):
    shape = x.get_shape().as_list()
    _, h, w, _ = [num if num is not None else -1 for num in shape]
    upsampled_x = tf.image.resize_nearest_neighbor(x, (h*scale, w*scale))
    return upsampled_x

def downsample(x,channel_num,scale):
	downsampled_x = slim.conv2d(x,channel_num,kernel_size=[3,3],stride=scale,activation_fn=tf.nn.elu,data_format='NHWC')
	return downsampled_x

# Generator

def Generator(z,hidden_num,repeat_num,reuse):
	with tf.variable_scope("Generator",reuse=reuse) as vs:
		layers = []
		
		with tf.variable_scope("Gen_fully_connected"):
			output_ch = int(np.prod([8,8,hidden_num]))
			output = slim.fully_connected(z,output_ch,activation_fn=None)
			output = tf.reshape(output,[-1,8,8,hidden_num]) 
			layers.append(output)

		for i in range(repeat_num-1):
			with tf.variable_scope("Gen_decoder_%d" %(i+1)):
				conv1 = slim.conv2d(layers[-1],hidden_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC') 
				conv2 = slim.conv2d(conv1,hidden_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC') 
				output = upsample(conv2,2)
				layers.append(output)

		with tf.variable_scope("Gen_decoder_%d" %(repeat_num)):
			conv1 = slim.conv2d(layers[-1],hidden_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC') 
			conv2 = slim.conv2d(conv1,hidden_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC') 
			output = slim.conv2d(conv2,3,kernel_size=[3,3],stride=1,activation_fn=None,data_format='NHWC') 
			layers.append(output)

	variables = tf.contrib.framework.get_variables(vs)
	return layers[-1], variables


# Discriminator

def Discriminator(x,z_num,repeat_num,hidden_num):
	with tf.variable_scope("Discriminator") as vs:
		layers = []
		with tf.variable_scope("Dis_convolution"):
			output = slim.conv2d(x,hidden_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC')
			layers.append(output)

		for i in range(repeat_num-1):
			with tf.variable_scope("Dis_encoder_%d" %(i+1)):
				channel_num = hidden_num * (i+1)
				conv1 = slim.conv2d(layers[-1],channel_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC')
				conv2 = slim.conv2d(conv1,channel_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC')
				output = downsample(conv2,channel_num,scale=2)
				layers.append(output)

		with tf.variable_scope("Dis_encoder_%d" %(repeat_num)):
			channel_num = hidden_num * (repeat_num)
			conv1 = slim.conv2d(layers[-1],channel_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC')
			conv2 = slim.conv2d(conv1,channel_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC')
			layers.append(conv2)

		with tf.variable_scope("Dis_encoder_fully_connected"):
			channel_num = hidden_num * (repeat_num)
			z = tf.reshape(layers[-1],[-1,np.prod([8,8,channel_num])])
			z = slim.fully_connected(z,z_num,activation_fn=None)
			layers.append(z)

		with tf.variable_scope("Dis_decoder_fully_connected"):
			output_ch = int(np.prod([8, 8, hidden_num]))
			output = slim.fully_connected(layers[-1], output_ch, activation_fn=None)
			output = tf.reshape(output,[-1,8,8,hidden_num]) 
			layers.append(output)

		for i in range(repeat_num-1):
			with tf.variable_scope("Dis_decoder_%d" %(i+1)):
				conv1 = slim.conv2d(layers[-1],hidden_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC')
				conv2 = slim.conv2d(conv1,hidden_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC')
				output = upsample(conv2,2)
				layers.append(output)

		with tf.variable_scope("Dis_decoder_%d" %(repeat_num)):
			conv1 = slim.conv2d(layers[-1],hidden_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC') 
			conv2 = slim.conv2d(conv1,hidden_num,kernel_size=[3,3],stride=1,activation_fn=tf.nn.elu,data_format='NHWC') 
			output = slim.conv2d(conv2,3,kernel_size=[3,3],stride=1,activation_fn=None,data_format='NHWC') 
			layers.append(output)

	variables = tf.contrib.framework.get_variables(vs)
	return layers[-1], z, variables

def train(config,data_loader):
	dataset = config.dataset
	batch_size = config.batch_size
	step = tf.Variable(0, name='step', trainable=False)
	beta1 = config.beta1
	beta2 = config.beta2
	g_lr = tf.Variable(config.g_lr, name='g_lr')
	d_lr = tf.Variable(config.d_lr, name='d_lr')
	g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
	d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')
	gamma = config.gamma
	lambda_k = config.lambda_k
	z_num = config.z_num
	hidden_num = config.hidden_num
	model_dir = config.model_dir
	load_path = config.load_path
	start_step = 0
	log_step = config.log_step
	max_step = config.max_step
	save_step = config.save_step
	lr_update_step = config.lr_update_step

	_, height, width, channel = data_loader.get_shape().as_list()
	repeat_num = int(np.log2(height)) - 2


	raw_img = data_loader
	img = raw_img/127.5 - 1

	z = tf.random_uniform((tf.shape(img)[0],z_num), minval=-1.0, maxval=1.0)
	k_t = tf.Variable(0., trainable=False, name='k_t')

	generated, gen_var = Generator(z, hidden_num, repeat_num, reuse=False)
	dis_output, dis_z, dis_var = Discriminator(tf.concat([generated,img],0),z_num,repeat_num,hidden_num)
	generated_reconstructed, real_reconstructed = tf.split(dis_output,2)

	
	generated_img =  tf.clip_by_value((generated+1)*127.5,0,255)
	generated_reconstructed_img = tf.clip_by_value((generated_reconstructed+1)*127.5,0,255)
	real_reconstructed_img = tf.clip_by_value((real_reconstructed+1)*127.5,0,255)

	optimizer = tf.train.AdamOptimizer
	gen_optimizer, dis_optimizer = optimizer(g_lr), optimizer(d_lr)

	dis_loss_real = tf.reduce_mean(tf.abs(real_reconstructed-img))
	dis_loss_fake = tf.reduce_mean(tf.abs(generated_reconstructed-generated))

	dis_loss = dis_loss_real - k_t * dis_loss_fake
	gen_loss = tf.reduce_mean(tf.abs(generated_reconstructed-generated))

	dis_optim = dis_optimizer.minimize(dis_loss, var_list=dis_var)
	gen_optim = gen_optimizer.minimize(gen_loss, global_step=step, var_list=gen_var)

	balance = gamma * dis_loss_real - gen_loss
	measure = dis_loss_real + tf.abs(balance)

	with tf.control_dependencies([dis_optim, gen_optim]):
		k_update = tf.assign(k_t, tf.clip_by_value(k_t+lambda_k*balance,0,1))

	summary_op = tf.summary.merge([
		tf.summary.image("Generated", generated_img),
		tf.summary.image("Reconstructed_generated", generated_reconstructed_img),
		tf.summary.image("Reconstructed_real", real_reconstructed_img),

		tf.summary.scalar("loss/dis_loss", dis_loss),
		tf.summary.scalar("loss/dis_loss_real", dis_loss_real),
		tf.summary.scalar("loss/dis_loss_fake", dis_loss_fake),
		tf.summary.scalar("loss/gen_loss", gen_loss),
		tf.summary.scalar("misc/measure", measure),
		tf.summary.scalar("misc/k_t", k_t),
		tf.summary.scalar("misc/d_lr", d_lr),
		tf.summary.scalar("misc/g_lr", g_lr),
		tf.summary.scalar("misc/balance", balance),
	])




	saver = tf.train.Saver()
	summary_writer = tf.summary.FileWriter(model_dir)

	sv = tf.train.Supervisor(logdir=model_dir, is_chief=True, saver=saver, summary_op=None, summary_writer=summary_writer, save_model_secs=300, global_step=step, ready_for_local_init_op=None)

	gpu_options = tf.GPUOptions(allow_growth=True)
	sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	sess = sv.prepare_or_wait_for_session(config=sess_config)



	z_fixed = np.random.uniform(-1,1,size=(batch_size,z_num))
	img_fixed = data_loader.eval(session=sess)
	save_image(img_fixed, '{}/img_fixed.png'.format(model_dir))

	prev_measure = 1
	measure_history = deque([0]*lr_update_step, lr_update_step)

	for i in trange(start_step,max_step):
		fetch_dict = {"k_update":k_update, "measure":measure}
		if i % log_step == 0 :
			fetch_dict.update({"summary":summary_op, "gen_loss":gen_loss, "dis_loss":dis_loss, "k_t":k_t})
		result = sess.run(fetch_dict)

		measure_history.append(result['measure'])

		if i % log_step == 0 :
			summary_writer.add_summary(result['summary'],i)
			summary_writer.flush()
			print("[{}/{}] loss_dis: {:.6f} loss_gen: {:.6f} measure: {:.4f}, k_t: {:.4f}".format(i, max_step, result['dis_loss'], result['gen_loss'], result['measure'], result['k_t']))

		if i % (log_step * 10) == 0:
			gened_img = sess.run(generated_img, {z:z_fixed})
			save_image(gened_img, os.path.join(model_dir, '{}_gened.png'.format(i)))
			real_recon_img = sess.run(real_reconstructed_img, {raw_img:img_fixed})
			save_image(real_recon_img,os.path.join(model_dir, '{}_recon_real.png'.format(i)))
			gened_recon_img = sess.run(real_reconstructed_img, {raw_img:gened_img})
			save_image(gened_recon_img,os.path.join(model_dir, '{}_recon_gened.png'.format(i)))      

		if i % lr_update_step == lr_update_step - 1:
			sess.run([g_lr_update,d_lr_update])



def test(config,data_loader):
	dataset = config.dataset
	batch_size = config.batch_size
	test_num = config.test_num
	step = tf.Variable(0, name='step', trainable=False)
	beta1 = config.beta1
	beta2 = config.beta2
	g_lr = tf.Variable(config.g_lr, name='g_lr')
	d_lr = tf.Variable(config.d_lr, name='d_lr')
	g_lr_update = tf.assign(g_lr, tf.maximum(g_lr * 0.5, config.lr_lower_boundary), name='g_lr_update')
	d_lr_update = tf.assign(d_lr, tf.maximum(d_lr * 0.5, config.lr_lower_boundary), name='d_lr_update')
	gamma = config.gamma
	lambda_k = config.lambda_k
	z_num = config.z_num
	hidden_num = config.hidden_num
	model_dir = config.model_dir
	load_path = config.load_path
	start_step = 0
	log_step = config.log_step
	max_step = config.max_step
	save_step = config.save_step
	lr_update_step = config.lr_update_step

	_, height, width, channel = data_loader.get_shape().as_list()
	repeat_num = int(np.log2(height)) - 2


	raw_img = data_loader
	img = raw_img/127.5 - 1

	z = tf.random_uniform((tf.shape(img)[0],z_num), minval=-1.0, maxval=1.0)
	k_t = tf.Variable(0., trainable=False, name='k_t')

	generated, gen_var = Generator(z, hidden_num, repeat_num, reuse=False)
	dis_output, dis_z, dis_var = Discriminator(tf.concat([generated,img],0),z_num,repeat_num,hidden_num)
	generated_reconstructed, real_reconstructed = tf.split(dis_output,2)

	
	generated_img =  tf.clip_by_value((generated+1)*127.5,0,255)
	generated_reconstructed_img = tf.clip_by_value((generated_reconstructed+1)*127.5,0,255)
	real_reconstructed_img = tf.clip_by_value((real_reconstructed+1)*127.5,0,255)

	optimizer = tf.train.AdamOptimizer
	gen_optimizer, dis_optimizer = optimizer(g_lr), optimizer(d_lr)

	dis_loss_real = tf.reduce_mean(tf.abs(real_reconstructed-img))
	dis_loss_fake = tf.reduce_mean(tf.abs(generated_reconstructed-generated))

	dis_loss = dis_loss_real - k_t * dis_loss_fake
	gen_loss = tf.reduce_mean(tf.abs(generated_reconstructed-generated))

	dis_optim = dis_optimizer.minimize(dis_loss, var_list=dis_var)
	gen_optim = gen_optimizer.minimize(gen_loss, global_step=step, var_list=gen_var)

	balance = gamma * dis_loss_real - gen_loss
	measure = dis_loss_real + tf.abs(balance)

	with tf.control_dependencies([dis_optim, gen_optim]):
		k_update = tf.assign(k_t, tf.clip_by_value(k_t+lambda_k*balance,0,1))

	summary_op = tf.summary.merge([
		tf.summary.image("Generated", generated_img),
		tf.summary.image("Reconstructed_generated", generated_reconstructed_img),
		tf.summary.image("Reconstructed_real", real_reconstructed_img),

		tf.summary.scalar("loss/dis_loss", dis_loss),
		tf.summary.scalar("loss/dis_loss_real", dis_loss_real),
		tf.summary.scalar("loss/dis_loss_fake", dis_loss_fake),
		tf.summary.scalar("loss/gen_loss", gen_loss),
		tf.summary.scalar("misc/measure", measure),
		tf.summary.scalar("misc/k_t", k_t),
		tf.summary.scalar("misc/d_lr", d_lr),
		tf.summary.scalar("misc/g_lr", g_lr),
		tf.summary.scalar("misc/balance", balance),
	])




	saver = tf.train.Saver()
	summary_writer = tf.summary.FileWriter(model_dir)

	sv = tf.train.Supervisor(logdir=model_dir, is_chief=True, saver=saver, summary_op=None, summary_writer=summary_writer, save_model_secs=300, global_step=step, ready_for_local_init_op=None)

	gpu_options = tf.GPUOptions(allow_growth=True)
	sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	sess = sv.prepare_or_wait_for_session(config=sess_config)

	g = tf.get_default_graph()
	g._finalized = False
	
	all_test_gened = None
	for i in range(test_num):
		z_fixed = np.random.uniform(-1,1,size=(batch_size,z_num))
		test_gened = sess.run(generated_img,{z:z_fixed})

		if all_test_gened is None:
			all_test_gened = test_gened
		else:
			all_test_gened = np.concatenate([all_test_gened,test_gened])

		save_image(all_test_gened,"{}/gened_{}.png".format(model_dir,i))

	save_image(all_test_gened,"{}/all_gened.png".format(model_dir),column_num=16)



def main():
	make_dirs(config)

	rng = np.random.RandomState(config.random_seed)
	tf.set_random_seed(config.random_seed)

	if config.mode == "test":
		config.batch_size = 64

	my_data_loader = data_loader(config.data_path,config.batch_size)
	
	if config.mode == "train":
	
		train(config,my_data_loader)
	else:
		if not config.load_path:
			raise Exception("[!] You should specify `load_path` to load a pretrained model")
		test(config,my_data_loader)


main()
