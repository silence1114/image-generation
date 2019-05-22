import os
from glob import glob
import collections
import time
import math
import random
import argparse
import tensorflow as tf
import numpy as np 
from PIL import Image
from datetime import datetime
from tqdm import trange, tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=str, default='')
parser.add_argument("--mode", required=True, choices=["train", "test"])
parser.add_argument("--output_dir", help="where to put output files")
parser.add_argument("--load_path", default=None, help="directory with checkpoint to resume training from or use for testing")


parser.add_argument("--max_epochs", type=int, default=1000)
parser.add_argument("--max_steps", type=int)
parser.add_argument("--log_step", type=int, default=1, help="display progress every progress_freq steps")
parser.add_argument("--save_step", type=int, default=10, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--hidden_num", type=int, default=64, help="number of filters in first conv layer")
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--l1_weight", type=float, default=1000.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

parser.add_argument('--data_dir', type=str, default='data')

config = parser.parse_args()
Eps = 1e-12

def make_dirs(config):
	if not config.output_dir:
		config.output_dir = "{}_{}_{}".format(config.dataset, config.mode, datetime.now().strftime("%y%m%d_%H%M%S"))
	
	if not os.path.exists(config.output_dir):
		os.makedirs(config.output_dir)

	image_dir = os.path.join(config.output_dir, "images")
	if not os.path.exists(image_dir):
		os.makedirs(image_dir)

	config.data_path = os.path.join(config.data_dir, config.dataset)
	if not os.path.exists(config.data_path):
		os.makedirs(config.data_path)

Data = collections.namedtuple("Data", "paths, inputs, targets, count, size, steps_per_epoch")

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


	if all(os.path.splitext(os.path.basename(path))[0].isdigit() for path in input_paths):
		input_paths = sorted(input_paths, key=lambda path: int(os.path.splitext(os.path.basename(path))[0]))
	else:
		input_paths = sorted(input_paths)

	path_queue = tf.train.string_input_producer(input_paths, shuffle=config.mode=="trian")
	reader = tf.WholeFileReader()
	paths, contents = reader.read(path_queue)
	raw_input = decode(contents)
	raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
	raw_input = tf.identity(raw_input)
	raw_input.set_shape([None,None,3])

	width = tf.shape(raw_input)[1]
	a_images = (raw_input[:,:width//2,:]) * 2 - 1 #[0,1] -> [-1,1]
	b_images = (raw_input[:,width//2:,:]) * 2 - 1
	inputs, targets = [a_images,b_images]
	inputs = tf.image.resize_images(inputs, [h, h], method=tf.image.ResizeMethod.AREA)
	targets = tf.image.resize_images(targets, [h, h], method=tf.image.ResizeMethod.AREA)
	paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, inputs, targets], batch_size=batch_size)
	config.steps_per_epoch = int(math.ceil(len(input_paths) / config.batch_size))
	config.data_num = len(input_paths)
	config.size = h

	return paths_batch, inputs_batch, targets_batch

def save_images(fetch, step=None):
	image_dir = os.path.join(config.output_dir, "images")
	for i, in_path in enumerate(fetch["paths"]):
		name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
		for kind in ["inputs", "outputs", "targets"]:
			filename = name + "-" + kind + ".png"
			if step is not None:
				filename = "%08d-%s" % (step, filename)
			out_path = os.path.join(image_dir, filename)
			contents = fetch[kind][i]
			with open(out_path, "wb") as f:
				f.write(contents)
		


def lrelu(x, a):
	x = tf.identity(x)
	return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def Generator(x,repeat_num,hidden_num=64):
	with tf.variable_scope("Generator") as vs:
		layers = []
		with tf.variable_scope("Gen_encoder_1"):
			output = tf.layers.conv2d(x, hidden_num, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
			layers.append(output)

		for i in range(1,repeat_num):
			with tf.variable_scope("Gen_encoder_%d" %(i+1)):
				channel_num = hidden_num * min(8,2**i)
				relu = lrelu(layers[-1],0.2)
				conv = tf.layers.conv2d(relu, channel_num, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
				output = tf.layers.batch_normalization(conv, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
				layers.append(output)


		encoder_layers_num = len(layers)
		for i in range(1,repeat_num):
			skip_layer = encoder_layers_num - i 
			with tf.variable_scope("Gen_decoder_%d" %(skip_layer + 1)):
				channel_num = hidden_num * min(8,2**(skip_layer-1))
				if i == 1:
					input_ = layers[-1]
				else:
					input_ = tf.concat([layers[-1],layers[skip_layer]], axis=3)
				relu = tf.nn.relu(input_)
				conv = tf.layers.conv2d_transpose(relu, channel_num, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
				output =  tf.layers.batch_normalization(conv, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
				if skip_layer > 4:
					output = tf.nn.dropout(output, keep_prob=0.5)
				layers.append(output)
				
		with tf.variable_scope("Gen_decoder_1"):
			input_ = tf.concat([layers[-1], layers[0]], axis=3)
			relu = tf.nn.relu(input_)
			conv =  tf.layers.conv2d_transpose(relu, 3, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=tf.random_normal_initializer(0, 0.02))
			output = tf.tanh(conv)	
			layers.append(output)

	return layers[-1]	

def Discriminator(x,y,mid_layers_num,hidden_num=64):
	with tf.variable_scope("Discriminator") as vs:
		layers = []
		input_ = tf.concat([x,y], axis=3)
		with tf.variable_scope("Dis_encoder_1"):
			conv = tf.layers.conv2d(tf.pad(input_, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"), hidden_num, kernel_size=4, strides=(2,2), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
			relu = lrelu(conv, 0.2)
			layers.append(relu)

		for i in range(mid_layers_num):
			with tf.variable_scope("Dis_encoder_%d" % (len(layers)+1)):
				channel_num = hidden_num * min(2**(i+1), 8)
				stride = 1 if i == mid_layers_num - 1 else 2  
				conv = tf.layers.conv2d(tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"), channel_num, kernel_size=4, strides=(stride,stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
				norm = tf.layers.batch_normalization(conv, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))
				relu = lrelu(norm, 0.2)
				layers.append(relu)

		with tf.variable_scope("Dis_encoder_%d" % (len(layers)+1)):
			conv =  tf.layers.conv2d(tf.pad(layers[-1], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"), 1, kernel_size=4, strides=(1,1), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))
			output = tf.sigmoid(conv)
			layers.append(output)

	return layers[-1]	


def train(paths, inputs, targets, repeat_num, mid_layers_num):
	with tf.variable_scope("generator"):
		generated = Generator(inputs, repeat_num=repeat_num)

	with tf.name_scope("real_discriminator"):
		with tf.variable_scope("discriminator"):
			real_output = Discriminator(inputs, targets, mid_layers_num=mid_layers_num)

		
	with tf.name_scope("fake_discriminator"):
		with tf.variable_scope("discriminator", reuse=True):
			fake_output = Discriminator(inputs, generated, mid_layers_num=mid_layers_num)

	dis_loss = tf.reduce_mean(-(tf.log(real_output + Eps) + tf.log(1 - fake_output + Eps)))
	gen_loss = tf.reduce_mean(-(tf.log(fake_output + Eps)))
	l1_loss = tf.reduce_mean(tf.abs(targets - generated))
	gen_loss_total = config.gan_weight * gen_loss + config.l1_weight * l1_loss

	dis_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
	dis_optimizer = tf.train.AdamOptimizer(config.lr, config.beta1)
	dis_grad = dis_optimizer.compute_gradients(dis_loss, var_list=dis_vars)
	dis_optim = dis_optimizer.apply_gradients(dis_grad)

	with tf.control_dependencies([dis_optim]):
		gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
		gen_optimizer = tf.train.AdamOptimizer(config.lr, config.beta1)
		gen_grad = gen_optimizer.compute_gradients(gen_loss_total, var_list=gen_vars)
		gen_optim = gen_optimizer.apply_gradients(gen_grad)

	global_step = tf.train.get_or_create_global_step()
	update_global_step = tf.assign(global_step, global_step+1)

	inputs_img = (inputs + 1) / 2  #[-1,1] -> [0,1]
	targets_img = (targets + 1) / 2
	outputs_img = (generated + 1) / 2

	inputs_img = tf.image.convert_image_dtype(inputs_img, dtype=tf.uint8, saturate=True)
	targets_img = tf.image.convert_image_dtype(targets_img, dtype=tf.uint8, saturate=True)
	outputs_img = tf.image.convert_image_dtype(outputs_img, dtype=tf.uint8, saturate=True)

	display_fetch = {"paths":paths, "inputs":tf.map_fn(tf.image.encode_png, inputs_img, dtype=tf.string, name="input_pngs"), "targets": tf.map_fn(tf.image.encode_png, targets_img, dtype=tf.string, name="target_pngs"), "outputs": tf.map_fn(tf.image.encode_png, outputs_img, dtype=tf.string, name="output_pngs"),}
	
	summary_op = tf.summary.merge([
		tf.summary.image("inputs", inputs_img),
		tf.summary.image("targets", targets_img),
		tf.summary.image("outputs", outputs_img),
	
		tf.summary.scalar("dis_loss", dis_loss),
		tf.summary.scalar("gen_loss", gen_loss),
		tf.summary.scalar("l1_loss", l1_loss),
	])

	saver = tf.train.Saver(max_to_keep=1)
	sv = tf.train.Supervisor(logdir=config.output_dir, save_summaries_secs=0, saver=None)

	with sv.managed_session() as sess:
		update_bar = False

		if config.load_path:
			print("loading trained model from load_path")
			checkpoint = tf.train.latest_checkpoint(config.load_path)
			saver.restore(sess, checkpoint)
			update_bar = True

		with tqdm(total=config.max_steps) as pbar:
			for i in range(config.max_steps):
				fetch_dict = {"gen_optim": gen_optim, "update_global_step":update_global_step, "global_step":sv.global_step}
				
				if (i + 1) % config.log_step == 0 or i == config.max_steps - 1:
					fetch_dict.update({"dis_loss":dis_loss, "gen_loss":gen_loss, "l1_loss":l1_loss, "summary":summary_op})
			
				if (i + 1) % (config.log_step * 10) == 0 or i == config.max_steps - 1:
					fetch_dict.update({"display":display_fetch})

				results = sess.run(fetch_dict)

				if (i + 1) % config.log_step == 0 or i == config.max_steps - 1:
					sv.summary_writer.add_summary(results["summary"], results["global_step"])
					print("[{}/{}] dis_loss: {:.6f} gen_loss: {:.6f} l1_loss: {:.6f}".format(results["global_step"], config.max_steps, results['dis_loss'], results['gen_loss'], results['l1_loss']))

				if (i + 1) % (config.log_step * 10) == 0 or i == config.max_steps - 1:
					print("saving display images")
					save_images(results["display"], step=results["global_step"])

				if (i + 1) % config.save_step == 0 or i == config.max_steps - 1:
					print("saving model")
					saver.save(sess, os.path.join(config.output_dir, "model"), global_step=results["global_step"])

				if sv.should_stop():
					break

				if results["global_step"] >= config.max_steps-1:
					print("saving model")
					saver.save(sess, os.path.join(config.output_dir, "model"), global_step=results["global_step"])
					break

				if config.load_path and update_bar:
					pbar.update(results["global_step"]+1)
					update_bar = False
				else:
					pbar.update(1)

		
def test(paths, inputs,targets,repeat_num,mid_layers_num):
	with tf.variable_scope("generator"):
		generated = Generator(inputs, repeat_num=repeat_num)

	with tf.name_scope("real_discriminator"):
		with tf.variable_scope("discriminator"):
			real_output = Discriminator(inputs, targets, mid_layers_num=mid_layers_num)

		
	with tf.name_scope("fake_discriminator"):
		with tf.variable_scope("discriminator", reuse=True):
			fake_output = Discriminator(inputs, generated, mid_layers_num=mid_layers_num)

	dis_loss = tf.reduce_mean(-(tf.log(real_output + Eps) + tf.log(1 - fake_output + Eps)))
	gen_loss = tf.reduce_mean(-(tf.log(fake_output + Eps)))
	l1_loss = tf.reduce_mean(tf.abs(targets - generated))
	gen_loss_total = config.gan_weight * gen_loss + config.l1_weight * l1_loss

	dis_vars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
	dis_optimizer = tf.train.AdamOptimizer(config.lr, config.beta1)
	dis_grad = dis_optimizer.compute_gradients(dis_loss, var_list=dis_vars)
	dis_optim = dis_optimizer.apply_gradients(dis_grad)

	with tf.control_dependencies([dis_optim]):
		gen_vars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
		gen_optimizer = tf.train.AdamOptimizer(config.lr, config.beta1)
		gen_grad = gen_optimizer.compute_gradients(gen_loss_total, var_list=gen_vars)
		gen_optim = gen_optimizer.apply_gradients(gen_grad)

	global_step = tf.train.get_or_create_global_step()
	update_global_step = tf.assign(global_step, global_step+1)

	inputs_img = (inputs + 1) / 2  #[-1,1] -> [0,1]
	targets_img = (targets + 1) / 2
	outputs_img = (generated + 1) / 2

	inputs_img = tf.image.convert_image_dtype(inputs_img, dtype=tf.uint8, saturate=True)
	targets_img = tf.image.convert_image_dtype(targets_img, dtype=tf.uint8, saturate=True)
	outputs_img = tf.image.convert_image_dtype(outputs_img, dtype=tf.uint8, saturate=True)

	display_fetch = {"paths":paths, "inputs":tf.map_fn(tf.image.encode_png, inputs_img, dtype=tf.string, name="input_pngs"), "targets": tf.map_fn(tf.image.encode_png, targets_img, dtype=tf.string, name="target_pngs"), "outputs": tf.map_fn(tf.image.encode_png, outputs_img, dtype=tf.string, name="output_pngs"),}
			
	saver = tf.train.Saver(max_to_keep=1)
	sv = tf.train.Supervisor(logdir=config.output_dir, save_summaries_secs=0, saver=None)

	with sv.managed_session() as sess:
		print("loading trained model from load_path")
		checkpoint = tf.train.latest_checkpoint(config.load_path)
		saver.restore(sess, checkpoint)

		for i in trange(config.steps_per_epoch):
			results = sess.run(display_fetch)
			save_images(results)


			
def main():
	make_dirs(config)

	seed = random.randint(0, 2**31 - 1)
	tf.set_random_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	if config.mode == "test":
		if not config.load_path:
			raise Exception("[!] You should specify `load_path` to load a pretrained model")

	for key, value in config._get_kwargs():
		print(key, "=", value)

	paths, inputs, targets = data_loader(config.data_path,config.batch_size)
	print("data numbers = %d" % config.data_num)

	repeat_num = int(np.log2(config.size))
	if config.size == 256:
		mid_layers_num = 3
	elif config.size == 384:
		mid_layers_num = 4
	elif config.size == 512:
		mid_layers_num = 5
	else:
		raise Exception("[!] You should specify the mid_layers_num for the image size") 

		
	if config.mode == "train":
		config.max_steps = config.steps_per_epoch * config.max_epochs
		train(paths, inputs, targets, repeat_num, mid_layers_num)
	
	elif config.mode == "test":
		if not config.load_path:
			raise Exception("[!] You should specify `load_path` to load a pretrained model")
		test(paths, inputs, targets, repeat_num, mid_layers_num)
			

			
main()			






	

