import subprocess as sp
import os

possible_ckpt_dir = ['E:\\c2p_yy2018\\ckpt',
					 '/home/lab501/c2p_newdata/ckpt',
					 '/home/guang/c2p/ckpt']
possible_eval_py_path = ['E:\\c2p_yy2018\\eval.py',
						 '/home/lab501/c2p_newdata/eval.py',
						 '/home/guang/c2p/eval.py']

def get_all_models(ckpt_dir):
	res = set()
	for rt,dirs,files in os.walk(ckpt_dir):
		for file in files:
			if file.startswith('m'):
				file = '.'.join(file.split('.')[:-1])
				res.add(os.path.join(rt, file))
	return list(res)

def evaluate_every_model(model_list, eval_py_path):
	save_file = open(os.path.join(os.getcwd(), 'best_evaluation_accuracy_and_model.txt'), 'w')
	best_accuracy = 0
	idx = 1
	for model in model_list:
		print('evaluating*******')
		cmd = ' '.join(['python', eval_py_path, '--ckpt_path='+model])
		popen = sp.Popen(cmd, stdout=sp.PIPE)
		popen.wait()
		eval_py_stdout = popen.stdout.readlines()
		model_acc_with_enter = eval_py_stdout[-1].decode('utf-8')
		last = -1
		while '0' > model_acc_with_enter[last] or model_acc_with_enter[last] > '9':
			last -= 1
		model_acc = float(model_acc_with_enter[:last])
		if model_acc > best_accuracy:
			best_accuracy = model_acc
			best_model = model
		print('evaluated model {}/{}'.format(idx, len(model_list)))
		idx += 1
	save_file.write(best_model + str(best_accuracy))
	print('best evaluation accuracy: {}\nbest model: {}'.format(best_accuracy, best_model))

def main():
	for p_ckpt in possible_ckpt_dir:
		if os.path.exists(p_ckpt):
			ckpt_dir = p_ckpt
			break
	for p_eval_py_path in possible_eval_py_path:
		if os.path.exists(p_eval_py_path):
			eval_py_path = p_eval_py_path
			break
	model_list = get_all_models(ckpt_dir)
	# print(model_list)
	evaluate_every_model(model_list, eval_py_path)

if __name__ == '__main__':
	main()