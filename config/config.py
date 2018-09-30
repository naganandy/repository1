data = "reactionsU"

import configargparse, os,sys,inspect
from configargparse import YAMLConfigFileParser




def parse():
	current_folder_name, default_config_file = get_default_config_file()
	yml_file_with_path = os.path.join(current_folder_name, data + ".yml")
	p = configargparse.ArgParser(config_file_parser_class = YAMLConfigFileParser, default_config_files=[yml_file_with_path])
	p.add('-c', '--my-config', is_config_file=True, help='config file path')
	p.add('--data', type=str, default=data, help='data name (reactions or coauthorship)')
	p.add('--dataset', type=str, default="", help='dataset name (e.g.: iAF692 for reactions, cora for coauthorship)')
	p.add('--numFeatures', type=int, default=16, help='number of features (random Gaussian) for GCN initialisation')
	p.add('--missingFrac', type=float, default=0.9, help='fraction of hyperlinks missing in the incomplete hypergraph')
	p.add('--hiddenGCN', type=int, default=8, help='hidden layer size for GCN')
	p.add('--dropoutGCN', type=float, default=0.5, help='dropout probability for GCN')
	p.add('--lr', type=float, default=0.01, help='learning rate')
	p.add('--wd', type=float, default=0.0005, help='weight decay')
	p.add('--no_cuda', type=bool, default=False, help='cuda for gpu')
	p.add('--gpu', type=int, default=4, help='gpu number to use')
	p.add('--epochs', type=int, default=200, help='number of epochs to train')
	p.add('--hiddenMLP', type=int, default=32, help='hidden layer size for MLP')
	p.add('--dropoutMLP', type=float, default=0.25, help='dropout probability for MLP')
	p.add('-f') # for jupyter default
	return p.parse_args()




def get_default_config_file():
	current_file = os.path.abspath(inspect.getfile(inspect.currentframe()))
	current_file_head, current_file_tail = os.path.split(current_file)
	current_file_name = current_file_tail.split('.')[0]

	current_file_name_yml = current_file_name + '.yml'
	default_config_file_with_path = os.path.join(current_file_head, current_file_name_yml)
	return current_file_head, default_config_file_with_path