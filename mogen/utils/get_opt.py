from argparse import Namespace
import re
from os.path import join as pjoin
from utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, **kwargs):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path, 'r') as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip('\n').split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # print(opt)
    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    opt.data_base = '/home/hynann/data'

    if opt.dataset_name == 't2m':
        opt.data_root = pjoin(opt.data_base, 'HumanML3D/HumanML3D/')

        motion_dir_name = 'new_joint_vecs'
        max_motion_length = 196
        max_motion_frame = 196
        dim_pose = 263

        opt.motion_dir = pjoin(opt.data_root, motion_dir_name)
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = dim_pose
        opt.max_motion_length = max_motion_length
        opt.max_motion_frame = max_motion_frame
        opt.max_motion_token = 55
    elif opt.dataset_name == 'ms':
        opt.data_root = pjoin(opt.data_base, 'HumanML3D_272')
        motion_dir_name = 'motion_data'
        max_motion_length = 300
        max_motion_frame = 300
        dim_pose = 272
        opt.motion_dir = pjoin(opt.data_root, motion_dir_name)
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.dim_pose = dim_pose
        opt.max_motion_length = max_motion_length
        opt.max_motion_frame = max_motion_frame
        opt.max_motion_token = 55
    elif opt.dataset_name == 'kit':
        opt.data_root = './dataset/KIT-ML/'
        opt.motion_dir = pjoin(opt.data_root, 'rep135')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.dim_pose = 251
        opt.max_motion_length = 196
        opt.max_motion_frame = 196
        opt.max_motion_token = 55
    else:
        raise KeyError('Dataset not recognized')

    if not hasattr(opt, 'unit_length'):
        opt.unit_length = 4

    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    opt_dict.update(kwargs) # Overwrite with kwargs params

    return opt