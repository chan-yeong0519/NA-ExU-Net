import os
from itertools import chain
import torch

def get_args():
    system_args = {
        # expeirment info
        'project'       : 'NAExUnet',
        'name'          : 'NA-ExU-Net_final_code',
        'tags'          : ['NAExUnet_final'],
        'description'   : '',

        # local
        'path_logging'  : '/results',

        # VoxCeleb1 DB
        'path_vox1_train'   : '',
        'path_vox1_test'    : '',
        'path_vox1_trials'  : '',

        # musan DB
        'path_musan'        : '',

        # device
        'num_workers'   : 20,
        'usable_gpu'    : '',
        'tqdm_ncols'    : 90,
        'path_scripts'     : os.path.dirname(os.path.realpath(__file__))
    }
    
    experiment_args = {
        # env
        'epoch'                     : 188,
        'batch_size'                : 320,
        'number_iteration_for_log'  : 50,
        'rand_seed'                 : 1234,
        'flag_reproduciable'        : True,
        
        # train process
        'do_train_feature_enhancement'  : True,
        'do_train_code_enhancement'     : True,

        # optimizer
        'optimizer'                 : 'adam',
        'amsgrad'                   : True,
        'learning_rate_scheduler'   : 'warmup',
        'lr_start'                  : 1e-7,
        'lr_end'                    : 1e-7,
        'number_cycle'              : 40,
        'step_size'                 : 2,
        'step_gamma'                : 0.95,
        'warmup_number_cycle'       : 1,
        'T_mult'                    : 1.5,
        'eta_max'                   : 1e-2,
        'gamma'                     : 0.5,
        'weigth_decay'              : 1e-4,


        # criterion
        'encoder_classification_loss'                       : 'softmax',
        'noise_classification_loss'                         : 'softmax',
        'code_classification_loss'                          : 'aam_softmax',
        'snr_reg_loss'                                      : 'huber_loss',
        'enhancement_loss'                                  : 'mse',
        'code_enhacement_loss'                              : 'angleproto',
        'weight_classification_loss'                        : 1,
        'weight_code_enhancement_loss'                      : 1,
        'weight_feature_enhancement_loss'                   : 1,

        # model
        'first_kernel_size'     : 7,
        'first_stride_size'     : (2,1),
        'first_padding_size'    : 2,
        'encoder_split_level'   : 2,
        'ext_skip_conv_level'   : 3,
        'ext_reduction_ratio'   : [1,1,2,2,1],
        'l_channel'             : [16, 32, 64, 128],
        'l_num_convblocks'      : [3, 4, 6, 3],
        'code_dim'              : 128,
        'stride'                : [1,2,2,1],

        # data
        'nb_utt_per_spk'    : 2,
        'max_seg_per_spk'   : 500,
        'winlen'            : 400,
        'winstep'           : 160,
        'train_frame'       : 254,
        'nfft'              : 1024,
        'samplerate'        : 16000,
        'nfilts'            : 64,
        'premphasis'        : 0.97,
        'winfunc'           : torch.hamming_window,
        'test_frame'        : 382,
        'spec_mask_F'       : 8,
        'spec_mask_T'       : 10,
        'aam_margin'        : 0.2,
        'aam_scale'         : 30,
    }

    # set args (system_args + experiment_args)
    args = {}
    for k, v in chain(system_args.items(), experiment_args.items()):
        args[k] = v

    return args, system_args, experiment_args
