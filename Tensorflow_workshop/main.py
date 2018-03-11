import workshop_utils


## ----[2.creat Bin input]-----%%%
# let's assume we have a folder with sub folders containing images from different classes
experience = 'binary'
workshop_utils.CreateBin('../binaryDataset',experience)
workshop_utils.CreateCSV('../binaryDataset',experience)
experience = 'cifar10'
workshop_utils.CreateBin('../cifar10',experience)
workshop_utils.CreateCSV('../cifar10',experience)


# ----[1. network training]-----%%%
experience = 'binary'; checkpoint_dir  = './exp_binary_scratch'; classes_num = 2
phase = 'train'
workshop_utils.network_train_validate(experience, checkpoint_dir, phase,classes_num)
#----------------------------------------------------------------
experience = 'binary'; checkpoint_dir  = './exp_binary_scratch'; classes_num = 2
phase = 'validate'
workshop_utils.network_train_validate(experience, checkpoint_dir, phase,classes_num)
#----------------------------------------------------------------
experience = 'cifar10'; checkpoint_dir  = './exp_cifar10_scratch'; classes_num = 10
phase = 'train'
workshop_utils.network_train_validate(experience, checkpoint_dir, phase,classes_num)
#----------------------------------------------------------------
experience = 'cifar10'; checkpoint_dir  = './exp_cifar10_scratch'; classes_num = 10
phase = 'validate'
workshop_utils.network_train_validate(experience, checkpoint_dir, phase,classes_num)
#----------------------------------------------------------------
#----[2. fine-tuning from a pre-trained models]-----%%%
experience = 'binary'; checkpoint_dir  = './exp_binary_finetuning'; classes_num = 2
phase = 'train'
# fine-tune cifar10 network to the given binary problem 
workshop_utils.network_train_validate(experience, checkpoint_dir, phase,classes_num,netpre_path='./exp_cifar10_scratch')
#----------------------------------------------------------------
experience = 'binary'; checkpoint_dir  = './exp_binary_finetuning'; classes_num = 2
phase = 'validate'
# fine-tune cifar10 network to the given binary problem 
workshop_utils.network_train_validate(experience, checkpoint_dir, phase,classes_num)

