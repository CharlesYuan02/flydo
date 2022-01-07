# Paths to images and labels
train_data_dir = "model/data/train"
train_coco = "model/data/drone.json"

# Parameters for dataloader
train_shuffle_dl = True
num_workers_dl = 4

# Training hyperparameters
train_batch_size = 1
lr = 1e-3
momentum = 0.9
weight_decay = 1e-3
num_classes = 2 # Two classes, only target class or background
num_epochs = 10