import yaml
import random, torch, os
import numpy as np
import pandas as pd

from scipy import stats
from tqdm import tqdm

from utils.train import trainNet, test, get_performance_dict
from utils.dataloader import sp_loc_dataset, collate_fn

from models.MHSA import TransEncoder


def load_config(path):
    """
    Loads config file:
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for _, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_trainedNets(config, model, train_loader, val_loader, device, log_dir):
    print("LocationPrediction - utils - train.py - Running get_trainedNets --- ")
    best_model, performance = trainNet(config, model, train_loader, val_loader, device, log_dir=log_dir)
    performance["type"] = "vali"

    return best_model, performance


def get_test_result(config, best_model, test_loader, device):
    return_dict, result_arr_user = test(config, best_model, test_loader, device)

    performance = get_performance_dict(return_dict)
    performance["type"] = "test"
    # print(performance)

    result_user_df = pd.DataFrame(result_arr_user).T
    result_user_df.columns = [
        "correct@1",
        "correct@3",
        "correct@5",
        "correct@10",
        "rr",
        "ndcg",
        "total",
    ]
    result_user_df.index.name = "user"

    return performance, result_user_df


def get_models(config, device):
    total_params = 0

    model = TransEncoder(config=config, total_loc_num=config.total_loc_num).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("LocationPrediction - utils - utils.py - Total number of trainable parameters: ", total_params)

    return model


def get_dataloaders(config):

    kwds_train = {
        "shuffle": False,
#        "shuffle": True,
        "num_workers": config["num_workers"],
#        "drop_last": False,
        "drop_last": True,
        "batch_size": config["batch_size"],
        "pin_memory": True,
    }
    kwds_val = {
        "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True,
    }
    kwds_test = {
        "shuffle": False,
        "num_workers": config["num_workers"],
        "batch_size": config["batch_size"],
        "pin_memory": True,
    }

    print("LocationPrediction - utils - utils.py - get_dataloaders - About to create dataset_train ---")
    dataset_train = sp_loc_dataset(
        config.source_root,
        data_type="train",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
    )
    print("LocationPrediction - utils - utils.py - get_dataloaders - About to create dataset_val ---")
    dataset_val = sp_loc_dataset(
        config.source_root,
        data_type="validation",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
    )
    print("LocationPrediction - utils - utils.py - get_dataloaders - About to create dataset_test ---")
    dataset_test = sp_loc_dataset(
        config.source_root,
        data_type="test",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
        day_selection=config.day_selection,
    )
    print("LocationPrediction - utils - utils.py - get_dataloaders - Completed creation of train, validation, and test datasets ---")

    fn = collate_fn

    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=fn, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=fn, **kwds_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=fn, **kwds_test)
    
    print("LocationPrediction - utils - utils.py - get_dataloaders - len(dataset_train) - ",len(dataset_train))
    print("LocationPrediction - utils - utils.py - get_dataloaders - dataset_train - ",dataset_train)
    print("LocationPrediction - utils - utils.py - get_dataloaders - len(dataset_val) - ",len(dataset_val))
    print("LocationPrediction - utils - utils.py - get_dataloaders - len(dataset_test) - ",len(dataset_test))
    print("LocationPrediction - utils - utils.py - get_dataloaders - len(train_loader) - ",len(train_loader))
    print("LocationPrediction - utils - utils.py - get_dataloaders - len(val_loader) - ",len(val_loader))
    print("LocationPrediction - utils - utils.py - get_dataloaders - len(test_loader) - ",len(test_loader))
    
#    for i in range(len(dataset_train)):
#    	print("LocationPrediction - utils - utils.py - get_dataloaders - i - ",i)
#    	print ("LocationPrediction - utils - utils.py - get_dataloaders - dataset_train[i] - ",dataset_train[i])

    
#    for i, inputs in enumerate(train_loader):
#    	print("LocationPrediction - utils - utils.py - get_dataloaders - i - ",i)

       
#    print("LocationPrediction - utils - utils.py - get_dataloaders - next(iter(train_loader)) - ",next(iter(train_loader)))
#    print("LocationPrediction - utils - utils.py - get_dataloaders - next(iter(val_loader)) - ",next(iter(val_loader)))
#    print("LocationPrediction - utils - utils.py - get_dataloaders - next(iter(test_loader)) - ",next(iter(test_loader)))
#    print("LocationPrediction - utils - utils.py - get_dataloaders - list(train_loader) - ",list(train_loader))
#    print("LocationPrediction - utils - utils.py - get_dataloaders - list(val_loader) - ",list(val_loader))
#    print("LocationPrediction - utils - utils.py - get_dataloaders - list(test_loader) - ",list(test_loader))

    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader
