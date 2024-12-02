import sys

def parse_arguments():
    # First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert sys.argv[0] in ['INDEPENDENT','SEQUENTIAL','JOINT','STANDARD']
    experiment = sys.argv[0]
    log_folder=sys.argv[1]
    data_folder=sys.argv[2]
    seed=sys.argv[3]
    return experiment,log_folder,data_folder

def run_experiment(experiment,args):
    from EECS-train import (
        train_X_to_C,   # Get_concepts
        train_oracle_C_to_y_and_test_on_Chat,  # Independent
        train_Chat_to_y_and_test_on_Chat,  # Sequential
        train_X_to_C_to_y,  # Joint
        train_X_to_y  # Standard
    )

    if experiment == "Independent":
        train_X_to_C(*args)
        train_oracle_C_to_y_and_test_on_Chat(*args)
    elif experiment == "Sequential":
        train_X_to_C(*args)
        train_Chat_to_y_and_test_on_Chat(*args)
    elif experiment == 'Joint':
        train_X_to_C_to_y(*args)
    elif experiment == 'Standard':
        train_X_to_y(*args)

    )
    

if __name__ == '__main__':

    import torch
    import numpy as np

    experiment,args = parse_arguments()

    # Seeds
    np.random.seed(args[0].seed)
    torch.manual_seed(args[0].seed)

    run_experiment(experiment,args)
