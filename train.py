# Imports python modules

# Imports functions created for this program
from train_utils.get_input_args import get_input_args
from train_utils.get_loader import get_loader
from train_utils.create_model import create_model
from train_utils.train_model import train_model
from train_utils.save_model import save_model


# Main program function defined below
def main() -> None:
    """
    Bunch of instruction for creating and training model.
    """
    # Get command line arguments
    in_arg = get_input_args()

    # Check command line arguments
    # TODO: Nice to have: check_command_line_arguments(in_arg)
    
    # Get data
    loader = get_loader(in_arg.data_dir)
    
    # Get model
    model = create_model(in_arg.arch, in_arg.hidden_units)
    
    # Train model
    model, optimizer = train_model(model, loader, in_arg.gpu, in_arg.learning_rate, in_arg.epochs)
    
    # Save model
    save_model(model, optimizer, in_arg.save_dir)
        

# Call to main function to run the program
if __name__ == "__main__":
    main()
