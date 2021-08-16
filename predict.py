# Imports python modules
import torch

# Imports functions created for this program
from predict_utils.get_input_args import get_input_args
from predict_utils.load_checkpoint import load_checkpoint
from predict_utils.process_image import process_image
from predict_utils.print_results import print_results


# Predict the class from an image file
def predict(image, model, gpu, top_k=5):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    if gpu:        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    model.to(device)
    
    top_p, top_class = torch.exp(model.forward(image)).topk(top_k, dim=1)
    
    return top_p[0], top_class[0]


# Main program function defined below
def main():
    # Get command line arguments
    in_arg = get_input_args()
        
    # Check command line arguments
    # TODO: Nice to have: check_command_line_arguments(in_arg)
    
    # Load model from checkpoint
    model = load_checkpoint(in_arg.checkpoint)
    
    # Process image
    image = process_image(in_arg.img_dir)

    # Predict
    top_p, top_class = predict(image, model, in_arg.gpu, in_arg.top_k)
    
    # Print class and a probability
    print_results(top_p, top_class, in_arg.category_names)
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
