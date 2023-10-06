import torch
import io


# count trainable parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# freeze given neural network

def freeze_model(model):    
    '''
    Function to freeze the layers of a model
    
    '''
    
    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
# unfreeze params of given model
def unfreeze_model(model):
    '''
    Function to unfreeze the layers of a model
    
    '''
    
    # unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
        

def get_model_size(model):
    """Returns size of PyTorch model in bytes"""
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024**2)
    print("Model size (MB):", model_size)
    # return GB size too
    #calculate size in GB
    model_size_g = model_size / 1024
    return model_size, model_size_g


def get_full_model_size(model):
    # Parameter sizes
    param_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    
    # Buffer sizes
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # State dict sizes
    state_dict = model.state_dict()
    state_stream = io.BytesIO()
    torch.save(state_dict, state_stream)
    state_dict_size = len(state_stream.getbuffer())
    
    total_size = (param_size + buffer_size + state_dict_size) / (1024**2)
    print("Total size (MB):", total_size)
    total_size_g = total_size / 1024
    return total_size, total_size_g