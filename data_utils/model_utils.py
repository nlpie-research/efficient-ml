import torch



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