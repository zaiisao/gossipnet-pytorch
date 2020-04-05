def xavierInitialization(model, biasValue=0):
    """
        Initialize the model with xavier initialization
        Input:
            layer: Could be single layer or a list of layers
    """
    if isinstance(model, list):
        for layer in model:
            xavierInitialization(layer, biasValue)
    else:
        for module in model.modules():
            print (module)