def to_numpy(pytorch_tensor):
    """
    This util method converts a PyTorch tensor to numpy then returns it.
    """
    return (
        pytorch_tensor.detach().cpu().numpy()
        if pytorch_tensor.requires_grad
        else pytorch_tensor.cpu().numpy()
    )
