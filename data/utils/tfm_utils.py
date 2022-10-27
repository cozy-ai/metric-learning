from torchvision import transforms

def get_transform_layer(tfm_name, *tfm_args, **tfm_kwargs):
    """
    Return a transform layer given the name of the layer.
    """
    tfm_name = tfm_name.lower()
    if tfm_name == 'random_affine':
        tfm_fn = transforms.RandomAffine
    elif tfm_name == 'resize':
        tfm_fn = transforms.Resize
    elif tfm_name == 'normalize':
        tfm_fn = transforms.Normalize
    elif tfm_name == 'to_tensor':
        tfm_fn = transforms.ToTensor
    elif tfm_name == 'random_resized_crop':
        tfm_fn = transforms.RandomResizedCrop
    elif tfm_name == 'random_horizontal_flip':
        tfm_fn = transforms.RandomHorizontalFlip
    elif tfm_name == 'random_apply':
        if 'transforms' in tfm_kwargs:
            tfms = tfm_kwargs.pop('transforms')
        else:
            tfms = tfm_args.pop(0)
        if isinstance(tfms[0], str):
            tfms = [tfms]
        fns = []
        for fn_name, fn_args, fn_kwargs in tfms:
            fns.append(get_transform_layer(fn_name, *fn_args, **fn_kwargs))
        tfm_kwargs['transforms'] = fns
        tfm_fn = transforms.RandomApply
    elif tfm_name == 'color_jitter':
        tfm_fn = transforms.ColorJitter
    elif tfm_name == 'random_grayscale':
        tfm_fn = transforms.RandomGrayscale
    elif tfm_name == 'gaussian_blur':
        tfm_fn = transforms.GaussianBlur
    else:
        raise ValueError()
    return tfm_fn(*tfm_args, **tfm_kwargs)

def get_transforms(tfm_layers):
    """
    Return the composed sequence of transform layers.
    """
    layers = []
    for tfm_name, tfm_args, tfm_kwargs in tfm_layers:
        layers.append(get_transform_layer(tfm_name, *tfm_args, **tfm_kwargs))
    return transforms.Compose(layers)

def get_perturb_fn(x):
    """
    Return a perturbation function, given the name of the puerturbation function.
    """
    raise NotImplementedError()