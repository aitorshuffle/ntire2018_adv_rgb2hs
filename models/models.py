def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'rgb2hs')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
