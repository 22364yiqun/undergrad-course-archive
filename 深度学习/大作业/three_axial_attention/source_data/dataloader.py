from .bear import bearing

def load_data(opt):
    data = bearing(signal_size=opt['signal_size'],
                    enc_size=opt['enc_size'],
                    batch_size=opt['batch_size'],
                    num_workers=opt['num_workers'],
                    domain=opt['domain'],
                    num_class=opt['label_class'])
    dataloader = data.dataloader_maker()
    label_class = data.label_class
    
    return dataloader,label_class