import os


def logdir(config):
    if config.architecture_type == 'rnn':
        layers = '_'.join([str(h) for h in config.hidden_sizes])
        type = config.cell_type
        if config.bidirectional:
            type += 'bidir'
    else:
        layers = '_'.join([str(h) for h in config.filter_size])
        type = 'cnn'
        if config.gru_include:
            type += 'gru'

    return os.path.join('models', 'model_%s_pred%s_%s_hl%s_bs%d_lr%.5f_fo%.5f_do%.5f' % (
        config.architecture_type,
        config.pred_type,
        type,
        layers,
        config.batch_size,
        config.learning_rate,
        config.frame_size,
        config.dropout))
