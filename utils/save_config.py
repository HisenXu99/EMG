import json

def save_config(file, model_name, data_shape, win_len, win_stride, lr, epoch):
    config={"model":model_name, "win_len":win_len, "win_stride":win_stride, "lr":lr, "epoch":epoch}
    config.update(data_shape)
    with open( file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(config))