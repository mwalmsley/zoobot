from zoobot.estimators import define_model

if __name__ == '__main__':

    pretrained_checkpoint = '/media/walml/alpha/beta/decals/long_term/models/decals_dr_train_labelled_m0/in_progress'
    initial_size = 300 # images will be resized from disk (424) to this before preprocessing
    # should match how the model was trained
    original_output_dim = 34
    crop_size = int(initial_size * 0.75)
    resize_size = 224  # 224 for paper

    # get headless model (inc. augmentations)
    # base_model = define_model.load_model(
    #   pretrained_checkpoint,
    #   include_top=False,
    #   input_size=initial_size,  # preprocessing above did not change size
    #   crop_size=crop_size,  # model augmentation layers apply a crop...
    #   resize_size=resize_size,  # ...and then apply a resize
    #   output_dim=original_output_dim
    # )
    # base_model.summary()
    # print(base_model.layers)

    base_model = define_model.get_model(original_output_dim, initial_size, crop_size, resize_size, include_top=False)

    # blocks are:
    # conv, batch, act, depth, batch, act, pool, reshape, conv, conv, multiply

    # conv, batch, 
    
    # conv, batch, act, depth, batch, act, pool, reshape, conv, conv, multiply

    # conv batch, dropout, merge

    # conv, batch, activation, pool, reshape


    effnet_layers = base_model.layers[-1].layers[0].layers

    for layer in effnet_layers:
      print(layer.name)
    exit()
    # effnet has blocks 1-7, + 'top'

    # print(effnet_layers)
    # batch = effnet_layers[-2]
    # print(batch.weights)

    # base_model.build(input_shape=(300, 300, 1))
    # conv_weights = base_model.layers[-1].layers[0].layers[-3].kernel
    # print(conv_weights.numpy().mean(), conv_weights.numpy().var(), conv_weights.numpy().min(), conv_weights.numpy().max())

    define_model.load_weights(base_model, pretrained_checkpoint, expect_partial=True)
    batch = base_model.layers[-1].layers[0].layers[-2]
    print(batch.weights)
    # conv_weights = base_model.layers[-1].layers[0].layers[-3].kernel
    # print(conv_weights.numpy().mean(), conv_weights.numpy().var(), conv_weights.numpy().min(), conv_weights.numpy().max())


    # # if allowed to finish, best model will be saved to log_dir/checkpoints/models/final for later use (see make_predictions.py)
    # # else, latest model will be log_dir/checkpoints
    # # partially_retrained_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured/v15/checkpoints/final_model/final'
    # # partially_retrained_dir = os.path.join(log_dir, 'checkpoints')
    # partially_retrained_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured/v42/checkpoints'
    # # partially_retrained_dir = '/raid/scratch/walml/galaxy_zoo/temp/finetune_featured/v15/checkpoints'  # the checkpoint, not the folder - tf understands
    # # weights_loc = os.path.join(log_dir, 'checkpoints')
    # define_model.load_weights(model=base_model, weights_loc=partially_retrained_dir, expect_partial=True)

    # print(conv_weights.numpy().mean(), conv_weights.numpy().var())