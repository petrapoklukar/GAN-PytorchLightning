python create_config.py 'example_mnist_config' \
    --gantraininput-epochs 15 \
    --gantraininput-snapshot 5 \
    --gantraininput-random-seed 1111 \
    --datainput-batch-size 128 \
    --datainput-num-workers 0 
    # --ganbaseinput-generator-dropout 0.3 \
    # --ganbaseinput-generator-latent-dim 100 \
    # --ganbaseinput-representation-dim 1000 \
    # --ganbaseinput-discriminator-dropout None \
    # --ganbaseinput-output-channels None \
    # --datainput-data-dir None \
