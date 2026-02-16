cd mogen
cd checkpoints

# molingo 263
cd t2m
echo -e "Downloading SAE model for 263-dim HumanML3D representation"
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/2kdZCFrbJSHgFoy/download/sae_l2_4_16_1024_d3_kl_1e-05_zero_cos_0.001.zip
echo -e "Downloading MoLingo model for 263-dim HumanML3D representation"
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/J4Fcqmq6agD4HaX/download/pretrained_model_263.zip
echo -e "Unzipping SAE model 263"
unzip sae_l2_4_16_1024_d3_kl_1e-05_zero_cos_0.001.zip
echo -e "Unzipping MoLingo model 263"
unzip pretrained_model_263.zip
echo -e "Cleaning SAE model 263"
rm sae_l2_4_16_1024_d3_kl_1e-05_zero_cos_0.001.zip
echo -e "Cleaning MoLingo model 263"
rm pretrained_model_263.zip

cd ../ms/

# molingo 272
echo -e "Downloading SAE model for 272-dim HumanML3D representation"
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/CgdDmnRKb8ERNca/download/sae_ms_l2_4_16_1024_d3_kl_1e-05_zero.zip
echo -e "Downloading MoLingo model for 272-dim HumanML3D representation"
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/GHG7RtsqFLRysEk/download/pretrained_model_272.zip
echo -e "Unzipping SAE model 272"
unzip sae_ms_l2_4_16_1024_d3_kl_1e-05_zero.zip
echo -e "Unzipping MoLingo model 272"
unzip pretrained_model_272.zip
echo -e "Cleaning SAE model 272"
rm sae_ms_l2_4_16_1024_d3_kl_1e-05_zero.zip
echo -e "Cleaning MoLingo model 272"
rm pretrained_model_272.zip

cd ../../../

echo -e "Downloading done!"