cd mogen
cd checkpoints
mkdir t2m

cd t2m
echo -e "Downloading evaluation models for TMR-263 representation"

echo -e "Downloading evaluation models for MARDM-67 representation"
gdown --fuzzy https://drive.google.com/file/d/1ejiz4NvyuoTj3BIdfNrTFFZBZ-zq4oKD/view?usp=sharing
echo -e "Unzipping humanml3d evaluators"
unzip evaluators_humanml3d.zip
mv text_mot_match text_mot_match_mardm

echo -e "Cleaning humanml3d evaluators zip"
rm evaluators_humanml3d.zip

echo -e "Downloading the length estimator"
gdown --fuzzy https://drive.google.com/file/d/1nWoEcN4rEFKi4Xyf_ObKinDmSQNPKXgU/view?usp=sharing
echo -e "Unzipping the length estimator"
unzip length_estimator.zip
echo -e "Cleaning the length estimator"
rm length_estimator.zip

cd ../ms/
echo -e "Downloading evaluation models for MS-272 representation"
wget https://huggingface.co/lxxiao/MotionStreamer/resolve/main/Evaluator_272/epoch=99.ckpt
cd ../../../

echo -e "Downloading done!"