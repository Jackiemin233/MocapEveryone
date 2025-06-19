
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
# pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

pip install -r requirements.txt

# NOTE: ONLY branch old_master are able to run
pip install git+https://github.com/nghorbani/human_body_prior.git@old_master
pip install dotmap omegaconf loguru dadaptation configer
pip install accelerate==0.31.0
# install fairmotion (modified)
cd fairmotion
pip install -e .
cd ..