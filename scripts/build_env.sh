cd ..

git config --global user.name "yhbcode000"
git config --global user.email "hobart.yang@qq.com"

apt update -y
apt upgrade -y
apt install pigz aria2 -y

pip install uv
uv sync --inexact
uv pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
uv pip install -e . --no-build-isolation
