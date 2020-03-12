
# https://github.com/pfnet-research/chainer-disentanglement-lib/blob/master/bin/download_dataset.sh

export DISENTANGLEMENT_LIB_DATA=./data/
cd ${DISENTANGLEMENT_LIB_DATA}

echo "Check dSprites"
if [[ ! -d "dsprites" ]]; then
  echo "Downloading dSprites dataset."
  mkdir dsprites
  wget -O dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
  echo "Downloading dSprites completed!"
fi

echo "Check mpi3d_toy"
# if [[ ! -d "mpi3d_toy" ]]; then
#   echo "Downloading mpi3d_toy dataset."
#   mkdir mpi3d_toy
#   wget -O mpi3d_toy/mpi3d_toy.npz https://storage.googleapis.com/disentanglement_dataset/data_npz/sim_toy_64x_ordered_without_heldout_factors.npz
#   echo "Downloading mpi3d_toy completed!"
# fi

echo "Check cars3d"
if [[ ! -d "cars3d" ]]; then
  echo "Downloading cars3d dataset."
  mkdir cars3d
  wget -O cars3d.tar.gz http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
  tar -zxf cars3d.tar.gz
  echo "Downloading cars3d completed!"
fi
