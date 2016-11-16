# Generating images part by part with composite generative adversarial networks
Tensorflow implementation of the paper "Generating images part by part with composite generative adversarial networks"as

The Composite GANs[1] (CGAN) disentangles complicated factors of images with multiple generators in which each generator generates some part of the image. Those parts are combined by an alpha blending process to create a new single image. For example, it can generate background, face, and hair sequentially with three generators. There is no supervision on what each generator should generate.
waa

<div align="center">
  <img width="600px" src="http://i.imgur.com/jePfTCx.png"><br><br>
</div>

<div align="center">
  <img width="600px" src="http://i.imgur.com/hpZ9DuP.png"><br><br>
</div>


#### Dependencies
* tensorflow 0.10.0rc0
* h5py 2.3.1
* scipy 0.18.0
* Pillow 3.1.0

#### How to use
1. First, go into the 'data' directory and download celebA dataset.

```
cd data
python download.py celebA
```

2. Preprocess the celebA dataset to create a hdf5 file. It resizes images to 64*64.

```
python preprocess.py
```

3. Finally, go into the 'code' directory and run 'main_cgan_triple_alpha_celeba.py'.

```
cd ../code
python main_cgan_triple_alpha_celeba.py
```

You can see samples in 'samples' directory.

