Training
========

The Intel Open Image Denoise source distribution includes a Python-based neural
network training toolkit (see the `training` directory), which can be used to
train the denoising filter models using image datasets provided by the user.
The toolkit consists of multiple command-line tools (e.g. dataset preprocessing,
training, inference, image comparison) that together can be used to train and
evaluate models.


Prerequisites
-------------

Before you can run the training toolkit you need the following prerequisites:

-   Python 3.7 or later

-   [PyTorch](https://pytorch.org/) 1.4 or later

-   [NumPy](https://numpy.org/) 1.17 or later

-   [OpenImageIO](http://openimageio.org/) 2.1 or later

-   [TensorBoard](https://www.tensorflow.org/tensorboard) 2.1 or later (*optional*)

The training toolkit has been tested only on Linux and other operating systems
are currently not supported.

Datasets
--------

A dataset should consist of a collection of noisy and corresponding noise-free
reference images. It is possible to have more than one noisy version of the
same image in the dataset, e.g. rendered at different samples per pixel and/or
using different seeds.

The training toolkit expects to have all datasets (e.g. training, validation)
in the same parent directory (e.g. `data`). Each dataset is stored in its own
subdirectory (e.g. `train`, `valid`), which can have an arbitrary name.

The images must be stored in [OpenEXR](https://www.openexr.com/) format (`.exr`
files) and the filenames must have a specific format but the files can be stored
in an arbitrary directory structure inside the dataset directory. The only
restriction is that all versions of an image (noisy images and the reference
image) must be located in the same directory. Each feature of an image (e.g.
color, albedo) must be stored in a separate image file, i.e. multi-channel EXR
image file are not supported. The format of the image filenames has to be the
following:

```regexp
[^_]+_([0-9]+(spp)?|ref|gt)\.(hdr|ldr|alb|nrm)\.exr
```