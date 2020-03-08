import numpy
from PIL import Image
from mxnet import gluon, nd


def load_synset(synset_path="models/synset.txt"):
    with open(synset_path, "r") as synset:
        return [line.split()[1] for line in synset]

def imread(filename="misc/image.jpg"):
    img = Image.open(filename)
    to_resize = 256
    to_crop = 224
    mean_vec = numpy.array([0.485, 0.456, 0.406])
    stddev_vec = numpy.array([0.229, 0.224, 0.225])

    # resize
    w, h = img.size
    scale = float(to_resize/w) if w <= h else float(to_resize/h)
    img = img.resize((int(w*scale), int(h*scale)), Image.ANTIALIAS)
    # crop
    w, h = img.size
    left = int((w-to_crop)/2)
    right = int((w+to_crop)/2)
    top = int((h-to_crop)/2)
    bot = int((h+to_crop)/2)
    img = img.crop((left, top, right, bot))
    # to tensor
    img = numpy.array(img).transpose(2, 0, 1).astype('float32')
    norm_img_data = numpy.zeros(img.shape).astype('float32')
    for i in range(img.shape[0]):
        norm_img_data[i,:,:] = (img[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data.reshape(1, 3, 224, 224).astype('float32')

def softmax(x):
    x = x.reshape(-1)
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(numpy.array(result)).tolist()

def check_error(proc_output, network, epsilon=1e-3):
    net = gluon.SymbolBlock.imports('mobilenet1.0-symbol.json', ['data'], 'mobilenet1.0-0000.params')
    if 'resnet' in network:
        net = gluon.SymbolBlock.imports('resnet101_v1-symbol.json', ['data'], 'resnet101_v1-0000.params')
    gluon_output = net(nd.array(imread())).asnumpy()
    error = numpy.max(proc_output - gluon_output)
    print(f"Error: {error}\ngluon:\n{gluon_output}\n{network}:\n{proc_output}")
    assert error < epsilon
