import mxnet
import mxnet.gluon.model_zoo.vision.resnet as res

net = a = res.resnet50_v2()
net(mxnet.sym.Variable("data")).save("resnet50_v2-symbol.json")
