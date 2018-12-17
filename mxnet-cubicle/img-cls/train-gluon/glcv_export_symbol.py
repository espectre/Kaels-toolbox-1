import mxnet
import mxnet.gluon.model_zoo.vision.resnet as res

net = res.resnet50_v2()
net(mxnet.sym.Variable("data")).save("resnet50_v2-symbol.json")

net = res.resnet50_v2()
net(mxnet.sym.Variable("data")).save("resnet50_v2-symbol.json")
