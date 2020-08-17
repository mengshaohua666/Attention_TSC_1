from torch import nn

from models.resnet import resnet18


class Attention(nn.Module):

    def __init__(self, size, **kwargs):
        super(Attention, self).__init__()
        self.trainable = True
        self.size = size

    def build(self, input_shape):
        # print(input_shape)
        self.q = self.add_weight(name='q', shape=(1, self.size), initializer='ones', trainable=self.trainable)
        super(Attention, self).build(input_shape)

    def forward(self, x):
        stream1, stream2 = x[0], x[1]

        d1 = Lambda(lambda x: K.sum(x * self.q, axis=1, keepdims=True))(stream1)  # sum over second axis
        d2 = Lambda(lambda x: K.sum(x * self.q, axis=1, keepdims=True))(stream2)
        ds = Concatenate(axis=1)([d1, d2])

        # d1 and d2 and of size (bs, 1) individually
        # ds of size (bs, 2)

        tmp = Softmax(axis=0)(ds)
        # print(tmp._keras_shape)
        w1 = Lambda(lambda x: x[:, 0])(tmp)
        w2 = Lambda(lambda x: x[:, 0])(tmp)

        w1 = Lambda(lambda x: K.expand_dims(x, -1))(w1)
        w2 = Lambda(lambda x: K.expand_dims(x, -1))(w2)
        # print(w1._keras_shape)
        # print(w1.shape)
        # print(w2.shape)

        stream1 = Lambda(lambda x: x[0] * x[1])([stream1, w1])
        stream2 = Lambda(lambda x: x[0] * x[1])([stream2, w2])
        # result = Lambda(lambda x: x[0]+x[1])([stream1, stream2])
        result = Add()([stream1, stream2])
        # print(result.shape)
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# stream1 = Input(batch_shape= (2, 10*10*2048,))
# stream2 = Input(batch_shape = (2,10*10*2048,))
# shapes = 2048*100
# stream1 = Input((shapes,))
# stream2 = Input((shapes,))
# output = Attention(size=shapes)([stream1, stream2])
# print(output._keras_shape)
# model = Model(inputs=[stream1, stream2], outputs=output)
# print(model.summary())

def attention_model(classes, backbone, shape):
    stream1 = resnet18(pretrained=True)
    stream2 = resnet18(pretrained=True)

    stream1.name = 'stream1'
    stream2.name = 'stream2'

    input1 = Input(shape)
    input2 = Input(shape)
    output1 = stream1(input1)
    output2 = stream2(input2)

    # stream1 = Flatten()(stream1)
    # stream2 = Flatten()(stream2)
    # print(stream1.shape)
    output = Attention(size=output1.shape[1])([output1, output2])
    # print(output.shape)
    if classes == 1:
        output = Dense(classes, activation='sigmoid', name='predictions')(output)
    else:
        output = Dense(classes, activation='softmax', name='predictions')(output)
    # print(output.shape)
    return Model(inputs=[input1, input2], outputs=output)

# model = attention_model(2)
# print(model.summary())
# from keras.utils import plot_model
# plot_model(model, 'model.png', show_shapes=True)
