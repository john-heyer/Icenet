import tensorflow as tf

class ConvNet:

    def __init__(self, regularizer=None, data_format='NHWC'):
        self.data_format = data_format
        self.regularizer = regularizer

    # ====================================================

    @staticmethod
    def bot_depth(bot):
        bot_shape = bot.get_shape().as_list()
        return bot_shape[3]

    @staticmethod
    def bot_size(bot):
        bot_shape = bot.get_shape().as_list()
        return bot_shape[1], bot_shape[2]

    @staticmethod
    def one_pad(val):
        return (1,) + val + (1,)

    def init_var(self, shape, name, regularize=False):
        var = tf.get_variable(name, 
            shape=shape,
            initializer=tf.uniform_unit_scaling_initializer(factor=1.43), 
            regularizer=self.regularizer if regularize else None)
        return var

    def conv(self, bot, ksize, nout, stride=(1,1), name='conv'):
        with tf.variable_scope(name):

            kernel = self.init_var( ksize + (self.bot_depth(bot), nout) , 'kernel', regularize=True)
            conv_layer = tf.nn.conv2d(bot, kernel, strides=self.one_pad(stride), padding='SAME', data_format=self.data_format)
            bias = self.init_var((nout,), 'bias')

            return tf.nn.bias_add(conv_layer, bias, data_format=self.data_format)

    @staticmethod
    def relu(bot):
        return tf.nn.relu(bot)

    def conv_relu(self, *args, **kw_args):
        conv_layer = self.conv(*args, **kw_args)
        relu_layer = self.relu(conv_layer)
        return relu_layer

    def conv_nd(self, bot, ksize_n_minus_one, ksize_n, nout, axis, stride_n_minus_one, stride_n, name='conv_nd'):
        assert ksize_n%2 == 1, "ksize_n must be odd"

        with tf.variable_scope(name):
            
            in_channels = bot.get_shape().as_list()[-1]
            kernels = [self.init_var( ksize_n_minus_one + (in_channels, nout), 'kernel' + str(i), regularize=True) for i in range(ksize_n)]

            # size of bot is [nbatch, x, y, z, t, nchannel]
            bot_shape = bot.get_shape().as_list()
            rank = len(bot_shape)
            axis_size = bot_shape[axis]
            
            shift = ksize_n // 2
            shifted_data = self.shifted_data([tf.identity(bot) for i in range(len(kernels))], rank, shift, axis, axis_size)
            #print("shifted data size: ", len(shifted_data), shifted_data[0].get_shape().as_list())
            # shifted data now has size [3 * [n,x,y,z,t]] and shift is taken care of

            # if axis = t, will now be [3 *[t * [nbatch, x, y, z, nchannel]]
            layers_n_minus_one = [[tf.squeeze(l, axis=[axis]) for l in tf.split(bot, axis_size, axis=axis)] for bot in shifted_data]

            if stride_n > 1:
                layers_n_minus_one = [ n_minus_1_layers[::stride_n] for n_minus_1_layers in layers_n_minus_one]

            conv_n_minus_one_layers = [
                [
                    tf.nn.convolution(
                        layers_n_minus_one[k][l], kernels[k], strides=stride_n_minus_one, padding='SAME'
                    )
                    #tf.nn.conv1d(layers_n_minus_one[k][l], kernels[k],stride=stride_n_minus_one, padding='SAME')
                    for l in range(len(layers_n_minus_one[0]))
                ]
                for k in range(len(kernels))
            ]

            # conv_3d_layers shape: [3 * [t * [nbatch, x, y, z, nchannel] ] ]
            conv_n_minus_one_layers = [[tf.expand_dims(l, axis=axis) for l in k] for k in conv_n_minus_one_layers]
            # inserted squeezed dimension, shape now: [3 * [t * [nbatch, x, y, z, 1, nchannel] ] ]
            reshape_nd = [tf.concat([l for l in k], axis=axis) for k in conv_n_minus_one_layers]
            #reshape_4d shape should now be: [3 * [nbatch, x, y, z, t, nchannel]]

            # -1 in our case of ksize_4 = 3
            # shift =  ksize_4 // 2
            # shifted_data = self.shifted_data(reshape_4d, rank, shift, axis, axis_size)
            bias = self.init_var((nout,), 'bias')
            conv = tf.add_n(reshape_nd)
                     
            return tf.nn.bias_add(conv, bias)

    @staticmethod
    def shifted_data(data, rank, shift, axis, axis_size):
            shifted_data = []
            for k in data:
                # shift right
                if shift > 0:
                    # crop right, pad on left
                    begin = [0 for i in range(rank)]
                    size = [-1 if i != axis else axis_size - shift for i in range(rank)]
                    paddings = [[0,0] if i != axis else [shift, 0] for i in range(rank)]
                # shift left
                elif shift < 0:
                    # crop left, pad on right 
                    begin = [0 if i != axis else -shift for i in range(rank)]
                    size = [-1 for i in range(rank)]
                    paddings = [[0,0] if i != axis else [0, -shift] for i in range(rank)]

                if shift != 0:
                    k = tf.slice(k, begin, size)
                    k = tf.pad(k, paddings)

                shifted_data.append(k)
                shift -= 1
            
            return shifted_data

    def conv_relu_4d(self, *args,**kw_args):
        return self.relu(self.conv_4d(*args, **kw_args))

    def pool(self, bot, ksize, stride, op):
        if op == 'max':
            return tf.nn.max_pool(bot, ksize=ConvNet.one_pad(ksize), strides=ConvNet.one_pad(stride), padding='SAME', data_format=self.data_format)
        elif op == 'ave':
            return tf.nn.avg_pool(bot, ksize=ConvNet.one_pad(ksize), strides=ConvNet.one_pad(stride), padding='SAME', data_format=self.data_format)

    def fc(self, bot, nout, name='fc'):
        with tf.variable_scope(name):
            bot_ndims = bot.get_shape().ndims
            if bot_ndims > 2:
                import functools
                bot_shape = bot.get_shape().as_list()
                bot_reshape = tf.reshape(bot, (bot_shape[0], functools.reduce(lambda x,y: x*y, bot_shape[1:])))
            elif bot_ndims == 2:
                bot_reshape = bot
            else:
                raise Exception("Unknown bottom layer shape.")

            matrix = self.init_var( (bot_reshape.get_shape().as_list()[1], nout), 'weights', regularize=True)
            return tf.matmul(bot_reshape, matrix)

    def fc_relu(self, *args, **kw_args):
        fc_layer = self.fc(*args, **kw_args)
        relu_layer = self.relu(fc_layer)
        return relu_layer


class LLNet(ConvNet):
    def __init__(self, *args, **kw_args):
        super(LLNet, self).__init__(*args, **kw_args)

        self.ll = False

    def init_var(self, shape, name, regularize=False):
        var_name = name
        var_shape = shape
        if self.ll:
            var_name += "_ll"
            var_shape = None
            with tf.variable_scope("var", reuse=True):
                init = tf.get_variable(name).initialized_value()
        else:
            if len(shape) > 1:
                init = tf.orthogonal_initializer()
            else:
                init = tf.zeros_initializer()
        #print(init)

        with tf.variable_scope("var"):
            var = tf.get_variable(var_name, 
                shape=var_shape, 
                initializer=init, 
                regularizer=self.regularizer if regularize else None)
        return var

    def looks_linear(self, bot, op):
        self.ll = False
        pos = op(bot)
        self.ll = True
        neg = op(tf.negative(bot))
        self.ll = False
        return pos - neg

    def relu_conv_ll(self, bot, *args, **kw_args):
        #conv_layer = self.conv(*args, **kw_args)
        relu_layer = self.looks_linear(bot, lambda b: self.conv(self.relu(b), *args, **kw_args))
        return relu_layer

    def relu_fc_ll(self, bot, *args, **kw_args):
        return self.looks_linear(bot, lambda b: self.fc(self.relu(b), *args, **kw_args))


class AbsLLNet(ConvNet):
    def __init__(self, *args, **kw_args):
        super(AbsLLNet, self).__init__(*args, **kw_args)

        self.ll = False

    @staticmethod
    def abs(bot):
        #Check
        return tf.abs(bot)
    
    def init_var(self, shape, name, regularize=False):
        var_name = name
        var_shape = shape
        if self.ll:
            var_name += "_ll"
            init = tf.zeros_initializer()
        else:
            if len(shape) > 1:
                init = tf.orthogonal_initializer()
            else:
                init = tf.zeros_initializer()
        #print(init)

        with tf.variable_scope("var"):
            var = tf.get_variable(var_name, 
                shape=var_shape, 
                initializer=init, 
                regularizer=self.regularizer if regularize else None)
        return var

    def looks_linear(self, bot, op):
        self.ll = False
        left = op(bot)
        self.ll = True
        right = op(self.abs(bot))
        self.ll = False
        return left + right
        
    def relu_conv_ll(self, bot, *args, **kw_args):
        relu_layer = self.looks_linear(bot, lambda b: self.conv(b, *args, **kw_args))
        return relu_layer

    def relu_fc_ll(self, bot, *args, **kw_args):
        return self.looks_linear(bot, lambda b: self.fc(b, *args, **kw_args))



class ILNet(ConvNet):
    def init_var(self, shape, name, regularize=False):
        var = tf.get_variable(name, 
            shape=shape, 
            initializer=tf.uniform_unit_scaling_initializer(factor=1.43), 
            regularizer=self.regularizer if regularize else None)
        return var

    @staticmethod
    def relu(bot):
        return tf.concat([tf.nn.relu(bot), tf.negative(tf.nn.relu(tf.negative(bot)))], axis=3) 


class ResNet(ConvNet):

    def res_module(self, bot, nout, ksize, stride, name='res_module'):
        with tf.variable_scope(name):
            highway_factor = 4
            internal_nch = nout//highway_factor

            
            if self.bot_depth(bot) == nout and stride[0]*stride[1] == 1:
                bypass_layer = bot
            else:
                bypass_layer = self.conv_relu(bot, ksize=(1,1), nout=nout, 
                                              stride=stride, name='conv_bypass')

        
            if self.bot_depth(bot) == internal_nch:
                input_layer = bot
            else:
                input_layer = self.conv_relu(bot, ksize=(1,1), nout=internal_nch, 
                                             stride=(1,1), name='conv_downscale')

        
            internal_layer = self.conv_relu(input_layer, ksize=ksize, nout=internal_nch, 
                                            stride=stride, name='conv_internal')
        
            output_layer = self.conv_relu(internal_layer, ksize=(1,1), nout=nout, 
                                          stride=(1,1), name='conv_upscale')

        return self.relu(bypass_layer + output_layer)

class WideResNet(ConvNet):

    def res_module(self, bot, nout, ksize, stride, name='res_module', dropout_p=None):
        with tf.variable_scope(name):
            
            if self.bot_depth(bot) == nout and stride[0]*stride[1] == 1:
                bypass_layer = bot
            else:
                bypass_layer = self.conv(bot, ksize=(1,1), nout=nout, 
                                              stride=stride, name='conv_bypass')

        
            input_layer = self.conv_relu(bot, ksize=ksize, nout=nout, 
                                            stride=stride, name='conv_relu_input')

            if dropout_p is not None:
                internal_layer = tf.nn.dropout(input_layer, keep_prob=dropout_p, name="dropout")
            else:
                internal_layer = input_layer
        
            output_layer = self.conv(internal_layer, ksize=ksize, nout=nout, 
                                          stride=(1,1), name='conv_output')

        return (bypass_layer + output_layer)

def sess(device=None):
    config = tf.ConfigProto()
    if device is not None:
        if isinstance(device, list):
            gpu_string = str(device[0])
            for gpu_id in device[1:]:
                gpu_string += "," + str(gpu_id)
        else:
            gpu_string = str(device)
        config.gpu_options.visible_device_list = gpu_string
    
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    config.allow_soft_placement = True

    return tf.Session(config=config)

class Model:

    def __init__(self, saver_filter=lambda x: True):
        #print([ v for v in tf.global_variables() if True])
        self.saver = None
        self.saver_filter = saver_filter

    def init(self, tf_sess):
        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver([ v for v in tf.global_variables() if self.saver_filter(v)]) #not v.name[:5] == 'unsup'

        tf_sess.run(tf.global_variables_initializer())

    def load(self, tf_sess, filename):
        self.saver.restore(tf_sess, filename)

    def save(self, tf_sess, filename):
        self.saver.save(tf_sess, filename)

    def eval(self, tf_sess, op):
        return tf_sess.run(op)

class Tester(Model):
    def __init__(self, *args, **kw_args):
        Model.__init__(self, *args, **kw_args)

        self.cost_op = self.cost() + tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        self.metric_op = self.metric()
        if not isinstance(self.metric_op, list):
            self.metric_op = [self.metric_op]

        self.test_op = self.metric_op + [self.cost_op]

    #def init(self, tf_sess):
    #    Evaluator.init(self, tf_sess)

    def test(self, tf_sess):
        results = self.eval(tf_sess, self.test_op)
        return [results[-1]], results[:-1]

class Trainer(Tester):
    def __init__(self, learning_rate, *args, **kw_args):
        Tester.__init__(self, *args, **kw_args)

        global_step = tf.Variable(0.0, trainable=False)
        self.optim_op = tf.train.AdamOptimizer(learning_rate=learning_rate(global_step)).minimize(
                self.cost_op, 
                global_step=global_step, 
                colocate_gradients_with_ops=True, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

        self.train_op = self.test_op + [self.optim_op]

    #def init(self, tf_sess):
    #    Tester.init(self, tf_sess)

    def train(self, tf_sess):
        results = self.eval(tf_sess, self.train_op)[:-1]
        return [results[-1]], results[:-1]

"""class Tester:
    def __init__(self, model):
        if not isinstance(model, list):
            model = [model]
        self.model = model

        self.cost_op, test_ops = [], []
        for sub_model in model:
            with sub_model.placement():
                self.cost_op.append(sub_model.cost())

                sub_metric_op = sub_model.metric()
            if not isinstance(sub_metric_op, list):
                sub_metric_op = [sub_metric_op]

            test_ops.append(sub_metric_op + [self.cost_op[-1]])

        if len(test_ops) == 1:
            self.test_op = test_ops[0]
        else:
            self.test_op = [ tf.concat(metric, 0) for metric in zip(*test_ops) ]
            #for metric in zip(*test_ops):
            #    self.test_op.append(tf.concat(metric, 0))

    def test(self, tf_sess):
        results = tf_sess.run(self.test_op)
        return [results[-1]], results[:-1]

class Trainer:
    def __init__(self, learning_rate, *args, **kw_args):
        super(Trainer, self).__init__(*args, **kw_args)

        global_step = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate(global_step))

        grad_opts = dict(aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N) #colocate_gradients_with_ops=True, 
        sub_grads = []
        for sub_cost, sub_model in zip(self.cost_op, self.model):
            with sub_model.placement():
                sub_grads.append( self.optimizer.compute_gradients(sub_cost, **grad_opts) )

        if len(self.cost_op) == 1:
            grad = sub_grads[0]
        else:
            grad = Trainer.average_gradients(sub_grads)

        self.optim_op = self.optimizer.apply_gradients(grads, global_step=global_step)

    @staticmethod
    def average_gradients(tower_grads):
      average_grads = []
      for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
        return average_grads"""

class ResultSummary:
    def __init__(self, op):
        self.ary = []

    def append(self, result):
        result=result[0]
        self.ary.append(result.mean(axis=0)  if len(result.shape) > 0 else result)

    def get_summary(self, n):
        #return map(lambda x: sum(x[-n:])/n, self.arys)
        return sum(self.ary[-n:])/n

class OnlineAverage:
    def __init__(self, dtype):
        self.avg = tf.Variable(0, dtype=dtype, trainable=False)
        self.n_type = tf.int64
        self.n = tf.Variable(0, dtype=self.n_type, trainable=False)

    def update(self, vals):
        vshape = vals.get_shape().as_list()
        assert(len(vshape) == 1)
        new_n = self.n.assign_add(tf.constant(vshape[0], dtype=self.n_type))
        sum_v = tf.reduce_sum(vals)
        return self.avg.assign(self.avg + (sum_v - self.avg)/tf.cast(new_n, self.avg.dtype))

def outer(op, a, b):
    A = tf.reshape(a, [a.get_shape().as_list()[0], 1])
    B = tf.reshape(b, [1, b.get_shape().as_list()[0]])
    return op(A, B)
