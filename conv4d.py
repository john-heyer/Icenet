    def conv_4d(self, bot, ksize_3d, ksize_4, nout, axis, stride_3d=(1,1,1), stride_4d=1, name='conv_4d'):
        assert ksize_4%2 == 1, "ksize_4 must be odd"

        with tf.variable_scope(name):
            
            in_channels = bot.get_shape().as_list()[-1]
            kernels = [self.init_var( ksize_3d + (in_channels, nout), 'kernel' + str(i), regularize=True) for i in range(ksize_4)]

            # size of bot is [nbatch, x, y, z, t, nchannel]
            bot_shape = bot.get_shape().as_list()
            rank = len(bot_shape)
            axis_size = bot_shape[axis]
            
            shift = ksize_4 // 2
            shifted_data = self.shifted_data([tf.identity(bot) for i in range(len(kernels))], rank, shift, axis, axis_size)
            print("shifted data size: ", len(shifted_data), shifted_data[0].get_shape().as_list())
            # shifted data now has size [3 * [n,x,y,z,t]] and shift is taken care of

            # if axis = t, will now be [3 *[t * [nbatch, x, y, z, nchannel]]
            layers_3d = [[tf.squeeze(l, axis=[axis]) for l in tf.split(bot, axis_size, axis=axis)] for bot in shifted_data]

            if stride_4d > 1:
                layers_3d = [ n_minus_1_layers[::stride_4d] for n_minus_1_layers in layers_3d]

            conv_3d_layers = [
                [        
                    tf.nn.conv3d(
                        layers_3d[k][l], kernels[k], strides=self.one_pad(stride_3d), padding='SAME', data_format='NDHWC'
                    )

                    for l in range(len(layers_3d[0]))
                ]
                for k in range(len(kernels))
            ]

            # conv_3d_layers shape: [3 * [t * [nbatch, x, y, z, nchannel] ] ]
            conv_3d_layers = [[tf.expand_dims(l3d, axis=axis) for l3d in k] for k in conv_3d_layers]
            # inserted squeezed dimension, shape now: [3 * [t * [nbatch, x, y, z, 1, nchannel] ] ]
            reshape_4d = [tf.concat([l3d for l3d in k], axis=axis) for k in conv_3d_layers]
            #reshape_4d shape should now be: [3 * [nbatch, x, y, z, t, nchannel]]

            # -1 in our case of ksize_4 = 3
            # shift =  ksize_4 // 2
            # shifted_data = self.shifted_data(reshape_4d, rank, shift, axis, axis_size)
            
            bias = self.init_var((nout,), 'bias')
            conv4_d = tf.add_n(reshape_4d)

            return tf.nn.bias_add(conv4_d, bias)