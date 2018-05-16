import gctf
import tensorflow as tf
import numpy as np

def load_dom_locs(filename):
    data = np.loadtxt(filename)
    dom_xyz = np.zeros((86, 64, 3))
    for dom_data in data:
        string, dom_no, x, y, z = dom_data
        dom_xyz[int(string)-1][int(dom_no)-1] = x, y, z
    return dom_xyz

def get_avg_string_locs(filename):
    dom_xyz = load_dom_locs(filename)
    return dom_xyz[:, :60, :2].mean(axis=1)



class iceNet(gctf.WideResNet):
    def __init__(self, train, *args, **kw_args):
        self.dropout_p = 0.7 if train else None
        gctf.WideResNet.__init__(self, *args, **kw_args)

    def to_strings(self, data_shape, layer):
        shape = layer.get_shape().as_list()
        return tf.reshape(layer, shape=[data_shape[0], data_shape[1]] + shape[1:])

    def main(self, data):
        data_shape = data.get_shape().as_list()
        with tf.variable_scope('main'):
            data = tf.reshape(data, shape=[data_shape[0]*data_shape[1], data_shape[2], data_shape[3], 1])
            layers = [data]
            layer_plan = [(10, (1,2)), (10, (2,2)), (10, (1,1)), (15, (2,2)), (15, (1,1)), (20, (2,2)), (20, (1,1)), (25, (2,2)), (30, (2,2))]
            for nout, stride in layer_plan:
                res_mod = self.res_module(layers[-1], nout=nout, ksize=(max(3,stride[0]*2-1), max(3,stride[1]*2-1)), 
                                          stride=stride, name='res_module_'+str(len(layers)), dropout_p=self.dropout_p)
                layers.append(res_mod)
            res_out = layers[-1]

            pool_layer = self.pool(res_out, ksize=(2,2), stride=(2,2), op='max')

            reduction_layer = self.to_strings(data_shape, self.fc(pool_layer, 8))
        return reduction_layer

    def network(self, data, n_classes):
        with tf.variable_scope('iceNet'):
            result = self.main(data)

            #fc0 = self.fc_elu(result, 32)
            #fc = self.fc(fc0, n_classes, name="score")
            fc = self.fc(result, n_classes)
        return fc

class iceNetLL(gctf.AbsLLNet):
    def __init__(self, train, *args, **kw_args):
        super(iceNetLL, self).__init__(*args, **kw_args)
        self.dropout_p = 0.7 if train else None

    # def input_xform(self, input_batch, x_size, y_size, out_ch, name="xform"):
    #     grid_xs, grid_ys = np.linspace(-500.0, 500.0, x_size), np.linspace(-500.0, 500.0, y_size)
    #     grid_points = np.rollaxis(np.array(np.meshgrid(grid_xs, grid_ys)), 0, 3)

    #     n_strings = 86
    #     r0 = 300 # was 180
    #     order_n, order_v = 12, 12 # was 6

    #     input_shape = input_batch.get_shape().as_list()
    #     assert(len(input_shape) == 3)
    #     in_ch = input_shape[-1]
        
    #     dom_xys = get_avg_string_locs("DOMLocation.txt")
    #     #print(dom_xys)
    #     dom_xs, dom_ys = dom_xys.T
    #     import scipy
    #     hull = scipy.spatial.Delaunay(dom_xys)
    #     grid_simplexes = hull.find_simplex(grid_points)
    #     inside_hull = grid_simplexes >= 0
    #     #print(inside_hull)

    #     delta_xs = grid_xs[:,None,None] - dom_xs[None,None,:]
    #     delta_ys = grid_ys[None,:,None] - dom_ys[None,None,:]
    #     rs = np.sqrt( (delta_xs)**2 + (delta_ys)**2 )
    #     thetas = np.arctan2(delta_xs, delta_ys)
    #     #print((rs[0,0], np.sqrt(((grid_points[:,:,None,:]-np.array([dom_xs, dom_ys]).T[None,None,:,:])**2).sum(axis=3))[0,0]))

    #     R = lambda n, alpha, r: scipy.special.jn(n, alpha*r/r0) * ( np.logical_and(r < r0, inside_hull[...,None]).astype(int) )
    #     j_zeros = [ scipy.special.jn_zeros(n, order_v) for n in range(order_n) ]
    #     basis_A = np.array([ [ R(n, alpha, rs) * np.sin(n*thetas) for alpha in j_zeros[n] ] for n in range(order_n) ])
    #     basis_B = np.array([ [ R(n, alpha, rs) * np.cos(n*thetas) for alpha in j_zeros[n] ] for n in range(order_n) ])
        
    #     with tf.variable_scope(name):
    #         tf_C = self.init_var([1], "C_offset")
    #         tf_As = self.init_var([order_n, order_v, in_ch, out_ch], "A_coefficients")
    #         tf_Bs = self.init_var([order_n, order_v, in_ch, out_ch], "B_coefficients")
    #         tf_basis_A = tf.constant(basis_A, dtype=tf.float32)
    #         tf_basis_B = tf.constant(basis_B, dtype=tf.float32)

    #     # n: order_n, v: order_v, i: in_ch, o: out_ch, x: dom_x, y: dom_y, d: dom/string_no, b: batch
    #     xform = tf.einsum('nvio,nvxyd->xyodi', tf_As, tf_basis_A) + tf.einsum('nvio,nvxyd->xyodi', tf_Bs, tf_basis_B) + tf_C
    #     #result = tf.einsum('nk,abck->abcn', tf.reshape(xform, shape=[x_size*y_size, n_strings]), input_batch)
    #     result = tf.einsum('xyodi,bdi->bxyo', xform, input_batch)
    #     return result

    def to_strings(self, data_shape, layer):
        shape = layer.get_shape().as_list()
        return tf.reshape(layer, shape=[data_shape[0], data_shape[1]] + shape[1:])

    def main(self, data):
        data_shape = data.get_shape().as_list()
        with tf.variable_scope('main'):
            data = tf.reshape(data, shape=[data_shape[0]*data_shape[1], data_shape[2], data_shape[3], 1])
            layers = [self.conv(data, nout=8, ksize=(3,5), stride=(1,2), name="input_conv")]
            layer_plan = [
                #(8, (5,5), (4,4)),
                (8, (3,3), (2,2)),
                (8, (3,3), (1,1)),
                (8, (3,3), (2,2)),
                (8, (3,3), (1,1)),
                #(15, (5,5), (2,2)), 
                #(8, (5,5), (4,4)),
                (8, (3,3), (2,2)),
                (8, (3,3), (1,1)), 
                (8, (3,3), (2,2)),
                (8, (3,3), (1,1)),
                #(25, (5,5), (2,2)), 
                (8, (3,3), (2,2)),
                (8, (3,3), (1,1)),
                (8, (3,3), (1,1))
            ]
            for nout, ksize, stride in layer_plan:
                next_layer = self.relu_conv_ll(layers[-1], nout=nout, ksize=ksize, #(max(3,stride[0]*2-1), max(3,stride[1]*2-1)) 
                                          stride=stride, name='ll_layer_'+str(len(layers)))
                layers.append(next_layer)
            res_out = layers[-1]

            pool_layer = self.pool(res_out, ksize=(2,2), stride=(2,2), op='max')

            #fc_layer0 = self.relu_fc_ll(pool_layer, 8, name='fc0')
            #fc_layer1 = self.relu_fc_ll(fc_layer0, 16, name='fc1')

            reduction_layer = self.to_strings(data_shape, pool_layer)

        return reduction_layer

    def main2(self, data):

        with tf.variable_scope("main2"):
            #pointwise_conv = input_xform(tf.squeeze(data, axis=[2,3]), x_size=16, y_size=16, out_ch=8)
            pointwise_conv = self.looks_linear(tf.squeeze(data, axis=[2,3]), lambda b: self.input_xform(self.relu(b), x_size=16, y_size=16, out_ch=8, name="pwise_conv"))

            layers = [pointwise_conv]
            layer_plan = [
                (8, (3,3), (2,2)),
                (8, (3,3), (1,1)),
                (8, (3,3), (2,2)),
                (8, (3,3), (1,1)),
                (8, (3,3), (2,2)),
                (8, (3,3), (1,1))
            ]
            for nout, ksize, stride in layer_plan:
                next_layer = self.relu_conv_ll(layers[-1], nout=nout, ksize=ksize, #(max(3,stride[0]*2-1), max(3,stride[1]*2-1)) 
                                          stride=stride, name='ll_layer_'+str(len(layers)))
                layers.append(next_layer)
            res_out = layers[-1]

            pool_layer2 = self.pool(res_out, ksize=(2,2), stride=(2,2), op='max')

        return pool_layer2

    def network(self, data, n_classes):
        with tf.variable_scope('iceNet'):
            result = self.main(data)
            result = self.main2(result)

            #fc0 = self.fc_elu(result, 32)
            #fc = self.fc(fc0, n_classes, name="score")
            #fc0 = self.relu_fc_ll(result, 32, name='fc0')
            fc = self.relu_fc_ll(result, n_classes, name='score')
        return fc

