import gctf
import tftables
import tensorflow as tf
import tqdm

def TauData(fnames, labels, batch_size):
    class_batch_size = batch_size//len(fnames)
    readers = [ tftables.TableReader(fname, class_batch_size) for fname in fnames ]

    with tf.device('/cpu:0'):
        batches, truths = [], []
        loaders = []
        weights = []
        biases = []
        for reader, label in zip(readers, labels):
            class_batch = reader.get_batch('/events', block_size=class_batch_size, n_procs=4, read_ahead=12)
            class_labels = tf.fill([class_batch_size], label)
            class_truth = tf.to_float(tf.one_hot(class_labels, len(labels), 1, 0))
            #class_batch = tf.slice(class_batch, [0, tf.cast(tf.argmax(tf.reduce_sum(class_batch, [2,3]), 1), tf.int32), 0, 0], [-1,1,-1,-1])
            #idxs = tf.transpose([tf.range(class_batch_size, dtype=tf.int32), tf.cast(tf.argmax(tf.reduce_sum(class_batch, [2,3]), 1), tf.int32)])
            #class_batch = tf.reshape(tf.gather_nd(class_batch, idxs), [class_batch_size, 1] + class_batch.get_shape().as_list()[2:])
            #print(class_batch.get_shape().as_list())
            class_waveforms = tf.cast(class_batch['waveforms'], tf.float16)

            class_one_weights, class_energy, class_nevents, class_nfiles = [tf.cast(class_batch[k], tf.float64) for k in ['one_weight', 'true_energy', 'NEvents', 'NFiles']]
            class_weights = tf.log(class_one_weights*tf.pow(class_energy/1E5, -2.13))/(class_nevents*class_nfiles)

            class_weight_averager = gctf.OnlineAverage(class_weights.dtype)
            class_weight_avg = class_weight_averager.update(class_weights)

            class_normed_weights = tf.cast(class_weights/class_weight_avg, tf.float32)

            class_loader = tftables.FIFOQueueLoader(reader, 10, [class_truth, class_waveforms, class_normed_weights], threads=4)
            loaders.append(class_loader)

            cpu_class_truth, cpu_class_batch, cpu_class_normed_weights = class_loader.dequeue()

            batches.append(cpu_class_batch)
            truths.append(cpu_class_truth)
            weights.append(cpu_class_normed_weights)
            biases.append(tf.constant((1.0 if label == 0 else 1.0), dtype=tf.float32, shape=[class_batch_size]))

        #cpu_batch = tf.transpose(tf.concat(0, batches), perm=[0,2,3,1])
        #print(batches)
        cpu_batch = tf.concat(batches, axis=0)
        cpu_truth = tf.concat(truths, axis=0)
        cpu_weight = tf.concat(weights, axis=0)
        bias = tf.concat(biases, axis=0)

        return cpu_truth, cpu_batch, cpu_weight, bias, loaders

def TauNet(args, batch, n_classes, train, reuse):
    
    import icenet
    regularizer = tf.contrib.layers.l2_regularizer(args.weight_decay)
    
    with tf.variable_scope('network', reuse=reuse):
        icenet_ = icenet.iceNetLL(train=train, regularizer=regularizer)
        # data mean ~70, std dev ~400
        pre = tf.cast(batch, tf.float32) #+ tf.abs(tf.truncated_normal(batch.get_shape(), mean=0, stddev=15, dtype=tf.float32))
        prediction = icenet_.network(pre, n_classes)
    
    with tf.variable_scope('network', reuse=True):
        icenet2_ = icenet.iceNetLL(train=train, regularizer=regularizer)

        mod = pre + tf.abs(tf.truncated_normal(batch.get_shape(), mean=0, stddev=40, dtype=tf.float32))
        mod_prediction = icenet2_.network(mod, n_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, -0.01*tf.reduce_sum(tf.nn.softmax(prediction)*tf.nn.log_softmax(mod_prediction)))
        #tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, -0.01*tf.reduce_sum(mod_prediction))

    return prediction

class iceTester(gctf.Tester):
    def __init__(self, net_args, fnames, labels, batch_size, *other_args, **kw_args):
        self.truth, self.batch, self.weight, self.bias, self.loaders = TauData(fnames, labels, batch_size)
        n_classes = self.truth.get_shape().as_list()[1]

        batches = tf.split(self.batch, 1)
        predictions = []
        for i in range(1):
            with tf.device('/gpu:' + str(i)):
                predictions.append( TauNet(net_args, batches[i], n_classes, train=False, reuse=True, *other_args, **kw_args) )
        self.prediction = tf.concat(predictions, 0)

        #self.prediction = TauNet(net_args, self.batch, n_classes, train=False, reuse=True, *other_args, **kw_args)

        with tf.device('/cpu:0'):
            gctf.Tester.__init__(self)

    def cost(self):
        return tf.reduce_mean(self.bias*self.weight*tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.truth))

    def metric(self):
        return tf.reduce_sum(self.weight*tf.cast(tf.equal(tf.argmax(self.truth, 1), tf.argmax(self.prediction, 1)), dtype=tf.float32))/tf.reduce_sum(self.weight)

class iceTrainer(gctf.Trainer):
    def __init__(self, net_args, fnames, labels, batch_size, *other_args, **kw_args):
        self.truth, self.batch, self.weight, self.bias, self.loaders = TauData(fnames, labels, batch_size)
        n_classes = self.truth.get_shape().as_list()[1]

        batches = tf.split(self.batch, 1)
        predictions = []
        reuse = False
        for i in range(1):
            with tf.device('/gpu:' + str(i)):
            #with tf.device('/gpu:0'):
                predictions.append( TauNet(net_args, batches[i], n_classes, train=False, reuse=reuse, *other_args, **kw_args) )
                reuse = True
        self.prediction = tf.concat(predictions, 0)

        #self.prediction = TauNet(net_args, self.batch, n_classes, train=True, reuse=False, *other_args, **kw_args)

        learning_rate = lambda global_step: tf.train.exponential_decay(args.base_learning_rate, global_step,
                                           args.decay_period, args.decay_constant, staircase=True)
        gctf.Trainer.__init__(self, learning_rate)#, saver_filter = lambda v: v.name[:7] == "network")

    def cost(self):
        return tf.reduce_mean(self.bias*self.weight*tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.truth))

    def metric(self):
        return tf.reduce_sum(self.weight*tf.cast(tf.equal(tf.argmax(self.truth, 1), tf.argmax(self.prediction, 1)), dtype=tf.float32))/tf.reduce_sum(self.weight)

def main(args, save_folder):
    trainer = iceTrainer(args, args.train_fnames, args.train_labels, args.batch_size)
    train_metric_summary = gctf.ResultSummary(trainer.metric_op)
    train_cost_summary = gctf.ResultSummary(trainer.cost_op)

    tester = iceTester(args, args.test_fnames, args.test_labels, args.batch_size)
    test_metric_summary = gctf.ResultSummary(tester.metric_op)
    test_cost_summary = gctf.ResultSummary(tester.cost_op)

    iterations = args.iterations + args.iterations//args.test_period
    test_summary_period = args.summary_period//args.test_period

    with gctf.sess(device=[0]) as tf_sess:
        trainer.init(tf_sess)
        #tester.init(tf_sess)
        tf.get_default_graph().finalize()
        [ loader.start(tf_sess) for loader in trainer.loaders ]
        [ loader.start(tf_sess) for loader in tester.loaders ]

        for i in tqdm.tqdm(range(iterations), smoothing=0.1):
            if i % args.test_period != 0:
                train_cost, train_metric = trainer.train(tf_sess)
                train_metric_summary.append(train_metric)
                train_cost_summary.append(train_cost)
                #print(tf_sess.run(trainer.weight))

            else:
                test_cost, test_metric = tester.test(tf_sess)
                test_metric_summary.append(test_metric)
                test_cost_summary.append(test_cost)

            if i % args.summary_period == 0:
                tqdm.tqdm.write("{:8d}: [ Train -- Accuracy: {: >#016.4f}%, \tCost: {: >#016.4f} \t] [ Test -- Accuracy: {: >#016.4f}%, \tCost: {:6f} ]".format(
                    i,
                    100*train_metric_summary.get_summary(args.summary_period),
                    train_cost_summary.get_summary(args.summary_period),
                    100*test_metric_summary.get_summary(test_summary_period),
                    test_cost_summary.get_summary(test_summary_period)
                ))

            if i % args.save_period == 0 and i != 0 and save_folder is not None:
                trainer.save(tf_sess, save_folder + "/iter_" + str(i) + ".mdl")

class Args:
    """Defaults"""
    train_fnames = ["/home/gabrielc/raid0/icecube/train_cc_nutau.h5", "/home/gabrielc/raid0/icecube/train_nc_nutau.h5"]
    train_labels = [0, 1]
    train_relweight = [3, 1]
    test_fnames = ["/home/gabrielc/raid0/icecube/test_cc_nutau.h5", "/home/gabrielc/raid0/icecube/test_nc_nutau.h5"]
    test_labels = [0, 1]
    labelnames = ["CC", "NC"]

    batch_size = 16
    iterations = 1000000
    test_period = 5
    summary_period = 1000
    weight_decay = 0.0001
    base_learning_rate = 0.0005
    #momentum = 0.999
    decay_period = 40000
    decay_constant = 0.1
    save_period = 50000

if __name__ == '__main__':
    args = Args()

    save_folder = None
    import sys
    if len(sys.argv) > 1:
        save_folder = sys.argv[1]

    main(args, save_folder)