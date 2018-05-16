import tensorflow as tf
import train
import gctf
import tftables
import numpy as np

def TauData(fname, label, n_labels, batch_size):
    class_batch_size = batch_size#//len(fnames)
    reader = tftables.TableReader(fname, class_batch_size)

    with tf.device('/cpu:0'):

        class_batch = reader.get_batch('/events', block_size=class_batch_size, n_procs=4, read_ahead=12, cyclic=False)
        class_labels = tf.fill([class_batch_size], label)
        class_truth = tf.to_float(tf.one_hot(class_labels, n_labels, 1, 0))
        #class_batch = tf.slice(class_batch, [0, tf.cast(tf.argmax(tf.reduce_sum(class_batch, [2,3]), 1), tf.int32), 0, 0], [-1,1,-1,-1])
        #idxs = tf.transpose([tf.range(class_batch_size, dtype=tf.int32), tf.cast(tf.argmax(tf.reduce_sum(class_batch, [2,3]), 1), tf.int32)])
        #class_batch = tf.reshape(tf.gather_nd(class_batch, idxs), [class_batch_size, 1] + class_batch.get_shape().as_list()[2:])
        #print(class_batch.get_shape().as_list())
        class_waveforms = tf.cast(class_batch['waveforms'], tf.float16)

        class_one_weights, class_energy, class_nevents, class_nfiles = [tf.cast(class_batch[k], tf.float64) for k in ['one_weight', 'true_energy', 'NEvents', 'NFiles']]
        #class_weights = tf.log(class_one_weights*tf.pow(class_energy, -2.5)/(class_nevents*class_nfiles)) ||||| tf.pow(class_energy, -2)*(tf.pow(class_energy/1e5, -0.5))*
        class_weights = (3600*24*365)*(0.9e-8)*tf.pow(class_energy, -2)*(tf.pow(class_energy/1e5, -0.13))*class_one_weights/(class_nevents*class_nfiles)
        #print(class_energy.dtype)

        #class_weight_averager = gctf.OnlineAverage(class_weights.dtype)
        #class_weight_avg = class_weight_averager.update(class_weights)

        #class_normed_weights = tf.cast(class_weights/class_weight_avg, tf.float32)

        class_loader = tftables.FIFOQueueLoader(reader, 10, [class_truth, class_waveforms, class_weights], threads=4)

        cpu_class_truth, cpu_class_batch, cpu_class_weights = class_loader.dequeue()

        return cpu_class_truth, cpu_class_batch, cpu_class_weights, class_loader

class iceClassifier(gctf.Tester):
    def __init__(self, net_args, fname, label, n_labels, batch_size, *other_args, **kw_args):

        self.truth, self.batch, self.weight, self.loader = TauData(fname, label, n_labels, batch_size)
        n_classes = self.truth.get_shape().as_list()[1]
        self.prediction = train.TauNet(net_args, self.batch, n_classes, train=False, *other_args, **kw_args)

        gctf.Tester.__init__(self)

    def cost(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.truth))

    def metric(self):
        return [self.truth, tf.nn.softmax(self.prediction), self.weight]

def classify(args, model_file, fname, label, n_labels, batch_size, reuse):
    classifier = iceClassifier(args, fname, label, n_labels, args.batch_size, reuse=reuse)

    iterations = args.iterations + args.iterations//args.test_period
    test_summary_period = args.summary_period//args.test_period

    truths, predictions, weights = [], [], []
    with gctf.sess(device=[0]) as tf_sess:
        classifier.init(tf_sess)
        tf.get_default_graph().finalize()
        classifier.load(tf_sess, model_file)

        with classifier.loader.begin(tf_sess):

            while True:
                cost, metric = classifier.test(tf_sess)
                truth, prediction, weight = metric
                truths.append(truth)
                predictions.append(prediction)
                weights.append(weight)

    return np.concatenate(truths, axis=0), np.concatenate(predictions, axis=0), np.concatenate(weights, axis=0)

def main(args, model, out):
    import pickle
    n_labels = len(args.test_labels)

    all_truths, all_predictions, all_weights = [], [], []
    reuse = False
    for fname, label in zip(args.test_fnames, args.test_labels):
        with tf.Graph().as_default():
            truths, predictions, weights = classify(args, model, fname, label, n_labels, args.batch_size, reuse=reuse)
        #print((truths.shape, predictions.shape))
        all_truths.append(truths)
        all_predictions.append(predictions)
        all_weights.append(weights)
        #reuse = True

    all_truths, all_predictions, all_weights = np.concatenate(all_truths, axis=0), np.concatenate(all_predictions, axis=0), np.concatenate(all_weights, axis=0)

    with open(out, 'wb') as outfile:
        pickle.dump(dict(
            truths=all_truths,
            predictions=all_predictions,
            weights=all_weights,
            label_names=args.labelnames
            ), outfile)

if __name__ == '__main__':
    args = train.Args()

    import sys
    model_file, predict_filename = sys.argv[1:3]

    main(args, model_file, predict_filename)
    