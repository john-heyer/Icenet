import tables
import tqdm
import numpy as np

def write(fname, events, idxs):
	out = tables.open_file(fname, 'w')
	ary = out.create_table(out.root, "events", 
		description=events.description,
		filters=events.filters, 
		chunkshape=(1,), 
		expectedrows=len(events))

	for i in tqdm.tqdm(range(0, len(idxs), 100)):
		ary.append(events[idxs[i:i+100]])

	out.close()


if __name__ == '__main__':
	import sys
	in_filename, train_filename, test_filename = sys.argv[1:]

	in_file = tables.open_file(in_filename, 'r')
	nevents = len(in_file.root.events)

	idxs = np.random.permutation(nevents)
	idx_cut = int(nevents*0.8)

	write(train_filename, in_file.root.events, idxs[:idx_cut])
	write(test_filename, in_file.root.events, idxs[idx_cut:])
