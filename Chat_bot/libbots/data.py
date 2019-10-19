import collections, os ,sys, logging, itertools, pickle
from . import cornell

UNKNOWN_TOKEN = '#UNK'
BEGIN_TOKEN = "#BEG"
END_TOKEN = "#END"
MAX_TOKENS = 20
MIN_TOKEN_FEQ = 10
SHUFFLE_SEED = 5871

EMB_DICT_NAME = "emb_dict.dat"
EMB_NAME = "emb.npy"

log = logging.getLogger("data")

def save_emb_dict(dir_name, emb_dict):
	with open(os.path.join(dir_name,EMB_DICT_NAME), "wb") as fd:
		pickle.dump(emb_dict, fd)

def load_emb_dict(dir_name):
	with open(os.path.join(dir_name, EMB_DICT_NAME),"rb") as fd:
		return pickle.load(fd)

def encode_words(words, emb_dict):
	res =[emb_dict[BEGIN_TOKEN]]
	unk_idx = emb_dict[UNKNOWN_TOKEN]
	for w in words:
		idx = emb_dict.get(w.lower(), unk_idx)
		res.append(idx)
	res.append(emb_dict[END_TOKEN])
	return res

def encode_phrase_pairs(phrase_pairs, emb_dict, filter_unknows=True):
	unk_token = emb_dict[UNKNOWN_TOKEN]
	result = []
	for p1, p2 in phrase_pairs:
		p = encode_words(p1, emb_dict), encode_words(p2, emb_dict)
		if unk_token in p[0] or unk_token in p[1]:
			continue:
		result.append(p)
	return result

def group_train_data(training_data):
	groups = collections.defaultdict(list)
	for p1, p2 in training_data:
		l = groups[tuple(p1)]
		l.append(p2)
	return list(groups.items())#######Check it

def iterate_batches(data, batch_size):
	assert isinstance(data,list)
	assert isinstance(batch_size, int)

	ofs = 0
	while True:
		batch = data[ofs*batch_size:(ofs+1)*batch_size]
		if len(batch) <= 1:
			break
		yield batch
		ofs += 1

def load_data(genre_filter, max_tokens=MAX_TOKENS, min_token_freq=MIN_TOKEN_FEQ):
	dialouges = cornell.load_dialogues(genre_filter=genre_filter)
	if not dialouges:
		loog.error("No dialouges found, exirt!!")
		sys.exit()
	log.info("Loaded %d dialouges with %d phrases, generating training pairs",sum(map(len,dialouges)))
	phrase_pairs = dialouges_to_pairs(dialouges, max_tokens=max_tokens)
	log.info("Counting frq of words...")
	word_counts = collections.Counter()
	for dial in dialouges:
		for p in dial:
			word_counts.update(p)
	freq_set = set(map(lambda p: p[0], filter(lambda p: p[1] >= min_token_freq, word_counts.items())))
	log.info("Data has %d uniq words, %d of them occur more than %d",
             len(word_counts), len(freq_set), min_token_freq)
	phrase_dict = phrase_pairs_dict(phrase_pairs, freq_set)
	return phrase_pairs, phrase_dict

def dialogues_to_pairs(dialouges, max_tokens=None):
	result = []
	for dial in dialouges:
		prev_phrase = None
		for phrase in dial:
			if prev_phrase is not None:
				if max_tokens is None or (len(prev_phrase) <= max_tokens and len(phrase) <= max_tokens):
					result.append(prev_phrase, phrase)
			prev_phrase = phrase
	return result

def decode_words(indices, rev_emb_dict):
	return [rev_emb_dict.get(idx, UNKNOWN_TOKEN) for idx in indices]

def trim_tokens_seq(tokens, end_token):
	res = []
	for t in tokens:
		res.append(t)
		if t == end_token:
			break
	return res

def split_train_test(data, train_ratio=0.95):
	count = int(len(data)*train_ratio)
	return data[:count], data[count:]