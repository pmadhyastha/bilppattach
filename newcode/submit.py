def f(w, pp):
    if w[3] is pp:
        return list(w[i] for i in [1,2,4])

samples = [l.strip().split() for l in open('datasets/cleantrain.txt')]
sam = [lambda x:x if x[3] is 'for' for x in samples]


