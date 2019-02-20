def test_tree(S, indexer, locater, hparams, verbose=False) :
    passed = True
    for i in range(10) :
        missed_idxs = []
        tree = indexer(S, hparams)
        if verbose :
            print(tree.keys())
        for idx, q in enumerate(S) :
            cset = locater(tree, q)
            if idx not in cset :
                print('Index %i, C.set:%s' % (idx, str(cset)))
                missed_idxs.append(idx)
        if len(missed_idxs) > 0 :
            print('Missed %i/%i points' % (len(missed_idxs), len(S)))
            print(missed_idxs)
            passed = False
    if passed :
        print('Working as expected')
    else :
        print('Tree search contains error')
# -- end function
