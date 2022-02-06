import datetime

from utility.helpers import get_random_vectors

if __name__ == '__main__':
    """
    Testing MVP implementation.
    """
    vectors = get_random_vectors(10000, 30, rank=10000, save=True)
    #
    # w = MVPAnnoyWrapper(30, 'hamming')
    # for v in vectors:
    #     w.add_vector_to_index(v)
    #
    # w.build_index()
    #
    # for i, v in enumerate(vectors):
    #     w.add_vector_to_pool(v)
    #     if i % 10 == 0:
    #         w.delete_vector(i)
    #
    # for i in range(10):
    #     t = datetime.datetime.now()
    #     print(w.search(vectors[i], 100))
    #     print(datetime.datetime.now() - t)
    w = MVPNMSLIBWrapper(None, 'cosinesimil')
    for v in vectors:
        w.add_vector(v)
    w.build_index()
    t = datetime.datetime.now()
    print(w.search(vectors[0], 10))
    print(datetime.datetime.now() - t)

    for i, v in enumerate(vectors):
        if i % 10 == 0:
            w.add_vector(v)
        else:
            w.delete_vector(i)

    t = datetime.datetime.now()
    print(w.search(vectors[0], 10))
    print(datetime.datetime.now() - t)
