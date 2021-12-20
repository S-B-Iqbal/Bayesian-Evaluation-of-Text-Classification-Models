
# TODO: Required packages


def paired_bootstrap_test(test_set, model1,model2, B, score,*args,**kwargs):
    """
    Function to generate \delta(x) and \delta(x^{(i)}) for B bootstrap samples.
    Reference: Jurafsky, Daniel, and James H. Martin. "Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition."
    Params:
    ------
    test_set: np.ndarray, Array of test outputs.
    model1: np.ndarray, Array of Model A's output.
    model2: np.ndarray, Array of Model B's output.
    B : int, No of Bootstrap's to be generated
    score: Evaluation algorithm.
    """
    N = test_set.shape[0]
    score1 =  score(test_set, model1, *args,**kwargs)
    score2 =  score(test_set, model2, *args,**kwargs)
    delta = score1-score2
    deltas = [] # for storing \delta(x) of bootstraps
    for boot in tqdm(range(B)):
        ind = np.random.randint(low=0, high=N, size=N)
        sampleY = test_set[ind,:]
        sample1 = model1[ind,:]
        sc1 = score(sampleY, sample1,*args,**kwargs)
        sample2 = model2[ind,:]
        sc2 = score(sampleY, sample2, *args, **kwargs)
        delta_b = sc1 - sc2
        deltas.append(delta_b)
    deltas = np.array(deltas)
    return (deltas, delta)

def hypothesis_test(dx_i,dx, significance=0.05):
    """
    Implementation of paired-bootstrap test.
    Reference: Berg-Kirkpatrick, et. al. An empirical investigation of statistical significance in nlp.
    """
    p_value = np.mean(dx_i>= (2*dx))
    if p_value<significance:
        print(f"We reject the null hypothesis at a significance of {significance}")
    else:
        print(f"We fail to reject the Null Hypothesis")
    return p_value
