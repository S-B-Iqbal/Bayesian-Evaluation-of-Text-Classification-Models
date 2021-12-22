

# Params
jobs=-1
neighbors = 45
kernel='linear'
cache = 200
layers = (64,)

## Naive Bayes
nb_tfv = Pipeline(steps = [('tf_idf', TfidfVectorizer(analyzer='word', stop_words = stop_words)),('nb', MultinomialNB()) ]) 

# # KNN
knn_tfv = Pipeline(steps = [('tf_idf', TfidfVectorizer(analyzer='word', stop_words = stop_words)),('knn', KNeighborsClassifier(n_neighbors=neighbors, n_jobs=jobs))])

# # SVM
svm_tfv = Pipeline(steps = [('tf_idf', TfidfVectorizer(analyzer='word', stop_words = stop_words)),('svm', SVC(kernel = kernel, cache_size=cache))])

# ## NNet
# # Feature scaling
nn_tfv = Pipeline(steps = [('tf_idf', TfidfVectorizer(analyzer='word', stop_words = stop_words)),('nnet',MLPClassifier(hidden_layer_sizes=layers))])

pipelines = [nb_tfv,knn_tfv,svm_tfv,nn_tfv]
classifiers = [OneVsRestClassifier(pipeline, n_jobs=-1) for pipeline in pipelines]

# Train Classifiers
for classifier in classifiers:
    classifier.fit(xtrain,ytrain)
    
## Save Classifiers
dump(classifiers[0], 'nb.joblib')
dump(classifiers[1], 'knn.joblib')
dump(classifiers[2], 'svc.joblib')
dump(classifiers[3], 'nnet.joblib')

# Save Outputs
# 0 : Naive Bayes; 1: KNN; 2: SVM; 3: Neural Net
y_pred_knn = classifiers[1].predict(xtest)
np.save(f"knn.npy", y_pred_knn)

