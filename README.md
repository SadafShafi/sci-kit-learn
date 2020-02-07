# sci-kit-learn
I compared a few algorithms on 2 different datasets and also applied CV, with hyperparameter tuning and without it.
I also pickled the objects, so we need to train models only once

Here is the Jist of results I got
test data =0.3



              Algorithm       Score_1        Score_2
0   logistic_regerssion  9.415205e-01       0.743781
1           GaussianNB:  9.239766e-01       0.619403
2        MultinomialNB:  8.654971e-01       0.664179
3         ComplementNB:  8.654971e-01       0.664179
4          BernoulliNB:  6.257310e-01       0.649254
5                  KNN:  8.947368e-01       0.624378
6              K means: -2.065823e+07 -335709.068480
7                  SVM:  8.771930e-01       0.733831
8       Random Forest:   9.298246e-01       0.733831
9         SGDClassifier  8.830409e-01       0.626866
10        LogisticR CV:  9.298246e-01       0.743781
11           ada boast:  9.356725e-01       0.679104



after
test data = 0.5
              Algorithm       Score_1        Score_2
0   logistic_regerssion  9.415205e-01       0.700000
1           GaussianNB:  9.239766e-01       0.608955
2        MultinomialNB:  8.654971e-01       0.646269
3         ComplementNB:  8.654971e-01       0.652239
4          BernoulliNB:  6.257310e-01       0.629851
5                  KNN:  8.947368e-01       0.641791
6              K means: -2.065823e+07 -553812.822919
7                  SVM:  8.771930e-01       0.701493
8       Random Forest:   9.298246e-01       0.698507
9         SGDClassifier  8.771930e-01       0.617910
10        LogisticR CV:  9.298246e-01       0.713433
11           ada boast:  9.356725e-01       0.676119

test data 0.1
              Algorithm       Score_1        Score_2
0   logistic_regerssion  9.415205e-01       0.723881
1           GaussianNB:  9.239766e-01       0.656716
2        MultinomialNB:  8.654971e-01       0.664179
3         ComplementNB:  8.654971e-01       0.694030
4          BernoulliNB:  6.257310e-01       0.604478
5                  KNN:  8.947368e-01       0.641791
6              K means: -2.065823e+07 -119658.131004
7                  SVM:  8.771930e-01       0.708955
8       Random Forest:   9.298246e-01       0.723881
9         SGDClassifier  8.538012e-01       0.589552
10        LogisticR CV:  9.298246e-01       0.761194
11           ada boast:  9.356725e-01       0.619403

               Algorithm       Score_1        Score_2
0    logistic_regerssion  9.415205e-01       0.743781
1            GaussianNB:  9.239766e-01       0.619403
2         MultinomialNB:  8.654971e-01       0.664179
3          ComplementNB:  8.654971e-01       0.664179
4           BernoulliNB:  6.257310e-01       0.649254
5                   KNN:  8.947368e-01       0.624378
6               K means: -2.065823e+07 -335709.068480
7                   SVM:  8.771930e-01       0.733831
8        Random Forest:   9.298246e-01       0.733831
9          SGDClassifier  8.596491e-01       0.629353
10         LogisticR CV:  9.298246e-01       0.743781
11            ada boast:  9.356725e-01       0.679104
12  Cross Validation LR:  9.415205e-01       0.736318
