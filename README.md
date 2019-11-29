# Genetic-Programming-Predictive
Prediction of Children's alcohol behaviors using Python and Genetic Programming algorithms

## 1. Abstract
The aim of this project is to build a regressive
model that is able to predict, at a macro level,
the percentage of children with dangerous
alcohol behaviors in a country. In order to
achieve this, we applied several techniques
of data preprocessing, feature engineering,
Machine Learning models and, additionally,
the introduction of Genetic Programing within
the scope of the project. The focus was on
benchmarking different models with different
parameters and compare and access the
results. The best model should be a model
that will have a good generalization ability
and small variation of its predictions, allowing
us to make rather good conclusions and
predictions. Regarding the target variable,
we want to predict the alcohol frequency,
represented by the proxy feature alcopops.

<p align="center"> <img src="03_Images/01_workflow.PNG" width="700"/>

## 02. Explore
Having collected the data from three different sources and three different years, we had to do a lot of preprocessing to get all the data on the same
granularity. Our first approach in the exploration phase
was to overview the variables and their distribution, briefly
discussed on the Project 2 Description. After a brief
exploration, we could observe which countries presented
higher frequencies of alcohol consumption. In Figure 2.
we present the gender gap in the last collected year
(2014). It is also noticeable that boys in Portugal claim
they have the 15th highest alcohol frequency consumption
between all collected countries.

<p align="center"> <img src="03_Images/19_explore.PNG" width="700"/>

### 02.1 Missing Values
Before splitting the data, we need to understand
the quality of the data, namely, the missing
values in the dataset. As we can see from the
Figure 3., we have some 6 features whose
missing value percentage is higher than the
established 3 % threshold. One of them is our
target variables. For the observations with
missing values in the target variable we decided
to drop them, since they will bring no predictive
power to our models. For the remaining
features, below and above the threshold, since
we have a small dataset, we decided not to drop
any observation and fill the missing values.

<p align="center"> <img src="03_Images/02_missing.PNG" width="700"/>

### 02.2 Correlations
The next step in the exploration phase is to
see how features are correlated amongst
themselves. At first sight, in chart 3,we can
see that the most noticeable correlation
with the alcohol frequency is the year when
the data was collected and with the children
having his/her first drunk experiences at the
ages of 13 years old or younger, with
correlation values of -0.43 and 0.51,
accordingly. For the remaining independent
features, we can also observe a strong
correlation between the country GDP per
capita and the teenage pregnancy (-0.56),
exercise and teenage pregnancy (-0.52) and,
the highest dependency, the age of the first
drunk experience and if the child has been drunk or not (0.91).

<p align="center"> <img src="03_Images/03_corr.PNG" width="700"/>
  
## 03. Modify
For the following chapters and project framework, we had to split the data between Training and Testing.
As we have a small dataset, the percentage defined for testing was set at 10%. Furthermore, we decided
to perform this action with 5 different seeds, so in the end, we can get the generalization ability of the
models in 5 different testing sets and make better conclusions. Since the scope and objective of this
project is not to explore all the possibilities in terms of preprocessing of data, we did not experiment a
lot of different methods.
For all decisions, we performed all possibilities and benchmarked them, however, in each chapter, we will
present the boxplot of the results only depending on that specific decision, over five runs, i.e.: For
Imputing Missing values, all other decisions are turned off except for this option.

### 03.1 Imputing Missing Values
For imputation of Missing Values, we
compared the generalization ability of dealing
with missing values or just dropping them. It
is important to notice that if we drop all
observations with missing values in the
training set, we would get around half of the
dataset size (from around to 200 to 100
observations).

<p align="center"> <img src="03_Images/04_impute_missings.PNG" width="700"/>

### 03.2 Deal with Outliers
Given that we are handling with data at a
macro level, we do not expect to encounter a
high number of outliers, since all individual
observations will be smoothed by the
majority. Having a small of data, we don’t
want to remove observations classified as
outliers, instead, we want to smooth them,
hoping that this will improve the predictive
ability of the models. As outlier options, we
decided to ignore them or to deal with them,
by applying a smoothing technique that will,
for all features, establish the 5th and 95th
percentile and, for values smaller of higher
than these thresholds, smooth the data according to the closest of the selected percentiles. A Priori, we
expect that dealing with outliers, in this case, will not have a huge impact in the results.

<p align="center"> <img src="03_Images/05_smooth_outliers.PNG" width="700"/>
  
## 04. Feature Selection
Having 21 independent features, another
important point to make is whether we will
ignore the Curse of Dimensionality or reduce
the number of variables, since it can improve
the computational effort of the algorithms and,
by removing redundant features, avoid
multicollinearity issues. However, since the
process is a trade-off, we can expect to lose
some information. For this method, we
performed a Recursive Feature Analysis with a
Linear Regressor estimator.

<p align="center"> <img src="03_Images/06_variable_select.PNG" width="700"/>
 
## 05. Decomposition: PCA
Instead of performing feature selection or, on
top of this step, we can also decompose the
data into principal components, which are not
correlated amongst themselves. For deciding
the ideal number of components, since we
cannot find a way of looking at the elbow
graph automatically in all the runs, we are
setting the threshold on the component
whose cumulative variance withholds 80% of
the explained total variance.

<p align="center"> <img src="03_Images/07_feature_decomp.PNG" width="700"/>
  
## 06. Baseline Models Benchmark
For the baseline models we experimented the following models:
§ Random Forest Regressor
§ Gradient Boosting Regressor
§ Ada Boost Regressor
§ Bagging Regressor
We experiment all preprocess option combination, over 5 different runs and seeds, resulting in 56
different outputs. In each run, a grid search with 5 cross validations was performed in order to find the
best hyperparameters and use the best estimator of each model. In table 1, the best and worst five
models, according to the Negative Mean Absolute Error, are shown.

<p align="center"> <img src="03_Images/08_ensemble_bench.PNG" width="700"/>

From Table 1., we can see that, amongst the baseline models (tuned in grid search), Gradient Boost
appears to have better results, as well as a rather small variation in its average results. To make a further
conclusion about the best decisions to make in preprocessing, we decided to evaluate the generalization
ability of the four models grouped by the decisions themselves. For this purpose, we created an id that
represents each model according to the choices that were made:
Figure 8. PCA Transformation Benchmark
A Priori assumptions
We expect the best generalization ability from the models with missing imputation and outlier smoothing, as
well as variable selection.
5
1. Impute Missings: True or False
2. Smooth Outliers: True or False
3. Variable Selection: True or False
4. PCA: True or False
{(𝑇, 𝐹); (𝑇, 𝐹); (𝑇, 𝐹); (𝑇, 𝐹)}

So, for example, TTFF stands for imputing
Missings, dealing with outliers but not
performing variable selection and
decomposition.
From Figure 9., we can see that some
decisions achieve a higher generalization
ability than others. Overall, imputing
Missings appears to be a good call. On the
other hand, decomposition and variable
selection appear to lower the generalization
ability of the models, unlike our a priori assumptions the most difficult decision to make was whether
outliers should be smoothed or not. Since the difference between this choice is not so relevant, we
decided to play it safe and perform outlier smoothing, making our decision representation as {𝑇, 𝑇, 𝐹, 𝐹}.

### 06.1 Artificial Data
Having a small amount of data, we
wondered if producing artificial data would
improve the generalization ability of our
models. Having in consideration that we are
dealing with a regression problem, the usual
packages to generate data such as SMOTE
did not work, since they are oriented
towards a classification problem. Following
the same logic, we decided to try our own
technique to generate data. We start by
generating clusters using the K-Means
algorithm and, using the centroids, we
calculate the distance between the cluster
center and each point in that group and then
generate a random point between these two
coordinates. We tried different number of clusters
and proportion of data to generate, presented in
Table 2. Based on the results, it appears that it is
not worth it, in terms of computational effort,
generate artificial data, since the payoff from the
generalization ability does not change a lot.
The conclusions obtained in this benchmark were
applied to the preprocessing of the data imputted
in the genetic programming framework.

<p align="center"> <img src="03_Images/09_artificial_data.PNG" width="700"/>
<p align="center"> <img src="03_Images/09_artificial_data2.PNG" width="700"/>


