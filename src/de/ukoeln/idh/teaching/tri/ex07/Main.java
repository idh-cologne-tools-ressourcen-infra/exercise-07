package com.vgavrilova;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.util.Random;

public class Main {

    public static void main(String[] args) throws Exception {
        // load data
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File("data/imdb.arff"));
        Instances instances = loader.getDataSet();
        instances.setClassIndex(instances.numAttributes() - 1);

        // filter data:
        // StringToVector and NumericToNominal
        StringToWordVector filter = new StringToWordVector(1000);
        WordTokenizer customTokenizer = new WordTokenizer();
        customTokenizer.setDelimiters(" \t\n.,;:'\"()?!+");

        filter.setLowerCaseTokens(true);
        filter.IDFTransformTipText();
        filter.setStopwordsHandler(new Rainbow());
        filter.setStemmer(new SnowballStemmer());
        filter.setTokenizer(customTokenizer);
        filter.setInputFormat(instances);

        Instances vectorInstances = Filter.useFilter(instances, filter);

        // numeric to nominal data
        NumericToNominal ntn = new NumericToNominal();
        ntn.setAttributeIndices("first");
        ntn.setInputFormat(vectorInstances);

        Instances nominalInstances = Filter.useFilter(vectorInstances, ntn);

        // train classifier
        NaiveBayes classifier = new NaiveBayes();
        classifier.buildClassifier(nominalInstances);

        // evaluation
        Evaluation eval = new Evaluation(nominalInstances);
        eval.crossValidateModel(classifier, nominalInstances, 10, new Random());

        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toSummaryString());
    }
}
