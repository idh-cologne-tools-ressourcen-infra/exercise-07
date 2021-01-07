package de.ukoeln.idh.teaching.tri.ex07;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Main {
	public static void main(String[] args) throws IOException, Exception {

		// load data
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("data/imdb.arff"));
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		// apply filter 1: Convert strings to word vectors
		WordTokenizer wtokenizer = new WordTokenizer();
		wtokenizer.setDelimiters("\t \n\r<>?:,.{})([]");

		StringToWordVector filter = new StringToWordVector();
		filter.setAttributeIndices("1");
		filter.setLowerCaseTokens(true);
		filter.setTokenizer(wtokenizer);
		filter.setMinTermFreq(10);
		filter.setInputFormat(instances);

		// run filter
		instances = Filter.useFilter(instances, filter);

		// apply filter 2: Convert target attribute from numeric to nominal
		NumericToNominal n2n = new NumericToNominal();
		n2n.setAttributeIndices("1");
		n2n.setInputFormat(instances);

		// run filter
		instances = Filter.useFilter(instances, n2n);

		// train classifier
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(instances);

		// Run evaluation
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random());
		System.out.println(eval.toSummaryString());
	}
}