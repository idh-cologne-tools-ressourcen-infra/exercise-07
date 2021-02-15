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
		
		// PREPROCESSING
		// Filter 1: String to Word
		// a) tokenizer
		WordTokenizer wtok = new WordTokenizer();
		wtok.setDelimiters("\t \n\r<>?:,.{})([]");
		// b) String to Word
		StringToWordVector s2w = new StringToWordVector();
		s2w.setAttributeIndices("1");
		s2w.setLowerCaseTokens(true);
		s2w.setTokenizer(wtok);
		s2w.setMinTermFreq(10);
		s2w.setInputFormat(instances);
		instances = Filter.useFilter(instances, s2w);

		// Filter 2: Numeric to Nominal
		NumericToNominal n2n = new NumericToNominal();
		n2n.setAttributeIndices("1");
		n2n.setInputFormat(instances);
		instances = Filter.useFilter(instances, n2n);

		// Train Classifier (with preprocessed data)
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(instances);
		
		// Evaluate
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random());
		System.out.println(eval.toSummaryString());
	}
}