package de.ukoeln.idh.teaching.tri.ex07;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.classifiers.bayes.NaiveBayes;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.tokenizers.WordTokenizer;

public class Main {
	public static void main(String[] args) throws IOException, Exception {
		// load data
		StringToWordVector stringToWordVector = new StringToWordVector(1000);
		stringToWordVector.setLowerCaseTokens(true);
		
		stringToWordVector.setStemmer(new SnowballStemmer());
		
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("data/imdb.arff"));
		
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);
		
		/* Source:
		 * stackoverflow.com/questions/15851780/delimiters-option-for-weka-wordtokenizer
		 * "shankshera"-Answer
		 */
	
		WordTokenizer wordTokenizer = new WordTokenizer();
		String delimiters = " \r\t\n.,;:\'\"()?!-><#$\\%&*+/@^_=[]{}|`~0123456789";
		wordTokenizer.setDelimiters(delimiters);
		stringToWordVector.setTokenizer(wordTokenizer);
		
		stringToWordVector.setInputFormat(instances);
		Instances filtered = Filter.useFilter(instances, stringToWordVector);
		
		NumericToNominal numericToNominal = new NumericToNominal();
		numericToNominal.setAttributeIndices("first");
		numericToNominal.setInputFormat(filtered);
		
		filtered = Filter.useFilter(filtered, numericToNominal);

		// train classifier
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(filtered);

		Evaluation eval = new Evaluation(filtered);
		eval.crossValidateModel(classifier, filtered, 10, new Random());
		
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
	}
}