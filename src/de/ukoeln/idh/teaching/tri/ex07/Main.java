package de.ukoeln.idh.teaching.tri.ex07;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;import weka.core.stemmers.SnowballStemmer;
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
//		instances.setClassIndex(instances.numAttributes() - 1);
		instances.setClassIndex(1);
		
		StringToWordVector stwFilter = new StringToWordVector(1000); 
		stwFilter.setLowerCaseTokens(true);
		stwFilter.setStemmer(new SnowballStemmer());
		WordTokenizer tokenizer = new WordTokenizer();
		tokenizer.setDelimiters(" \r\t\n.,;:'\"()?!-><#$\\%&*+/@^_=[]{}|`~0123456789");
		stwFilter.setTokenizer(tokenizer);
		stwFilter.setInputFormat(instances);

		Instances filteredSet = Filter.useFilter(instances, stwFilter);
		
		NumericToNominal ntnFilter = new NumericToNominal();
		ntnFilter.setAttributeIndices("first");
		ntnFilter.setInputFormat(filteredSet);
		
		Instances filteredSet2 = Filter.useFilter(filteredSet, ntnFilter);
		
//		System.out.println(filteredSet2);
		
		// train classifier
//		OneR classifier = new OneR();
//		classifier.buildClassifier(filteredSet2);
		
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(filteredSet2);

		Evaluation eval = new Evaluation(filteredSet2);
		eval.crossValidateModel(classifier, filteredSet2, 10, new Random());
		
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toSummaryString());
		
		
	}
}