package de.ukoeln.idh.teaching.tri.ex07;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.unsupervised.attribute.NumericalToNominal;

public class Main {
	public static void main(String[] args) throws IOException, Exception {

		// load data
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("data.arff"));
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		// train classifier
		RandomForest classifier = new RandomForest();
		classifier.buildClassifier(instances);

		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random());
		
		//filters
		/*
		 * https://stackoverflow.com/questions/15851780/delimiters-option-for-weka-wordtokenizer
		 */
		StringToWordVector stwvfilter = new StringToWordVector(1000);
			
		WordTokenizer tokenizer = new WordTokenizer();
        String delimiters = " \r\t\n.,;:\'\"()?!-><#$\\%&*+/@^_=[]{}|`~0123456789";
        tokenizer.setDelimiters(delimiters);
        stwvfilter.setTokenizer(tokenizer);
        stwvfilter.setInputFormat(data);
        
        
        /*
         * https://stackoverflow.com/questions/20014412/weka-only-changing-numeric-to-nominal
         */
		NumericalToNominal ntnfilter = new NumericalToNominal();
		String[] options = new String[2];
		options[0]="-R";
        options[1]="1-2";

        convert.setOptions(options);
        convert.setInputFormat(originalTrain);

        Instances newData=Filter.useFilter(originalTrain, convert);
		
	}
}