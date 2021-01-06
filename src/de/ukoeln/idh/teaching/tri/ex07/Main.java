package de.ukoeln.idh.teaching.tri.ex07;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.*;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ConverterUtils.*;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Main {
	public static void main(String[] args) throws IOException, Exception {
		
		// load data
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("/exercise-07/data/imdb.arff"));
		Instances data = loader.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		data.setClassIndex(1);
		
		NumericToNominal nuToNo = new NumericToNominal();
		nuToNo.setInputFormat(data);
		Instances filtered = Filter.useFilter(data, nuToNo);
		System.out.println(filtered);
		
		StringToWordVector stToWo = new StringToWordVector(1000);
		stToWo.setInputFormat(filtered);
		Instances filtered2 = Filter.useFilter(filtered, stToWo);
		
		System.out.println(filtered2);
		
		System.out.println();
		// train classifier
		OneR classifier = new OneR();
		classifier.buildClassifier(data);

		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random());

	}
}