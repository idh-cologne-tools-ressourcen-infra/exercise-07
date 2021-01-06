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
import weka.filters.unsupervised.attribute.NumericToNominal;


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
	
		//filter...
	}	
}