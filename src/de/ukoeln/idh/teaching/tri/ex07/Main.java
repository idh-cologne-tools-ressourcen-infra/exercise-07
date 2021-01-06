package de.ukoeln.idh.teaching.tri.ex07;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.stemmers.NullStemmer;
import weka.core.tokenizers.WordTokenizer;

public class Main {
	public static void main(String[] args) throws IOException, Exception {
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File(new StringBuilder().append("data").append(File.separator).append("imdb.arff").toString()));
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		//one, two, don't know what to do..

        StringToWordVector filtern = new StringToWordVector(1000);
		WordTokenizer ownTokenizer = new WordTokenizer();
		ownTokenizer.setDelimiters(" \t\n.,;:'\"()?!+'");

        filtern.setLowerCaseTokens(true);
        filtern.setStemmer(new NullStemmer());
        filtern.setInputFormat(instances);
//        System.out.println(instances);

        Instances gefiltert = Filter.useFilter(instances, filtern);
//        System.out.println(gefiltert);

		NumericToNominal num2nom = new NumericToNominal();
		num2nom.setAttributeIndices("first");
		num2nom.setInputFormat(gefiltert);

		Instances nomgef = Filter.useFilter(gefiltert, num2nom);
//		System.out.println(nomgef);

		RandomForest classifier = new RandomForest();
		classifier.buildClassifier(nomgef);

		Evaluation eval = new Evaluation(nomgef);
		eval.crossValidateModel(classifier, nomgef, 10, new Random());

		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toSummaryString());
	}
}
