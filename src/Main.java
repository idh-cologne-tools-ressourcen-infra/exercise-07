import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter; 
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.attribute.NumericToNominal;


public class Main {
	public static void main(String[] args) throws IOException, Exception {

		// load data
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("data/imdb.arff"));
		Instances instances = loader.getDataSet();
	 	instances.setClassIndex(instances.numAttributes() - 1);

		
		StringToWordVector stringToWordVector = new StringToWordVector(1000);
		stringToWordVector.setLowerCaseTokens(true);
		stringToWordVector.setStemmer(new SnowballStemmer());
		
		WordTokenizer wordTokenizer = new WordTokenizer();
		String delimiters = " \r\t\n.,;:\'\"()?!-><#$\\%&*+/@^_=[]{}|`~0123456789";
		wordTokenizer.setDelimiters(delimiters);
		stringToWordVector.setTokenizer(wordTokenizer);

		stringToWordVector.setInputFormat(instances);
		Instances filteredInstance = Filter.useFilter(instances, stringToWordVector);
		
		NumericToNominal numericToNominal = new NumericToNominal();
		numericToNominal.setAttributeIndices("first");
		numericToNominal.setInputFormat(filteredInstance);
		
		Instances filteredInstance1 = Filter.useFilter(filteredInstance, numericToNominal);
		
		// train classifier
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(filteredInstance1);

		Evaluation eval = new Evaluation(filteredInstance1);
		eval.crossValidateModel(classifier, filteredInstance1, 10, new Random());
		
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());

	}
}