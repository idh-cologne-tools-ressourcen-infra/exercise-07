package weka.exercise.imdb;

import java.io.File;
import java.util.Random;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.core.tokenizers.WordTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Application {

	public static void main(String[] args) throws Exception {
		// load data
		ArffLoader loader = new ArffLoader();
		loader.setFile(new File("src/weka/exercise/imdb/imdb.arff"));
		
		Instances instances = loader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);
		
		StringToWordVector stringToWordVector = new StringToWordVector(1000);
		stringToWordVector.setLowerCaseTokens(true);
		stringToWordVector.setStemmer(new SnowballStemmer());
		
		/* Source:
		 * stackoverflow.com/questions/15851780/delimiters-option-for-weka-wordtokenizer
		 * "shankshera"-Answer
		 */
	
		WordTokenizer wordTokenizer = new WordTokenizer();
		String delimiters = " \r\t\n.,;:\'\"()?!-><#$\\%&*+/@^_=[]{}|`~0123456789";
		wordTokenizer.setDelimiters(delimiters);
		stringToWordVector.setTokenizer(wordTokenizer);
		
		stringToWordVector.setInputFormat(instances);
		Instances filteredInstance = Filter.useFilter(instances, stringToWordVector);
		
		NumericToNominal numericToNominal = new NumericToNominal();
		numericToNominal.setAttributeIndices("first");
		numericToNominal.setInputFormat(filteredInstance);
		
		Instances filteredInstance2 = Filter.useFilter(filteredInstance, numericToNominal);

		// train classifier
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(filteredInstance2);

		Evaluation eval = new Evaluation(filteredInstance2);
		eval.crossValidateModel(classifier, filteredInstance2, 10, new Random());
		
		System.out.println(eval.toClassDetailsString());
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());

	}

}
