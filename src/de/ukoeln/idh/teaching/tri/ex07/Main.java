package de.ukoeln.idh.teaching.tri.ex07;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.tokenizers.WordTokenizer;
import weka.core.tokenizers.Tokenizer;
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
		
		//Filter String to Word
		WordTokenizer wtokenizer = new WordTokenizer();
		wtokenizer.setDelimiters("\t \n\r<>?:,.{})([]");
		
		StringToWordVector stToWo = new StringToWordVector();
		stToWo.setAttributeIndices("1");
		stToWo.setLowerCaseTokens(true);
		stToWo.setTokenizer(wtokenizer);
		stToWo.setMinTermFreq(10);
		stToWo.setInputFormat(data);
		
		//Filter ausführen
		data = Filter.useFilter(data, stToWo);
		
		//Filter Numeric toNominal
		NumericToNominal nuToNo = new NumericToNominal();
		nuToNo.setAttributeIndices("1");
		nuToNo.setInputFormat(data);
		
		//ausführen NtN
		data = Filter.useFilter(data, nuToNo);
		
		// train classifier
		NaiveBayes classifier = new NaiveBayes();
		classifier.buildClassifier(data);
		
		//run eval
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(classifier, data, 10, new Random());
		System.out.println(eval.toSummaryString());

	}
}