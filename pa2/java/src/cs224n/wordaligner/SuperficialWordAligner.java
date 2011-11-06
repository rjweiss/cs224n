package cs224n.wordaligner;


import cs224n.langmodel.EmpiricalUnigramLanguageModel;
import cs224n.langmodel.LanguageModel;
import cs224n.util.Alignment;
import cs224n.util.CounterMap;
import cs224n.util.SentencePair;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SuperficialWordAligner implements WordAligner {
    private CounterMap<String, String> counterMap;
    private LanguageModel sourceModel;
    private LanguageModel targetModel;

    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();
        for (int i = 0; i < sentencePair.getFrenchWords().size(); i++ ) {
        //for (String frenchWord : sentencePair.getFrenchWords()) {
            String frenchWord = sentencePair.getFrenchWords().get(i);
            double frenchProb = sourceModel.getWordProbability(sentencePair.getFrenchWords(), i);
            double maxPointwiseMutualInformation = 0;
            int maxIndex = -1;
            List<Integer> maxIndices = new ArrayList<Integer>();
            for (int j = 0; j < sentencePair.getEnglishWords().size(); j++) {
            //for (String englishWord : sentencePair.getEnglishWords()) {
                String englishWord = sentencePair.getEnglishWords().get(j);
                double englishProb = targetModel.getWordProbability(sentencePair.getEnglishWords(), j);
                double jointCount = counterMap.getCount(frenchWord, englishWord);
                double jointProb = jointCount / counterMap.totalCount();
                double pointwiseMutualInformation = jointProb / (frenchProb * englishProb); // needs to return log if pmi
//                System.out.printf("%8.3f  ", pointwiseMutualInformation);
                if (pointwiseMutualInformation > maxPointwiseMutualInformation) {
                    maxPointwiseMutualInformation = pointwiseMutualInformation;
                    maxIndex = j;
                    maxIndices.clear();

                }
                if (pointwiseMutualInformation == maxPointwiseMutualInformation) {
                    maxIndices.add(j);
                }
            }
  //          System.out.println();

            if (maxIndices.size() > 1) {
                for (int index : maxIndices) {
                    alignment.addAlignment(index, i, false);
                }
            }
            else {
                alignment.addAlignment(maxIndex, i, true);
            }
        }
        return alignment;
    }


    public double getAlignmentProb(List<String> targetSentence, List<String> sourceSentence, Alignment alignment) {
        return 0; 
    }

    public CounterMap<String, String> getProbSourceGivenTarget() {
        return counterMap; 
    }

    public void train(List<SentencePair> trainingPairs) {
        counterMap = new CounterMap<String,String>();

        List<List<String>> sourceSentences = new ArrayList<List<String>>();
        List<List<String>> targetSentences = new ArrayList<List<String>>();

        for (SentencePair pair : trainingPairs) {
            List<String> targetWords = pair.getEnglishWords();
            List<String> sourceWords = pair.getFrenchWords();
            sourceSentences.add(sourceWords);
            targetSentences.add(targetWords);
            for (String source : sourceWords) {
                for(String target : targetWords) {
                    counterMap.incrementCount(source, target, 1.0); // TODO: try incrementCount
                }
            }
        }
        sourceModel = new EmpiricalUnigramLanguageModel(sourceSentences);
        targetModel = new EmpiricalUnigramLanguageModel(targetSentences);

    }
}
