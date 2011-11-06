package cs224n.wordaligner;

import cs224n.langmodel.EmpiricalUnigramLanguageModel;
import cs224n.langmodel.LanguageModel;
import cs224n.util.Alignment;
import cs224n.util.Counter;
import cs224n.util.CounterMap;
import cs224n.util.SentencePair;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ModelOneWordAligner implements WordAligner {

    private static final double NULL_MASS = 0.05;
//    private static final double SURE_THRESHOLD = 0.8;

    private CounterMap<String, String> transProbMap;
    private LanguageModel sourceModel;
    private LanguageModel targetModel;
    private double totalFrenchWords;
    private double totalEnglishWords;

   public Alignment alignSentencePair(SentencePair sentencePair) {
//        double threshold = SURE_THRESHOLD;

        Alignment alignment = new Alignment();
        List<String> englishWords = sentencePair.getEnglishWords();
        List<String> frenchWords = sentencePair.getFrenchWords();
        englishWords.add(NULL_WORD);
        int numEnglishWords = sentencePair.getEnglishWords().size();
        int numFrenchWords = sentencePair.getFrenchWords().size();
        for (int frenchIndex = 0; frenchIndex < numFrenchWords; frenchIndex++) {
            List<Integer> transProbMaximaIndices = new ArrayList<Integer>();
            String frenchWord = frenchWords.get(frenchIndex);
            double maxTransProb = -1;
            double almostMaxTransProb = -1;

            for (int englishIndex = 0; englishIndex < numEnglishWords; englishIndex++) {
                double transProb = transProbMap.getCount(englishWords.get(englishIndex), frenchWord);
                if (transProb > maxTransProb) {
                    transProbMaximaIndices.clear();
                    almostMaxTransProb = maxTransProb;
                    maxTransProb = transProb;
                }
                if (transProb == maxTransProb) {
                    transProbMaximaIndices.add(englishIndex);
                }
            }

            if (transProbMaximaIndices.size() == 1) {
                alignment.addAlignment(transProbMaximaIndices.get(0), frenchIndex, true);
            }
            else {
                int diffMin = Integer.MAX_VALUE;
                int index = -1;
                for (int maxIndex : transProbMaximaIndices) {
                    int diff = Math.abs(frenchIndex - maxIndex);
                    if (diff < diffMin) {
                        diffMin = diff;
                        index = maxIndex;
                    }
                    alignment.addAlignment(index, frenchIndex, false);
                }
            }
        }
        englishWords.remove(NULL_WORD);
        return alignment;
    }

    public double getAlignmentProb(List<String> targetSentence, List<String> sourceSentence, Alignment alignment) {
        double product = 1;

//        for (int j = 0; j < sourceSentence.size(); j++) {
//            double t = transProbMap.getCount(sourceSentence.get(j), targetSentence.get(alignment.getAlignedTarget(j)));
//            product *= t;
//        }

        for (int j = 0; j < sourceSentence.size(); j++) {
            int alignedTarget = alignment.getAlignedTarget(j);
            if (alignedTarget == -1) {
                // Null word.
                product *= NULL_MASS;
            }
//            if (targetSentence.get(alignment.getAlignedTarget(j)).equals(NULL_WORD)) {
//                product *= NULL_MASS;
//            }
            else {
                double p = transProbMap.getCount(sourceSentence.get(j), targetSentence.get(alignment.getAlignedTarget(j)));
                product *= (1 - NULL_MASS) * p;
            }
        }
        return product;
    }

    public CounterMap<String, String> getProbSourceGivenTarget() {
        CounterMap<String, String> reversePropMap = new CounterMap<String, String>();
        for (String key : transProbMap.keySet()) {
            for (String value : transProbMap.getCounter(key).keySet()) {
                reversePropMap.setCount(value, key, transProbMap.getCount(key, value));
            }
        }
        return reversePropMap;
    }

    public void train(List<SentencePair> trainingPairs) {
        transProbMap = new CounterMap<String,String>();

        List<List<String>> sourceSentences = new ArrayList<List<String>>();
        List<List<String>> targetSentences = new ArrayList<List<String>>();

        for (SentencePair pair : trainingPairs) {
            List<String> targetWords = pair.getEnglishWords();
            List<String> sourceWords = pair.getFrenchWords();
            targetWords.add(NULL_WORD); // I have to add the NULL case to the English sentences
            sourceSentences.add(sourceWords);
            targetSentences.add(targetWords);
        }

        sourceModel = new EmpiricalUnigramLanguageModel(sourceSentences);
        targetModel = new EmpiricalUnigramLanguageModel(targetSentences);

        totalEnglishWords = targetModel.getVocabulary().size();
        totalFrenchWords = sourceModel.getVocabulary().size();

        for (String englishWord : targetModel.getVocabulary()) {
            for (String frenchWord : sourceModel.getVocabulary()) {
                transProbMap.setCount(englishWord, frenchWord, 1 / totalFrenchWords);
            }
        }

        int n = 40; // do this 40 times for EM
        do {
            CounterMap<String, String> tCountMap = new CounterMap<String, String>();
            Counter<String> totalEnglishCounter = new Counter<String>();
            Set<String> frenchWords = new HashSet<String>();

            for (SentencePair pair : trainingPairs) {
                List<String> targetWords = pair.getEnglishWords();
                List<String> sourceWords = pair.getFrenchWords();


                for (String frenchWord : sourceWords) {
                    frenchWords.add(frenchWord);
//                    System.out.print(frenchWord);
                    double totalSentenceProb = 0;
                    for (String englishWord : targetWords) {
                        totalSentenceProb += transProbMap.getCount(englishWord, frenchWord);
                    }
//                    System.out.println(": " + totalSentenceProb);

                    for (String englishWord : targetWords) {
                        double tProb = transProbMap.getCount(englishWord, frenchWord);
                        tCountMap.incrementCount(englishWord, frenchWord, tProb / totalSentenceProb);
                        totalEnglishCounter.incrementCount(englishWord, tProb / totalSentenceProb);
                    }
                }
            }

            for (String englishWord : totalEnglishCounter.keySet()) {
                for (String frenchWord : frenchWords) {
                    transProbMap.setCount(englishWord, frenchWord, tCountMap.getCount(englishWord, frenchWord) / totalEnglishCounter.getCount(englishWord));
                }
            }

//            System.out.println();
        } while (n-- > 0);

        for (SentencePair pair : trainingPairs) {
            List<String> targetWords = pair.getEnglishWords();
            targetWords.remove(NULL_WORD);
        }

//        System.out.println();
    }
}
