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

//TODO: need to account for sign of buckets
//TODO: sure word vs possible word probabilities

public class ModelTwoWordAligner implements WordAligner {

    private static final double SURE_THRESHOLD = 0.6;
    private static final double NULL_MASS = 0.1;
    private static final double BUCKET_SIZE = 7;

    private CounterMap<String, String> transProbMap;
    private Counter<Double> bucketProbCounter;
    private LanguageModel sourceModel;
    private LanguageModel targetModel;
    private double totalFrenchWords;
    private double totalEnglishWords;

    public Alignment alignSentencePair(SentencePair sentencePair) {
        double threshold = SURE_THRESHOLD;

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
                transProb *= bucketProbCounter.getCount(getBucket(englishIndex, frenchIndex, englishWords.size(), frenchWords.size()));
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
        for (int j = 0; j < sourceSentence.size(); j++) {
            int alignedTarget = alignment.getAlignedTarget(j);
            if (alignedTarget == -1) {
                // Null word.
                product *= NULL_MASS;
            }
            else {
                double p = transProbMap.getCount(sourceSentence.get(j), targetSentence.get(alignment.getAlignedTarget(j)));
                p *= bucketProbCounter.getCount(getBucket(alignment, j, targetSentence.size(), sourceSentence.size()));
                product *= (1-NULL_MASS) * p;
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

    private double getBucket(int englishIndex, int frenchIndex, int englishLength, int frenchLength) {
        double bucket = englishIndex - frenchIndex * englishLength/(double)frenchLength;
        return Math.round(bucket / BUCKET_SIZE); // look at bucket distribution as normal, divide by some n to get into a smaller space
    }

    private double getBucket(Alignment align, int frenchIndex, int englishLength, int frenchLength) {
        return getBucket(align.getAlignedTarget(frenchIndex), frenchIndex, englishLength, frenchLength);
    }

    public void train(List<SentencePair> trainingPairs) {
        transProbMap = new CounterMap<String,String>();
        bucketProbCounter = new Counter<Double>();

        List<List<String>> sourceSentences = new ArrayList<List<String>>();
        List<List<String>> targetSentences = new ArrayList<List<String>>();

        for (SentencePair pair : trainingPairs) {
            List<String> targetWords = pair.getEnglishWords();
            List<String> sourceWords = pair.getFrenchWords();
            targetWords.add(NULL_WORD); // I have to add the NULL case to the English sentences
            sourceSentences.add(sourceWords);
            targetSentences.add(targetWords);

            for (int englishIndex = 0; englishIndex < targetWords.size(); englishIndex++) {
                for (int frenchIndex = 0; frenchIndex < sourceWords.size(); frenchIndex++) {
                    bucketProbCounter.setCount(getBucket(englishIndex, frenchIndex, targetWords.size(), sourceWords.size()), 1.0);
                    //System.out.println(bucketParam); used this to observe distribution of buckets
                }
            }
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

        int n = 40;
        do {
            CounterMap<String, String> tCountMap = new CounterMap<String, String>();
            Counter<String> totalEnglishCounter = new Counter<String>();
            Set<String> frenchWords = new HashSet<String>();
            Counter<Double> bucketCounter = new Counter<Double>();

            for (SentencePair pair : trainingPairs) {
                List<String> targetWords = pair.getEnglishWords();
                List<String> sourceWords = pair.getFrenchWords();
                targetWords.add(NULL_WORD);

//              for (String frenchWord : sourceWords) {
                for (int frenchIndex = 0; frenchIndex < sourceWords.size(); frenchIndex++) {
                    String frenchWord = sourceWords.get(frenchIndex);
                    frenchWords.add(frenchWord);
//                    System.out.print(frenchWord);
                    double totalSentenceProb = 0;
                    for (int englishIndex = 0; englishIndex < targetWords.size(); englishIndex++) {
                        String englishWord  = targetWords.get(englishIndex);
                        totalSentenceProb += transProbMap.getCount(englishWord, frenchWord)* bucketProbCounter.getCount(getBucket(englishIndex, frenchIndex, targetWords.size(), sourceWords.size()));  // * bucket probability
                    }
//                    System.out.println(": " + totalSentenceProb);

//                  for (String englishWord : targetWords) {
                    for (int englishIndex = 0; englishIndex < targetWords.size(); englishIndex++) {
                        String englishWord  = targetWords.get(englishIndex);
                        double tProb = transProbMap.getCount(englishWord, frenchWord) * bucketProbCounter.getCount(getBucket(englishIndex, frenchIndex, targetWords.size(), sourceWords.size()));
                        tCountMap.incrementCount(englishWord, frenchWord, tProb / totalSentenceProb);
                        totalEnglishCounter.incrementCount(englishWord, tProb / totalSentenceProb);
                        bucketCounter.incrementCount(getBucket(englishIndex, frenchIndex, targetWords.size(), sourceWords.size()), 1.0);
                    }
                }

                targetWords.remove(NULL_WORD);
            }

            for (String englishWord : totalEnglishCounter.keySet()) {
                for (String frenchWord : frenchWords) {
                    transProbMap.setCount(englishWord, frenchWord, tCountMap.getCount(englishWord, frenchWord) / totalEnglishCounter.getCount(englishWord));
                }
            }

            for (Double bucket : bucketCounter.keySet()) {
                bucketProbCounter.setCount(bucket, bucketCounter.getCount(bucket) / bucketCounter.totalCount());
            }

        } while (n-- > 0);

        for (SentencePair pair : trainingPairs) {
            List<String> targetWords = pair.getEnglishWords();
            targetWords.remove(NULL_WORD);
        }

    }

}
