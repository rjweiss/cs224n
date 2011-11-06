package cs224n.langmodel;


import cs224n.util.*;

import java.util.Collection;
import java.util.List;

public abstract class TrigramModel extends LanguageModel {
    protected UnigramModel unigramModel;
//    protected BigramModel bigramModel;

    protected Counter<Pair<Pair<String, String>, String>> trigramCounter;
//    protected FastTriCounter fastTriCounter;
    protected double totalTrigramCount;

    @Override
    public void train(Collection<List<String>> sentences) {
        unigramModel = new UnigramModel() {
            @Override
            public double getWordProbability(List<String> sentence, int index) {
                throw new RuntimeException("Not implemented.");
            }
        };
        unigramModel.train(sentences);

        // XXX  Needed for back-off.
//        bigramModel = new BigramModel() {
//            @Override
//            public double getWordProbability(List<String> sentence, int index) {
//                throw new NotImplementedException();
//            }
//        };
//        bigramModel.train(sentences);

        trigramCounter = new Counter<Pair<Pair<String, String>, String>>();
//        fastTriCounter = new FastTriCounter();

        for (List<String> sentence : sentences) {
            for (int i=0; i<=sentence.size(); i++) {
                String first = unigramModel.getWord(sentence, i-2);
                String second = unigramModel.getWord(sentence, i-1);
                String third = unigramModel.getWord(sentence, i);
                trigramCounter.incrementCount(new Pair<Pair<String, String>, String>(new Pair<String, String>(first, second), third), 1.0);
//                fastTriCounter.increment(first, second, third);
            }
        }
        totalTrigramCount = trigramCounter.totalCount();
//        totalTrigramCount = fastTriCounter.getTotalCount();
    }

    @Override
    public Collection<String> getVocabulary() {
        return unigramModel.getVocabulary();
    }

    @Override
    public List<String> generateSentence() {
        return unigramModel.generateSentence();
    }


    // Word probability methods.

    private double getJointProbability(String first, String second, String third) {
        double trigramCount = trigramCounter.getCount(new Pair<Pair<String, String>, String>(new Pair<String, String>(first, second), third));
//        double trigramCount = fastTriCounter.get(first, second, third);
        return trigramCount / totalTrigramCount;
    }

    private double getMarginalProbability(String word) {
        double unigramCount = unigramModel.getWordCount(word);
        return unigramCount / totalTrigramCount;
    }

    public double getMleWordProbability(List<String> sentence, int index) {
        String first = unigramModel.getWord(sentence, index - 2);
        String second = unigramModel.getWord(sentence, index - 1);
        String third = unigramModel.getWord(sentence, index);
        double joint = getJointProbability(first, second, third);
        double marginal = getMarginalProbability(third);
        return joint / marginal;
    }
}



