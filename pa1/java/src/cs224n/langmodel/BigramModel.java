package cs224n.langmodel;


import cs224n.util.Counter;
import cs224n.util.Pair;

import java.util.Collection;
import java.util.List;

public abstract class BigramModel extends LanguageModel {

    protected UnigramModel unigramModel;

    protected Counter<Pair<String, String>> bigramCounter;
    protected double totalBigramCount;

    @Override
    public void train(Collection<List<String>> sentences) {
        unigramModel = new UnigramModel() {
            @Override
            public double getWordProbability(List<String> sentence, int index) {
                throw new RuntimeException("Not implemented.");
            }
        };
        unigramModel.train(sentences);

        bigramCounter = new Counter<Pair<String, String>>();

        for (List<String> sentence : sentences) {
            for (int i=0; i<=sentence.size(); i++) {
                String lastWord = unigramModel.getWord(sentence, i-1);
                String word = unigramModel.getWord(sentence, i);
                bigramCounter.incrementCount(new Pair<String, String>(lastWord, word), 1.0);
            }
        }
        // add ..
        // XXX REMOVE
//        bigramCounter.incrementCount(new Pair<String, String>(START, STOP), 1.0);
//        bigramCounter.incrementCount(new Pair<String, String>("A", "C"), 0.0);
//        bigramCounter.incrementCount(new Pair<String, String>("B", "C"), 0.0);
//        bigramCounter.incrementCount(new Pair<String, String>("C", "A"), 0.0);
//        bigramCounter.incrementCount(new Pair<String, String>("C", "B"), 0.0);
        totalBigramCount = bigramCounter.totalCount();

//        TreeSet<String> starters = new TreeSet<String>(unigramModel.getVocabulary());
//        starters.remove(STOP);
//        starters.add(START);
//        System.out.print("                ");
//        for (String x : unigramModel.getVocabulary()) {
//            System.out.printf("%8s", x);
//        }
//        System.out.println();
//        for (String y : starters) {
//            System.out.printf("%9s | ", y);
//            for (String x : unigramModel.getVocabulary()) {
//                System.out.printf("%9.0f", bigramCounter.getCount(new Pair<String, String>(y, x)));
//            }
//            System.out.println();
//        }

//        System.out.println();
    }

    @Override
    public Collection<String> getVocabulary() {
        return unigramModel.getVocabulary();
    }

    public double getTotalWordCount() {
        return unigramModel.getTotalWordCount();
    }

    public double getWordCount(String word) {
        return unigramModel.getWordCount(word);
    }

    @Override
    public List<String> generateSentence() {
        return unigramModel.generateSentence();
    }


    // Word probability methods.

    private double getJointProbability(String first, String second) {
        double bigramCount = bigramCounter.getCount(new Pair<String, String>(first, second));
//        return (bigramCount + 1) / (totalBigramCount + (unigramModel.getVocabulary().size() * unigramModel.getVocabulary().size()));
        return bigramCount / totalBigramCount;
    }

    private double getMarginalProbability(String word) {
        double unigramCount = unigramModel.getWordCount(word);
//        return (unigramCount + unigramModel.getVocabulary().size()) / (totalBigramCount + (unigramModel.getVocabulary().size() * unigramModel.getVocabulary().size()));
        return unigramCount / totalBigramCount;
    }

    public double getMleWordProbability(List<String> sentence, int index) {
        String first = unigramModel.getWord(sentence, index - 1);
        String second = unigramModel.getWord(sentence, index);
        return getMleWordProbability(first, second);
    }

    public double getMleWordProbability(String first, String second) {
        double joint = getJointProbability(first, second);
        double marginal = getMarginalProbability(second);
        return joint / marginal;
    }

    public double getAddOneWordProbability(List<String> sentence, int index) {
        String first = unigramModel.getWord(sentence, index - 1);
        String second = unigramModel.getWord(sentence, index);
        double jointCount = bigramCounter.getCount(new Pair<String, String>(first, second));
        double marginalCount = unigramModel.getWordCount(second);
        return (jointCount + 1) / (marginalCount + unigramModel.getVocabulary().size());
    }

    public double getLaplaceSmoothedMleWordProbability(List<String> sentence, int index) {
        String prev = unigramModel.getWord(sentence, index - 1);
        String word = unigramModel.getWord(sentence, index);
        return getLaplaceSmoothedMleWordProbability(prev, word);
    }

    public double getLaplaceSmoothedMleWordProbability(String prev, String word) {
        double count = bigramCounter.getCount(new Pair<String, String>(prev, word));
        // Apply Laplace.
//        return (count + 1) / (unigramModel.getTotalWordCount() + unigramModel.getVocabulary().size());
        return count / totalBigramCount;
    }
}
