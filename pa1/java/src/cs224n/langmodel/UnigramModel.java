package cs224n.langmodel;

import cs224n.util.Counter;

import java.util.*;

public abstract class UnigramModel extends LanguageModel {

    private Counter<String> wordCounter;
    private double totalCount;
    private Set<String> vocabulary;

    @Override
    public void train(Collection<List<String>> sentences) {
        wordCounter = new Counter<String>();
        for (List<String> sentence : sentences) {
            for (String word : sentence) {
                wordCounter.incrementCount(word, 1.0);
            }
            wordCounter.incrementCount(STOP, 1.0);
        }
        totalCount = wordCounter.totalCount();

        vocabulary = new TreeSet<String>();
        vocabulary.addAll(wordCounter.keySet());
        vocabulary.add(UNK);
        vocabulary = Collections.unmodifiableSet(vocabulary);
    }

    @Override
    public Collection<String> getVocabulary() {
        return vocabulary;
    }

    public double getTotalWordCount() {
        return totalCount;
    }

    public double getWordCount(String word) {
        return wordCounter.getCount(word);
    }

    @Override
    public List<String> generateSentence() {
        List<String> sentence = new ArrayList<String>();
        String word;
        do {
            word = generateWord();
            sentence.add(word);
        }
        while (!word.equals(STOP));
        return sentence;
    }

    private String generateWord() {
        double sample = Math.random();
        double sum = 0.0;
        for (String word : wordCounter.keySet()) {
            sum += wordCounter.getCount(word) / (totalCount + 1.0);
            if (sum > sample) {
                return word;
            }
        }
        // Some mass reserved for unknown words: 1 / (totalCount+1)
        return LanguageModel.UNK;
    }

    protected String getWord(List<String> sentence, int index) {
        if (index < 0) {
            return START;
        }
        if (index == sentence.size()) {
            return STOP;
        }
        else {
            String word = sentence.get(index);
            if (!vocabulary.contains(word)) {
                word = UNK;
            }
            return word;
        }
    }


    // Word probability methods.

    public double getMleWordProbability(List<String> sentence, int index) {
        String word = getWord(sentence, index);
        return getMleWordProbability(word);
    }

    public double getMleWordProbability(String word) {
        // Some mass reserved for unknown words: 1 / (totalCount+1)
        double count = Math.max(1.0, wordCounter.getCount(word));
        return count / (totalCount + 1.0);
    }

}
