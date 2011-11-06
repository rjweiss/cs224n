package cs224n.langmodel;

import cs224n.util.Counter;

import java.util.Collection;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;


public class GoodTuringUnigramModel extends UnigramModel {

    private Counter<Double> countCounter;
    private double cutoff;
    private double totalPpgtProbability;

    @Override
    public void train(Collection<List<String>> sentences) {
        super.train(sentences);

        countCounter = new Counter<Double>();
        Collection<String> words = getVocabulary();
        for (String word : words) {
            countCounter.incrementCount(this.getWordCount(word), 1.0);
        }

        SortedSet<Double> counts = new TreeSet<Double>(countCounter.keySet());
        for (double k : counts) {
            if (countCounter.getCount(k + 1) == 0) {
                cutoff = k;
                break;
            }
        }

        totalPpgtProbability = 0;
        for (String word : words) {
            double ppgt = getPpgt(word);
            totalPpgtProbability += ppgt;
        }

        System.out.println();
    }

    @Override
    public double getWordProbability(List<String> sentence, int index) {
        String word = this.getWord(sentence, index);
        return getGtSmoothedWordProbability(word);
    }


    private double getPgt(String word) {
        double k = this.getWordCount(word);
        if (k >= cutoff) {
            return 0;
        }
        double nk = countCounter.getCount(k);
        double nk1 = countCounter.getCount(k + 1);
        return ((k + 1) * (nk1 / nk)) / this.getTotalWordCount();
    }

    private double getPpgt(String word) {
        double p = getPgt(word);
        if (p == 0) {
            p = this.getMleWordProbability(word);
        }
        return p;
    }

    public double getGtSmoothedWordProbability(String word) {
        return getPpgt(word) / totalPpgtProbability;
    }

}
