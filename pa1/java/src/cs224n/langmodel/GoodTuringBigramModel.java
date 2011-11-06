package cs224n.langmodel;


import cs224n.util.Counter;
import cs224n.util.Pair;

import java.util.*;

public class GoodTuringBigramModel extends BigramModel {

    private Counter<Double> countCounter;
    private double unseenBigramCount;
    private double cutoff;
    private double totalPpgtProbability;
    private double totalModifiedCountStarCount;
    HashMap<String, Double> marginalCountTable;

    @Override
    public void train(Collection<List<String>> sentences) {
        super.train(sentences);

        countCounter = new Counter<Double>();
        for (Pair<String, String> bigram : bigramCounter.keySet()) {
            countCounter.incrementCount(bigramCounter.getCount(bigram), 1.0);
        }

        // XXX  Instead, count the seen trigrams, and subtract from the computed number of possible trigrams.
        unseenBigramCount = 0;
        for (String first : getVocabulary()) {
            if (first.equals(STOP)) {
                first = START;
            }
            for (String second : getVocabulary()) {
                double count = bigramCounter.getCount(new Pair<String, String>(first, second));
                if (count == 0) {
                    unseenBigramCount++;
                }
            }
        }

        SortedSet<Double> counts = new TreeSet<Double>(countCounter.keySet());
        for (double k : counts) {
            if (countCounter.getCount(k + 1) == 0) {
                cutoff = k;
                break;
            }
        }

        totalPpgtProbability = 0;
        for (Pair<String, String> bigram : bigramCounter.keySet()) {
            double ppgt = getPpgt(bigram.getFirst(), bigram.getSecond());
            totalPpgtProbability += ppgt;
        }
        totalPpgtProbability += (unseenBigramCount * (countCounter.getCount(1.0) / totalBigramCount));


        marginalCountTable = new HashMap<String, Double>();
        for (String first : this.getVocabulary()) {
            if (first.equals(STOP)) {
                first = START;
            }
            double marginalCountSum = 0;
            for (String second : this.getVocabulary()) {
                double count = getModifiedCountStar(first, second);
                marginalCountSum += count;
            }
            marginalCountTable.put(first, marginalCountSum);
        }

        if (false) {
            System.out.printf("\n%20s | %6s %6s %6s %6s %6s %6s %6s\n", "x", "k", "Nk", "Nk+1", "k*", "Pgt", "P'gt", "P''gt");
            System.out.println("-----------------------------------------------------------------------");
            for (String first : getVocabulary()) {
                if (first.equals(STOP)) {
                    first = START;
                }
                for (String second : getVocabulary()) {
                    double k = bigramCounter.getCount(new Pair<String, String>(first, second));
                    double nk = countCounter.getCount(k);
                    if (k == 0) {
                        nk = unseenBigramCount;
                    }
                    double nk1 = countCounter.getCount(k + 1);
                    double kStar = getCountStar(first, second);
                    double pgt = getPgt(first, second);
                    double ppgt = getPpgt(first, second);
                    double pppgt = getGtSmoothedWordProbability(first, second);
                    System.out.printf("%9s, %9s | %6.0f %6.0f %6.0f %6.3f %6.3f %6.3f %6.3f\n", first, second, k, nk, nk1, kStar, pgt, ppgt, pppgt);
                }
            }
            System.out.println();
            System.out.printf("totalBigramCount: %.0f\n", totalBigramCount);
            System.out.printf("totalModifiedBigramCount: %.0f\n", totalBigramCount + (unseenBigramCount * (countCounter.getCount(1.0) / totalBigramCount)));
            System.out.printf("cutoff: %.0f\n", cutoff);
            System.out.printf("totalPpgtProbability: %.5f\n", totalPpgtProbability);
            System.out.println();
        }

        if (false) {
            System.out.printf("\n%-12s| ", "k*");
            for (String second : getVocabulary()) {
                System.out.printf("%9s", second);
            }
            System.out.println();
            for (String first : getVocabulary()) {
                if (first.equals(STOP)) {
                    first = START;
                }
                System.out.printf("%11s | ", first);
                for (String second : getVocabulary()) {
                    System.out.printf("%9.3f", getModifiedCountStar(first, second));
                }
                System.out.println();
            }
        }

        if (false) {
            System.out.printf("\n%-12s| ", "JOINT");
            for (String first : getVocabulary()) {
                System.out.printf("%9s", first);
            }
            System.out.println("\n-----------------------------------------------------------");
            for (String first : getVocabulary()) {
                if (first.equals(STOP)) {
                    first = START;
                }
                System.out.printf("%11s | ", first);
                for (String second : getVocabulary()) {
                    System.out.printf("%9.3f", getJointProbability(first, second));
                }
                System.out.println();
            }
        }

        if (false) {
            System.out.printf("\n%11s | ", "MARGINAL");
            System.out.printf("%9s", "total");
            System.out.println("\n-----------------------");
            for (String first : getVocabulary()) {
                if (first.equals(STOP)) {
                    first = START;
                }
                System.out.printf("%11s | %9.3f\n", first, getMarginalProbability(first));
            }
            System.out.println();
        }

    }

    private double getModifiedCountStar(String first, String second) {
        double k = bigramCounter.getCount(new Pair<String, String>(first, second));
        if (k >= cutoff) {
            return k;
        }
        double nk = countCounter.getCount(k);
        if (k == 0) {
            nk = unseenBigramCount;
        }
        double nk1 = countCounter.getCount(k+1);
        return (k+1) * nk1 / nk;
    }

    private double getCountStar(String first, String second) {
        double k = bigramCounter.getCount(new Pair<String, String>(first, second));
        double nk = countCounter.getCount(k);
        if (k == 0) {
            nk = unseenBigramCount;
        }
        double nk1 = countCounter.getCount(k+1);
        return (k+1) * nk1 / nk;
    }

    private double getPgt(String first, String second) {
        double k = bigramCounter.getCount(new Pair<String, String>(first, second));
        if (k == 0) {
            return countCounter.getCount(1.0) / totalBigramCount;
        }
        return (getCountStar(first, second)) / totalBigramCount;
    }

    private double getPpgt(String first, String second) {
        double p = getPgt(first, second);
        if (p == 0) {
            // XXX  May be incorrect.
            p = this.getMleWordProbability(first, second);
        }
        return p;
    }

//    private double getPgt(String first, String second) {
//        double k = bigramCounter.getCount(new Pair<String, String>(first, second));
//        if (k >= cutoff) {
//            return 0;
//        }
//        double nk = countCounter.getCount(k);
//        double nk1 = countCounter.getCount(k + 1);
//        return ((k + 1) * (nk1 / nk)) / totalBigramCount;
//    }

//    private double getPpgt(String first, String second) {
//        double p = getPgt(first, second);
//        if (p == 0) {
//            p = this.getMleWordProbability(first, second);
//        }
//        return p;
//    }

    public double getGtSmoothedWordProbability(String first, String second) {
        return getPpgt(first, second) / totalPpgtProbability;
    }

    private double getJointProbability(String first, String second) {
        double count = getModifiedCountStar(first, second);
        return count / totalBigramCount;
    }

    private double getMarginalProbability(String word) {
        double marginalCountSum = marginalCountTable.get(word);
        return marginalCountSum / totalBigramCount;
    }

    @Override
    public double getWordProbability(List<String> sentence, int index) {
        String first = unigramModel.getWord(sentence, index - 1);
        String second = unigramModel.getWord(sentence, index);
        double joint = getJointProbability(first, second);
        double marginal = getMarginalProbability(first);
        double probability = joint / marginal;
        return probability;
    }
}
