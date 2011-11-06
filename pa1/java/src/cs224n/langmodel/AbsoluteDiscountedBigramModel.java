package cs224n.langmodel;


import cs224n.util.Pair;

import java.util.*;

public class AbsoluteDiscountedBigramModel extends BigramModel {

    double unseenCount = 0;
    HashMap<String, Double> marginalCountTable;
    HashMap<String, Double> katzBackoffTable;

    @Override
    public void train(Collection<List<String>> sentences) {
        super.train(sentences);


        if (false) {
            TreeSet<String> starters = new TreeSet<String>(unigramModel.getVocabulary());
            starters.remove(STOP);
            starters.add(START);
            System.out.printf("\n%-12s| ", "BEFORE");
            for (String x : unigramModel.getVocabulary()) {
                System.out.printf("%8s", x);
            }
            System.out.println();
            for (String y : starters) {
                System.out.printf("%9s | ", y);
                for (String x : unigramModel.getVocabulary()) {
                    System.out.printf("%9.3f", bigramCounter.getCount(new Pair<String, String>(y, x)));
                }
                System.out.println();
            }
        }

        double discount = 0.75;
        double discountedTotal = 0;

        Set<Pair<String, String>> bigrams = bigramCounter.keySet();
        for (Pair<String, String> bigram : bigrams) {
            double count = bigramCounter.getCount(bigram);
            bigramCounter.setCount(bigram, count - discount);
            discountedTotal += discount;
        }

        double unseenBigramCount = 0;

        for (String first : this.getVocabulary()) {
            if (first.equals(STOP)) {
                first = START;
            }
            for (String second : this.getVocabulary()) {
                Pair<String, String> bigram = new Pair<String, String>(first, second);
                double count = bigramCounter.getCount(bigram);
                if (count == 0) {
                    unseenBigramCount++;
                }
            }
        }

        unseenCount = discountedTotal / unseenBigramCount;

//        marginalCountTable = new HashMap<String, Double>();
//        for (String second : this.getVocabulary()) {
//            double marginalCountSum = 0;
//            for (String first : this.getVocabulary()) {
//                if (first.equals(STOP)) {
//                    first = START;
//                }
//                double count = bigramCounter.getCount(new Pair<String, String>(first, second));
//                if (count == 0) {
//                    count = unseenCount;
//                }
//                marginalCountSum += count;
//            }
//            marginalCountTable.put(second, marginalCountSum);
//        }

        marginalCountTable = new HashMap<String, Double>();
        for (String first : this.getVocabulary()) {
            if (first.equals(STOP)) {
                first = START;
            }
            double marginalCountSum = 0;
            for (String second : getVocabulary()) {
                double count = bigramCounter.getCount(new Pair<String, String>(first, second));
                if (count == 0) {
                    count = unseenCount;
                }
                marginalCountSum += count;
            }
            marginalCountTable.put(first, marginalCountSum);
        }

        //double marginalUnigramMleProbability = 0;
        //for (String word : getVocabulary()) {
        //    System.out.printf("%9s: %6.3f\n", word, unigramModel.getMleWordProbability(word));
        //    marginalUnigramMleProbability += unigramModel.getMleWordProbability(word);
        //}
        //System.out.printf("marginalUnigramMleProbability: %.5f\n", marginalUnigramMleProbability);

        katzBackoffTable = new HashMap<String, Double>();
        for (String first : getVocabulary()) {
            if (first.equals(STOP)) {
                first = START;
            }
            double marginalUnseenBigramConditionalProbability = 0;
            double marginalUnigramMleProbability = 0;
            for (String second : getVocabulary()) {
                double count = bigramCounter.getCount(new Pair<String, String>(first, second));
                if (count == 0) {
                    marginalUnseenBigramConditionalProbability += getConditionalProbability(first, second);
                    marginalUnigramMleProbability += unigramModel.getMleWordProbability(second);
                }

            }
            double alpha = marginalUnseenBigramConditionalProbability / marginalUnigramMleProbability;
//            System.out.printf("%s %.4f %.4f %.4f\n", first, marginalUnseenBigramConditionalProbability, marginalUnigramMleProbability, alpha);

            for (String second : getVocabulary()) {
                katzBackoffTable.put(first, alpha);
            }
        }



        if (false) {
            System.out.printf("\n%20s | %6s %6s %6s\n", "x", "k", "c", "up");//, "Nk", "Nk+1", "k*", "Pgt", "P'gt", "P''gt");
            System.out.println("-----------------------------------------------------------------------");
            for (String first : getVocabulary()) {
                if (first.equals(STOP)) {
                    first = START;
                }
                for (String second : getVocabulary()) {
                    double count = bigramCounter.getCount(new Pair<String, String>(first, second));
                    double cp = getConditionalProbability(first, second);
                    if (count == 0) {

                    }
                    System.out.printf("%9s, %9s | %6.3f %6.3f\n", first, second, count, cp);
                }
            }
            System.out.println();
            System.out.printf("totalBigramCount: %.0f\n", totalBigramCount);
            System.out.println();
        }

        if (false) {
            TreeSet<String> starters = new TreeSet<String>(getVocabulary());
            starters.remove(STOP);
            starters.add(START);
            System.out.printf("\n%-12s| ", "AFTER");
            for (String x : getVocabulary()) {
                System.out.printf("%8s", x);
            }
            System.out.println();
            for (String y : starters) {
                System.out.printf("%9s | ", y);
                for (String x : getVocabulary()) {
                    System.out.printf("%9.3f", bigramCounter.getCount(new Pair<String, String>(y, x)));
                }
                System.out.println();
            }
        }


        if (false) {
            TreeSet<String> starters = new TreeSet<String>(getVocabulary());
            starters.remove(STOP);
            starters.add(START);
            System.out.printf("\n%-12s| ", "JOINT");
            for (String x : getVocabulary()) {
                System.out.printf("%9s", x);
            }
            System.out.println("\n-----------------------------------------------------------");
            for (String y : starters) {
                System.out.printf("%9s | ", y);
                for (String x : getVocabulary()) {
                    System.out.printf("%9.3f", getJointProbability(y, x));
                }
                System.out.println();
            }
        }


//        {
//            System.out.printf("\n%11s | ", "MARGINAL");
//            for (String x : unigramModel.getVocabulary()) {
//                System.out.printf("%9s", x);
//            }
//            System.out.println("\n-----------------------------------------------------------");
//            System.out.printf("%11s | ", "total");
//            for (String x : unigramModel.getVocabulary()) {
//                System.out.printf("%9.3f", getMarginalProbability(x));
//            }
//            System.out.println();
//        }

        if (false) {
            System.out.printf("\n%11s | ", "MARGINAL");
            System.out.printf("%9s", "total");
            System.out.println("\n-----------------------");
            for (String first : unigramModel.getVocabulary()) {
                if (first.equals(STOP)) {
                    first = START;
                }
                System.out.printf("%11s | %9.3f\n", first, getMarginalProbability(first));
            }
            System.out.println();
        }

//        {
//            TreeSet<String> starters = new TreeSet<String>(unigramModel.getVocabulary());
//            starters.remove(STOP);
//            starters.add(START);
//            System.out.printf("\n%11s | ", "CONDITIONAL");
//            for (String x : unigramModel.getVocabulary()) {
//                System.out.printf("%9s", x);
//            }
//            System.out.println("\n-----------------------------------------------------------");
//            for (String first : starters) {
//                System.out.printf("%11s | ", first);
//                for (String second : unigramModel.getVocabulary()) {
//                    System.out.printf("%9.3f", getAbsoluteDiscountedWordProbability(first, second));
//                }
//                System.out.println();
//            }
//            System.out.println();
//        }

        if (false) {
            System.out.printf("\n%11s | ", "CONDITIONAL");
            for (String x : unigramModel.getVocabulary()) {
                System.out.printf("%9s", x);
            }
            System.out.println("\n-----------------------------------------------------------");
            for (String first : this.getVocabulary()) {
                if (first.equals(STOP)) {
                    first = START;
                }
                System.out.printf("%11s | ", first);
                for (String second : this.getVocabulary()) {
                    System.out.printf("%9.3f", getAbsoluteDiscountedWordProbability(first, second));
                }
                System.out.println();
            }
            System.out.println();
        }



        System.out.println();
    }

    private double getJointProbability(String first, String second) {
        double bigramCount = bigramCounter.getCount(new Pair<String, String>(first, second));
        if (bigramCount == 0) {
            bigramCount = unseenCount;
        }
        return bigramCount / totalBigramCount;
    }

    private double getMarginalProbability(String word) {
//        double marginalCountSum = 0;
//        for (String first : this.getVocabulary()) {
//            if (first.equals(STOP)) {
//                first = START;
//            }
//            double count = bigramCounter.getCount(new Pair<String, String>(first, word));
//            if (count == 0) {
//                count = unseenCount;
//            }
//            marginalCountSum += count;
//        }
//        System.err.println("!: " + word);
        double marginalCountSum = marginalCountTable.get(word);

        return marginalCountSum / totalBigramCount;
    }

    private double getConditionalProbability(String first, String second) {
        double joint = getJointProbability(first, second);
        double marginal = getMarginalProbability(first);
        double probability = joint / marginal;
        return probability;
    }

    private double getKatzBackoffProbability(String first, String second) {
        Pair<String, String> bigram = new Pair<String, String>(first, second);
        double count = bigramCounter.getCount(bigram);
        if (count == 0) {
            double alpha = katzBackoffTable.get(first);
            return alpha * unigramModel.getMleWordProbability(second);
        }
        else {
            return getConditionalProbability(bigram.getFirst(), bigram.getSecond());
        }
    }

    public double getAbsoluteDiscountedWordProbability(List<String> sentence, int index) {
        String first = unigramModel.getWord(sentence, index - 1);
        String second = unigramModel.getWord(sentence, index);
        return getAbsoluteDiscountedWordProbability(first, second);
    }

    public double getAbsoluteDiscountedWordProbability(String first, String second) {
        double joint = getJointProbability(first, second);
        double marginal = getMarginalProbability(first);
        double probability = joint / marginal;
//        System.out.printf("(%s, %s): %.5f\n", first, second, probability);
        return probability;
    }

    @Override
    public double getWordProbability(List<String> sentence, int index) {
        return getAbsoluteDiscountedWordProbability(sentence, index);
    }
}
