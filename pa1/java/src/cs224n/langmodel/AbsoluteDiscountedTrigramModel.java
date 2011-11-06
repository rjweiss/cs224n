package cs224n.langmodel;


import cs224n.util.Pair;

import java.util.*;

public class AbsoluteDiscountedTrigramModel extends TrigramModel {

    double unseenCount = 0;
    HashMap<Pair<String, String>, Double> marginalCountTable;

    @Override
    public void train(Collection<List<String>> sentences) {
        super.train(sentences);

//        if (false) {
//            System.out.printf("\n%-12s| ", "BEFORE");
//            for (String x : unigramModel.getVocabulary()) {
//                System.out.printf("%8s", x);
//            }
//            System.out.println("\n-----------------------------------------------------------");
//            for (Pair<Pair<String, String>, String> trigram : trigramCounter.keySet()) {
//                System.out.printf("%11s | ", trigram.getFirst());
//                for (String x : unigramModel.getVocabulary()) {
//                    System.out.printf("%9.3f", trigramCounter.getCount(new Pair<Pair<String, String>, String>(trigram.getFirst(), x)));
//                }
//                System.out.println();
//            }
//        }

        double unseenBigramCount = Math.pow(getVocabulary().size(), 3) - totalTrigramCount;

        double discount = 0.75;
        double discountedTotal = 0;

        for (Pair<Pair<String, String>, String> trigram : trigramCounter.keySet()) {
            double count = trigramCounter.getCount(trigram);
            trigramCounter.setCount(trigram, count - discount);
            discountedTotal += discount;
        }
//        discountedTotal = fastTriCounter.getTotalCount() * discount;
//        fastTriCounter.subtract(discount);


//        // XXX  Instead, count the seen trigrams, and subtract from the computed number of possible trigrams.
//        for (String first : this.getVocabulary()) {
//            if (first.equals(STOP)) {
//                first = START;
//            }
//            for (String second : this.getVocabulary()) {
//                if (second.equals(STOP)) {
//                    second = START;
//                }
//                for (String third : this.getVocabulary()) {
//                    Pair<Pair<String, String>, String> trigram = new Pair<Pair<String, String>, String>(new Pair<String, String>(first, second), third);
//                    double count = trigramCounter.getCount(trigram);
//                    if (count == 0) {
//                        unseenBigramCount++;
//                    }
//                }
//            }
//        }

        System.out.println();
        int m = 0, n = 0;
        int size = getVocabulary().size();
        unseenCount = discountedTotal / unseenBigramCount;
        marginalCountTable = new HashMap<Pair<String, String>, Double>();
        for (String first : getVocabulary()) {
//            System.out.printf("%d / %d\n", n++, size);
            m = 0;
            if (first.equals(STOP)) {
                first = START;
            }
            for (String second : getVocabulary()) {
                if ((m++ % 100) == 0) {
//                    System.out.printf("    %d / %d\n", m, size);
                }
                if (second.equals(STOP)) {
                    second = START;
                }
                double marginalCountSum = 0;
                for (String third : getVocabulary()) {
                    double count = trigramCounter.getCount(new Pair<Pair<String, String>, String>(new Pair<String, String>(first, second), third));
//                    double count = fastTriCounter.get(first, second, third);
                    if (count == 0) {
                        count = unseenCount;
                    }
                    marginalCountSum += count;
                }
                marginalCountTable.put(new Pair<String, String>(first, second), marginalCountSum);
            }
        }



//        if (false) {
//            System.out.printf("\n%-12s| ", "AFTER");
//            for (String x : unigramModel.getVocabulary()) {
//                System.out.printf("%8s", x);
//            }
//            System.out.println("\n-----------------------------------------------------------");
//            for (Pair<Pair<String, String>, String> trigram : trigramCounter.keySet()) {
//                System.out.printf("%11s | ", trigram.getFirst());
//                for (String x : unigramModel.getVocabulary()) {
//                    System.out.printf("%9.3f", trigramCounter.getCount(new Pair<Pair<String, String>, String>(trigram.getFirst(), x)));
//                }
//                System.out.println();
//            }
//        }


        if (false) {
            System.out.printf("\n%-12s| ", "JOINT");
            for (String x : unigramModel.getVocabulary()) {
                System.out.printf("%9s", x);
            }
            System.out.println("\n-----------------------------------------------------------");
            for (String y : unigramModel.getVocabulary()) {
                if (y.equals(STOP)) {
                    y = START;
                }
                for (String z : unigramModel.getVocabulary()) {
                    if (z.equals(STOP)) {
                        z = START;
                    }
                    System.out.printf("%s, %s | ", y, z);
                    for (String x : unigramModel.getVocabulary()) {
                        System.out.printf("%9.3f", getJointProbability(y, z, x));
                    }
                    System.out.println();
                }
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

//        if (false) {
//            System.out.printf("\n%11s | ", "MARGINAL");
//            System.out.printf("%9s", "total");
//            System.out.println("\n-----------------------");
//            for (String first : unigramModel.getVocabulary()) {
//                if (first.equals(STOP)) {
//                    first = START;
//                }
//                System.out.printf("%11s | %9.3f\n", first, getMarginalProbability(first));
//            }
//            System.out.println();
//        }

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

//        if (false) {
//            System.out.printf("\n%11s | ", "CONDITIONAL");
//            for (String x : unigramModel.getVocabulary()) {
//                System.out.printf("%9s", x);
//            }
//            System.out.println("\n-----------------------------------------------------------");
//            for (String first : this.getVocabulary()) {
//                if (first.equals(STOP)) {
//                    first = START;
//                }
//                System.out.printf("%11s | ", first);
//                for (String second : this.getVocabulary()) {
//                    System.out.printf("%9.3f", getAbsoluteDiscountedWordProbability(first, second));
//                }
//                System.out.println();
//            }
//            System.out.println();
//        }

        System.out.println();
    }


    private double getJointProbability(String first, String second, String third) {
        double trigramCount = trigramCounter.getCount(new Pair<Pair<String, String>, String>(new Pair<String, String>(first, second), third));
//        double trigramCount = fastTriCounter.get(first, second, third);
        if (trigramCount == 0) {
            trigramCount = unseenCount;
        }
        return trigramCount / totalTrigramCount;
    }

    private double getMarginalProbability(Pair<String, String> bigram) {
        double marginalCountSum = marginalCountTable.get(bigram);
        return marginalCountSum / totalTrigramCount;
    }

    public double getAbsoluteDiscountedWordProbability(List<String> sentence, int index) {
        String first = unigramModel.getWord(sentence, index - 2);
        String second = unigramModel.getWord(sentence, index - 1);
        String third = unigramModel.getWord(sentence, index);
        return getAbsoluteDiscountedWordProbability(first, second, third);
    }

    public double getAbsoluteDiscountedWordProbability(String first, String second, String third) {
        double joint = getJointProbability(first, second, third);
        double marginal = getMarginalProbability(new Pair<String, String>(first, second));
        double probability = joint / marginal;
//        System.out.printf("(%s, %s): %.5f\n", first, second, probability);
        return probability;
    }

    @Override
    public double getWordProbability(List<String> sentence, int index) {
        return getAbsoluteDiscountedWordProbability(sentence, index);
    }

}
