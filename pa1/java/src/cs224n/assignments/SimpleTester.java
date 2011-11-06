package cs224n.assignments;

import cs224n.langmodel.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SimpleTester {

    public static void trainUnigramModel(LanguageModel model) {
        List<List<String>> sentences = new ArrayList<List<String>>();
        sentences.add(Arrays.asList("A B B A".split(" ")));
        sentences.add(Arrays.asList("A B B A A A".split(" ")));
        sentences.add(Arrays.asList("B A A B".split(" ")));
        sentences.add(Arrays.asList("A".split(" ")));
        sentences.add(Arrays.asList("A A A".split(" ")));
        sentences.add(Arrays.asList("".split(" ")));
        model.train(sentences);
    }

    public static void trainBigramModel(LanguageModel model) {
        List<List<String>> sentences = new ArrayList<List<String>>();
        sentences.add(Arrays.asList("A B B A".split(" ")));
        sentences.add(Arrays.asList("A B B A A A".split(" ")));
        sentences.add(Arrays.asList("B A A B".split(" ")));
        sentences.add(Arrays.asList("A".split(" ")));
        sentences.add(Arrays.asList("A A A".split(" ")));
//        sentences.add(Arrays.asList("".split(" ")));
        sentences.add(Arrays.asList("C C C C C C C C C C C".split(" ")));
        model.train(sentences);
    }

    public static void testProbabilityDistribution(LanguageModel model) {
        List<List<String>> contexts = new ArrayList<List<String>>();
        contexts.add(new ArrayList<String>(Arrays.asList("".split(" "))));
        contexts.add(new ArrayList<String>(Arrays.asList("united".split(" "))));
        contexts.add(new ArrayList<String>(Arrays.asList("to the".split(" "))));
        contexts.add(new ArrayList<String>(Arrays.asList("the quick brown".split(" "))));
        contexts.add(new ArrayList<String>(Arrays.asList("lalok nok crrok".split(" "))));

        for (int i = 0; i < 10; i++) {
            List<String> randomSentence = new ArrayList<String>(model.generateSentence());
            contexts.add(randomSentence.subList(0, (int) (Math.random() * randomSentence.size())));
        }

        for (List<String> context : contexts) {
            System.out.print("Testing context " + context + " ... ");
            double modelsum = model.checkProbability(context);
            if (Math.abs(1.0 - modelsum) > 1e-6) {
                System.out.println("\nWARNING: probability distribution does not sum up to one. Sum:" + modelsum);
            }
            else {
                System.out.println("GOOD!");
            }
        }
        System.out.println();
    }


    public static void main(String[] args) {
        LanguageModel model;
//        model = new EmpiricalUnigramLanguageModel();
//        model = new EmpiricalUnigramModel();
//        model = new GoodTuringUnigramModel();
//        model = new EmpiricalBigramModel();
//        model = new AbsoluteDiscountedBigramModel();
//        model = new GoodTuringBigramModel();
//        model = new EmpiricalTrigramModel();
        model = new AbsoluteDiscountedTrigramModel();
//        model = new EmpiricalBigramLanguageModel();
//        model = new BigramModel() {
//            @Override
//            public double getWordProbability(List<String> sentence, int index) {
//                return getLaplaceSmoothedMleWordProbability(sentence, index);
//            }
//        };

//        trainUnigramModel(model);
        trainBigramModel(model);
        
        testProbabilityDistribution(model);

    }

}
