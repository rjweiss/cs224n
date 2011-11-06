package cs224n.assignments;

import cs224n.classify.*;
import cs224n.math.*;
import cs224n.util.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * Harness for building and testing a maximum-entropy classifier.
 *
 * @author Dan Klein
 * @author Jenny Finkel (changed for sequence classification / NER)
 */
public class MaximumEntropyClassifierTester {
  public static class MaximumEntropyClassifier <F,L> implements ProbabilisticClassifier<F, L> {
    /**
     * Factory for training MaximumEntropyClassifiers.
     */
    public static class Factory <F,L> implements ClassifierFactory<F, L> {
      double sigma;
      int iterations;

      public ProbabilisticClassifier<F, L> trainClassifier(List<LabeledDatum<F, L>> trainingData) {
        // build data encodings so the inner loops can be efficient
        Encoding<F, L> encoding = buildEncoding(trainingData);
        IndexLinearizer indexLinearizer = buildIndexLinearizer(encoding);
        double[] initialWeights = buildInitialWeights(indexLinearizer);
        EncodedDatum[] data = encodeData(trainingData, encoding);
        // build a minimizer object
        GradientMinimizer minimizer = new LBFGSMinimizer(iterations);
        // build the objective function for this data
        DifferentiableFunction objective = new ObjectiveFunction<F,L>(encoding, data, indexLinearizer, sigma);
        // learn our voting weights
        double[] weights = minimizer.minimize(objective, initialWeights, 1e-4);
        // build a classifer using these weights (and the data encodings)
        return new MaximumEntropyClassifier<F, L>(weights, encoding, indexLinearizer);
      }

      private double[] buildInitialWeights(IndexLinearizer indexLinearizer) {
        return DoubleArrays.constantArray(0.0, indexLinearizer.getNumLinearIndexes());
      }

      private IndexLinearizer buildIndexLinearizer(Encoding<F, L> encoding) {
        return new IndexLinearizer(encoding.getNumFeatures(), encoding.getNumLabels());
      }

      private Encoding<F, L> buildEncoding(List<LabeledDatum<F, L>> data) {
        Index<F> featureIndex = new Index<F>();
        Index<L> labelIndex = new Index<L>();
        for (LabeledDatum<F, L> labeledDatum : data) {
          labelIndex.add(labeledDatum.getLabel());
          for (F feature : labeledDatum.getFeatures()) {
            featureIndex.add(feature);
          }
        }
        return new Encoding<F, L>(featureIndex, labelIndex);
      }

      private EncodedDatum[] encodeData(List<LabeledDatum<F, L>> data, Encoding<F, L> encoding) {
        EncodedDatum[] encodedData = new EncodedDatum[data.size()];
        for (int i = 0; i < data.size(); i++) {
          LabeledDatum<F, L> labeledDatum = data.get(i);
          encodedData[i] = EncodedDatum.encodeLabeledDatum(labeledDatum, encoding);
        }
        return encodedData;
      }

      /**
       * Sigma controls the variance on the prior / penalty term.  1.0 is a
       * reasonable value for large problems, bigger sigma means LESS smoothing.
       * Zero sigma is a special indicator that no smoothing is to be done.
       * <p/>
       * Iterations determines the maximum number of iterations the optimization
       * code can take before stopping.
       */
      public Factory(double sigma, int iterations) {
        this.sigma = sigma;
        this.iterations = iterations;
      }
    }

    /**
     * This is the MaximumEntropy objective function: the (negative) log
     * conditional likelihood of the training data, possibly with a penalty for
     * large weights.  Note that this objective get MINIMIZED so it's the
     * negative of the objective we normally think of.
     */
    public static class ObjectiveFunction<F,L> implements DifferentiableFunction {
      IndexLinearizer indexLinearizer;
      Encoding<F,L> encoding;
      EncodedDatum[] data;

      double sigma;

      double lastValue;
      double[] lastDerivative;
      double[] lastX;

      public int dimension() {
        return indexLinearizer.getNumLinearIndexes();
      }

      public double valueAt(double[] x) {
        ensureCache(x);
        return lastValue;
      }

      public double[] derivativeAt(double[] x) {
        ensureCache(x);
        return lastDerivative;
      }

      private void ensureCache(double[] x) {
        if (requiresUpdate(lastX, x)) {
          Pair<Double, double[]> currentValueAndDerivative = calculate(x);
          lastValue = currentValueAndDerivative.getFirst();
          lastDerivative = currentValueAndDerivative.getSecond();
          lastX = x;
        }
      }

      private boolean requiresUpdate(double[] lastX, double[] x) {
        if (lastX == null) return true;
        for (int i = 0; i < x.length; i++) {
          if (lastX[i] != x[i])
            return true;
        }
        return false;
      }

        /**
         * The important part of the classifier learning process!  This method
         * determines, for the given weight vector x, what the (negative) log
         * conditional likelihood of the data is, as well as the derivatives of
         * that likelihood wrt each weight parameter.
         */
        private Pair<Double, double[]> calculate(double[] x) {
            double objective = 0.0;
            int dim = dimension();
            double[] derivatives = new double[dim];
            double[] leftTerm = new double[dim];
            double[] rightTerm = new double[dim];

            // DONE: compute the objective
            // the objective function is the negative conditional log likelihood of the training data (C,D)
            // in other words, the sum of all log probabilities of a proposed class c
            // objective = -1 * (sum_{c,d} [log(P(c|d, x))])
            // c = class, d = datum, x = proposed weight vector
            // i corresponds to the indexes generated by the IndexLinearizer


            
            for (EncodedDatum datum : data) {
                double[] logProbabilities = getLogProbabilities(datum, x, encoding, indexLinearizer);
                objective -= logProbabilities[datum.getLabelIndex()];

                // TODO: compute the derivatives
                //derivative(x_i) = -1 * (sum_{c,d}[f_i(c,d] - sum_{c,d}[sum_{c'}[p(c'|d, x)f_i(c',d)]])
                //left sum shows how many times i occurs in the data for each feature i
                //right sum is the expectation of the same quantity using the label distributions the model predicts

                int numActiveFeatures = datum.getNumActiveFeatures();
                int numLabels = encoding.getNumLabels();

                for (int i=0; i < numActiveFeatures; i++) {
                    double count = datum.getFeatureCount(i);
                    int linearIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(i), datum.getLabelIndex());
                    derivatives[linearIndex] -= count;

                    for (int j=0; j < numLabels; j++) {
                        linearIndex = indexLinearizer.getLinearIndex(datum.getFeatureIndex(i), j);
                        double labelProb = Math.exp(logProbabilities[j]);
                        derivatives[linearIndex] += labelProb;
                    }
                }
            }

//            for (int i = 0; i < dim; i++) {
//                derivatives[i] = -1 * (leftTerm[i] - rightTerm[i]);
//            }

            // dummy code; just says that the objective is 42 and the derivatives are flat
//        objective = 42;
//        for (int i = 0; i < derivatives.length; i++) {
//          derivatives[i] = 0.0;
//        }
            // end dummy code

            // penalties

            // TODO: penalize large weights
            // modify objective function according to the following
            // G(x) = F(x) + sum_i[x_i^2/2*sigma^2]
            // see p 205 J&M

            // TODO: derivatives change correspondingly
            //derivative(G(x)) = derivative(F(x) + x_i/sigma^2


            double square = sigma * sigma;
            for (int i = 0; i < dim; i++) {
                objective += x[i] * x[i] / (2 * square);
                derivatives[i] += x[i] / square;
            }


            return new Pair<Double, double[]>(objective, derivatives);
        }

      public ObjectiveFunction(Encoding<F,L> encoding, EncodedDatum[] data, IndexLinearizer indexLinearizer, double sigma) {
        this.indexLinearizer = indexLinearizer;
        this.encoding = encoding;
        this.data = data;
        this.sigma = sigma;
      }
    }

    /**
     * EncodedDatums are sparse representations of (labeled) feature count
     * vectors for a given data point.  Features and labels are encoded
     * as numbers using the Encoding class.
     *
     * Use getNumActiveFeatures() to see how many features have non-zero 
     * count in a datum.  Then, for each non-zero feature use 
     * getFeatureIndex() to get the index of the feature
     * (number used to encode the feature in Encoding) and
     * getFeatureCount() to retrieve the count of the feature.
     * Use getLabelIndex() to get the label's index 
     * (number used to encode the label in Encoding).
     */
    public static class EncodedDatum {

      public static <F,L> EncodedDatum encodeDatum(Datum<F> datum, Encoding<F, L> encoding) {
        Collection<F> features = datum.getFeatures();
        Counter<F> featureCounter = new Counter<F>();
        for (F feature : features) {
          if (encoding.getFeatureIndex(feature) < 0)
            continue;
          featureCounter.incrementCount(feature, 1.0);
        }
        int numActiveFeatures = featureCounter.keySet().size();
        int[] featureIndexes = new int[numActiveFeatures];
        double[] featureCounts = new double[featureCounter.keySet().size()];
        int i = 0;
        for (F feature : featureCounter.keySet()) {
          int index = encoding.getFeatureIndex(feature);
          double count = featureCounter.getCount(feature);
          featureIndexes[i] = index;
          featureCounts[i] = count;
          i++;
        }
        EncodedDatum encodedDatum = new EncodedDatum(-1, featureIndexes, featureCounts);
        return encodedDatum;
      }

      public static <F,L> EncodedDatum encodeLabeledDatum(LabeledDatum<F, L> labeledDatum, Encoding<F, L> encoding) {
        EncodedDatum encodedDatum = encodeDatum(labeledDatum, encoding);
        encodedDatum.labelIndex = encoding.getLabelIndex(labeledDatum.getLabel());
        return encodedDatum;
      }

      int labelIndex;
      int[] featureIndexes;
      double[] featureCounts;

      /**
       * Returns the index of the label (for lookup in Encoding)
       * associated with this datum
       */
      public int getLabelIndex() {
        return labelIndex;
      }

      /**
       * Returns number of active features (features with non-zero counts)
       * for this datum
       */
      public int getNumActiveFeatures() {
        return featureCounts.length;
      }

      /**
       * Returns feature index (for lookup in Encoding) of the num'th 
       *  active feature
       **/
      public int getFeatureIndex(int num) {
        return featureIndexes[num];
      }

      /**
       * Returns the count for the num'th active feature 
       *  (number of times the feature occurred in this datum)
       **/
      public double getFeatureCount(int num) {
        return featureCounts[num];
      }

      public EncodedDatum(int labelIndex, int[] featureIndexes, double[] featureCounts) {
        this.labelIndex = labelIndex;
        this.featureIndexes = featureIndexes;
        this.featureCounts = featureCounts;
      }
    }

    /**
     * The Encoding maintains correspondences between the various representions
     * of the data, labels, and features.  The external representations of
     * labels and features are object-based.  The functions getLabelIndex() and
     * getFeatureIndex() can be used to translate those objects to integer
     * representations: numbers between 0 and getNumLabels() or getNumFeatures()
     * (exclusive).  The inverses of this map are the getLabel() and
     * getFeature() functions.
     */
    public static class Encoding <F,L> {
      Index<F> featureIndex;
      Index<L> labelIndex;

      public int getNumFeatures() {
        return featureIndex.size();
      }

      public int getFeatureIndex(F feature) {
        return featureIndex.indexOf(feature);
      }

      public F getFeature(int idx) {
        return featureIndex.get(idx);
      }

      public int getNumLabels() {
        return labelIndex.size();
      }

      public int getLabelIndex(L label) {
        return labelIndex.indexOf(label);
      }

      public L getLabel(int idx) {
        return labelIndex.get(idx);
      }

      public Encoding(Index<F> featureIndex, Index<L> labelIndex) {
        this.featureIndex = featureIndex;
        this.labelIndex = labelIndex;
      }
    }

    /**
     * The IndexLinearizer maintains the linearization of the two-dimensional
     * features-by-labels pair space.  This is because, while we might think
     * about lambdas and derivatives as being indexed by a feature-label pair,
     * the optimization code expects one long vector for lambdas and
     * derivatives.  To go from a pair featureIndex, labelIndex to a single
     * pairIndex, use getLinearIndex().
     */
    public static class IndexLinearizer {
      int numFeatures;
      int numLabels;

      public int getNumLinearIndexes() {
        return numFeatures * numLabels;
      }

      public int getLinearIndex(int featureIndex, int labelIndex) {
        return labelIndex + featureIndex * numLabels;
      }

      public int getFeatureIndex(int linearIndex) {
        return linearIndex / numLabels;
      }

      public int getLabelIndex(int linearIndex) {
        return linearIndex % numLabels;
      }

      public IndexLinearizer(int numFeatures, int numLabels) {
        this.numFeatures = numFeatures;
        this.numLabels = numLabels;
      }
    }


    private double[] weights;
    private Encoding<F, L> encoding;
    private IndexLinearizer indexLinearizer;

    /**
     * Calculate the log probabilities of each class, for the given datum
     * (feature bundle).  Note that the weighted votes (refered to as
     * activations) are *almost* log probabilities, but need to be normalized.
     */
    private static <F,L> double[] getLogProbabilities(EncodedDatum datum, double[] weights, Encoding<F, L> encoding, IndexLinearizer indexLinearizer) {
      // TODO: neaten this section up (comments)
      // DONE: apply the classifier to this feature vector
        // this method takes a datum and produces the (log) distribution, according to the model, over the possible labels for the datum
        //1) get log class probabilities  2) normalize (Z factor) to make the exponential into a true probability
        // see page 201 J&M

      // DONE: compute log label probabilities
                //c = label, N = number of features, w = weight, f_i = current feature
        //c = , N = datum.getNumActiveFeatures(), w = weights[], f_i =


      // DONE: normalize log probabilities
        //apply normalization factor z
        //z = sum_c'[exp(sum_{i=0}^N [w_c'i * f_i(c',x])]
        //see logAdd(x, y) and logAdd(x[]) in math.SloppyMath


        int numLabels = encoding.getNumLabels();
        double[] logProbs = new double[numLabels];

        double denominator = 0;
        for (int i=0; i<numLabels; i++) {
            double exponent = 0;
            for (int j=0; j<datum.getNumActiveFeatures(); j++) {
                double weight = weights[indexLinearizer.getLinearIndex(datum.getFeatureIndex(j), i)];
                exponent += weight;
            }
            logProbs[i] = exponent;
            denominator += Math.exp(logProbs[i]);
        }

        denominator = Math.log(denominator);

        for (int i=0; i<numLabels; i++) {
            logProbs[i] = logProbs[i] - denominator;
        }

        //P(label|x) = 1/z(exp(sum_{i=0}^N[w_ci * f_i(c,x)]))
        //double logLabelProb = 1/z*Math.exp(sum);

      return logProbs;
    }

    public Counter<L> getProbabilities(Datum<F> datum) {
      EncodedDatum encodedDatum = EncodedDatum.encodeDatum(datum, encoding);
      double[] logProbabilities = getLogProbabilities(encodedDatum, weights, encoding, indexLinearizer);
      return logProbabiltyArrayToProbabiltyCounter(logProbabilities);
    }

    public Counter<L> getLogProbabilities(Datum<F> datum) {
      EncodedDatum encodedDatum = EncodedDatum.encodeDatum(datum, encoding);
      double[] logProbabilities = getLogProbabilities(encodedDatum, weights, encoding, indexLinearizer);
      Counter<L> probabiltyCounter = new Counter<L>();
      for (int labelIndex = 0; labelIndex < logProbabilities.length; labelIndex++) {
        double logProbability = logProbabilities[labelIndex];
        L label = encoding.getLabel(labelIndex);
        probabiltyCounter.setCount(label, logProbability);
      }
      return probabiltyCounter;
    }

    
    private Counter<L> logProbabiltyArrayToProbabiltyCounter(double[] logProbabilities) {
      Counter<L> probabiltyCounter = new Counter<L>();
      for (int labelIndex = 0; labelIndex < logProbabilities.length; labelIndex++) {
        double logProbability = logProbabilities[labelIndex];
        double probability = Math.exp(logProbability);
        L label = encoding.getLabel(labelIndex);
        probabiltyCounter.setCount(label, probability);
      }
      return probabiltyCounter;
    }

    public L getLabel(Datum<F> datum) {
      return getProbabilities(datum).argMax();
    }
    
    public MaximumEntropyClassifier(double[] weights, Encoding<F, L> encoding, IndexLinearizer indexLinearizer) {
      this.weights = weights;
      this.encoding = encoding;
      this.indexLinearizer = indexLinearizer;
    }
  }

  /**
   * This method takes a sentence, a position in that sentence, and the previous
   * label, and produces a list of features for that word.  You should use the previous
   * labels in (at least some of) your features, or else there won't really be a
   * sequence component to the model.
  */
  private static List<String> extractFeatures(List<String> sentence, int position, String prevLabel) {
    List<String> features = new ArrayList<String>();

    // add feature for the label:
    String word = sentence.get(position);
    features.add("WORD-" + word);
    
    // add feature for previous label:
    features.add("PREV_LABEL-" + prevLabel);


    if (position > 0){
      features.add("PREV_WORD-" + sentence.get(position - 1));
    }
    
    if (position > 1){
      features.add("PREV_PREV_WORD-" + sentence.get(position - 2));
    }
    
    if (position < sentence.size() - 1){
    	features.add("NEXT_WORD-" + sentence.get(position +1));
    }
    
    if (position < sentence.size() - 2){
    	features.add("NEXT_NEXT_WORD-" + sentence.get(position+2));
    }
    // does word contain a number?
    // does word contain a hyphen?
    // does the word have spelled-out greek letters?
    // current word and previous label?

  //some suggestions: orthographic features (whether word starts with capital, contains a number, hyphens); regex to see if the word has spelled out greek letters; feature conjunctions such as capitalization and previous label
      //think of features that will get activated with some frequency!
      //features.add() will add the feature to the feature list

    // TODO: more features
    // TODO: more features
    // TODO: more features
    
    return features;
  }

  private static List<LabeledDatum<String, String>> transformData(List<Pair<List<String>,List<String>>> sentences) {
    List<LabeledDatum<String, String>> data = new ArrayList<LabeledDatum<String, String>>();
    for (Pair<List<String>,List<String>> sentenceAndLabels : sentences) {
      List<String> sentence = sentenceAndLabels.getFirst();
      List<String> labels = sentenceAndLabels.getSecond();
      String prevLabel = "O";
      for (int i = 0; i < sentence.size(); i++) {
        String label = labels.get(i);
        List<String> features = extractFeatures(sentence, i, prevLabel);
        LabeledDatum<String, String> datum = new BasicLabeledDatum<String, String>(label, features);
        data.add(datum);
        prevLabel = label;
      }
    }
    return data;
  }

  static List<Pair<List<String>,List<String>>> loadData(String fileName) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(fileName));
    List<Pair<List<String>,List<String>>> sentences = new ArrayList<Pair<List<String>,List<String>>>();
    List<String> sentence = new ArrayList<String>();
    List<String> labels = new ArrayList<String>();
    while (reader.ready()) {
      String line = reader.readLine();
      if (line.trim().length() == 0) {
        sentences.add(new Pair<List<String>,List<String>>(sentence, labels));
        sentence = new ArrayList<String>();
        labels = new ArrayList<String>();
        continue;
      }
      String[] parts = line.split("\t");
      sentence.add(parts[0]);
      labels.add(parts[1]);
    }
    if (!sentence.isEmpty()) {
      sentences.add(new Pair<List<String>,List<String>>(sentence, labels));
    }
    return sentences;
  }

  public static List<Pair<String,String>> labelAndChunkSequence(ProbabilisticClassifier<String, String> classifier, List<String> sentence) {
    List<String> labels = labelSequence(classifier, sentence);

    List<Pair<String,String>> chunkedSentence = new ArrayList<Pair<String,String>>();

    String curLabel = "O";
    String curWord = "";
    
    for (int i = 0; i < sentence.size(); i++) {
      String label = labels.get(i);
      String word = sentence.get(i);
      if (label.equals("O")) {
        if (curWord.length() > 0) {
          chunkedSentence.add(new Pair<String,String>(curWord, curLabel));
        }

        chunkedSentence.add(new Pair<String,String>(word, label));
        curLabel = "";
        curWord = "";          

      } else {
        if (label.startsWith("B-") || label.startsWith("I-")) {
          label = label.substring(2,label.length());
        }
        if (label.equals(curLabel)) {
          if (curWord.length() > 0) { curWord += "_"; }
          curWord += word;
        } else {
          if (curWord.length() > 0) {
            chunkedSentence.add(new Pair<String,String>(curWord, curLabel));
          }
          curLabel = label;
          curWord = word;                    
        }
      }
    }
    return chunkedSentence;
  }
  
  public static List<String> labelSequence(ProbabilisticClassifier<String, String> classifier, List<String> sentence) {
    // do viterbi
    Counter<String>[] viterbiScores = new Counter[sentence.size()];
    Map<String,String>[] bestPrevLabel = new HashMap[sentence.size()-1];
    List<String> features = extractFeatures(sentence, 0, "O");
    Datum<String> datum = new BasicLabeledDatum<String,String>("NULL_LABEL", features);
    viterbiScores[0] = classifier.getLogProbabilities(datum);
    for (int position = 1; position < sentence.size(); position++) {
      for (String prevLabel : viterbiScores[position-1].keySet()) {
        features = extractFeatures(sentence, position, prevLabel);
        datum = new BasicLabeledDatum<String,String>("NULL_LABEL", features);
        Counter<String> scores = classifier.getLogProbabilities(datum);
        double prevScore = viterbiScores[position-1].getCount(prevLabel);
        if (viterbiScores[position] == null) {
          viterbiScores[position] = new Counter<String>();
          bestPrevLabel[position-1] = new HashMap<String,String>();
          for (String label : scores.keySet()) {              
            viterbiScores[position].setCount(label, prevScore + scores.getCount(label));
            bestPrevLabel[position-1].put(label, prevLabel);
          }
        } else {
          for (String label : scores.keySet()) {
            double score = prevScore + scores.getCount(label);
            if (viterbiScores[position].getCount(label) < score) {
              viterbiScores[position].setCount(label, score);
              bestPrevLabel[position-1].put(label, prevLabel);
            }
          }
        }
      }
    }
    String[] labels = new String[sentence.size()];
    labels[labels.length-1] = viterbiScores[viterbiScores.length-1].argMax();
    for (int position = labels.length-2; position >= 0; position--) {
      labels[position] = bestPrevLabel[position].get(labels[position+1]);
    }
    return Arrays.asList(labels);
  }
  
  static <F> void testClassifier(ProbabilisticClassifier<F, String> classifier, List<LabeledDatum<F, String>> testData) {
    double numCorrect = 0.0;
    double numTotal = 0.0;
    for (LabeledDatum<F, String> testDatum : testData) {
      String label = classifier.getLabel(testDatum);
      if (label.equals(testDatum.getLabel()))
        numCorrect += 1.0;
      numTotal += 1.0;
    }
    double accuracy = numCorrect / numTotal;
    System.out.println("Accuracy: " + accuracy);
  }

  private static void miniTest() {
    LabeledDatum<String, String> datum1 = new BasicLabeledDatum<String, String>("cat", Arrays.asList(new String[]{"fuzzy", "claws", "small"}));
    LabeledDatum<String, String> datum2 = new BasicLabeledDatum<String, String>("bear", Arrays.asList(new String[]{"fuzzy", "claws", "big"}));
    LabeledDatum<String, String> datum3 = new BasicLabeledDatum<String, String>("cat", Arrays.asList(new String[]{"claws", "medium"}));
    LabeledDatum<String, String> datum4 = new BasicLabeledDatum<String, String>("cat", Arrays.asList(new String[]{"claws", "small"}));
    List<LabeledDatum<String, String>> trainingData = new ArrayList<LabeledDatum<String, String>>();
    trainingData.add(datum1);
    trainingData.add(datum2);
    trainingData.add(datum3);
    List<LabeledDatum<String, String>> testData = new ArrayList<LabeledDatum<String, String>>();
    testData.add(datum4);
    MaximumEntropyClassifier.Factory<String, String> maximumEntropyClassifierFactory = new MaximumEntropyClassifier.Factory<String, String>(1.0, 20);
    ProbabilisticClassifier<String, String> maximumEntropyClassifier = maximumEntropyClassifierFactory.trainClassifier(trainingData);
    System.out.println("Probabilities on test instance: " + maximumEntropyClassifier.getProbabilities(datum4));
    testClassifier(maximumEntropyClassifier, testData);
    System.exit(0);
  }

  public static ProbabilisticClassifier<String,String> getClassifier(String trainFile) throws IOException {
    List<LabeledDatum<String, String>> trainingData = transformData(loadData(trainFile));
    MaximumEntropyClassifier.Factory<String, String> maximumEntropyClassifierFactory = new MaximumEntropyClassifier.Factory<String, String>(1.0, 40);
    ProbabilisticClassifier<String, String> maximumEntropyClassifier = maximumEntropyClassifierFactory.trainClassifier(trainingData);
    return maximumEntropyClassifier;
  }
  
  public static void main(String[] args) throws IOException {
    if (args.length > 0 && args[0].equalsIgnoreCase("-mini")) {
      miniTest();
      return;
    }

    Map<String,String> props = CommandLineUtils.simpleCommandLineParser(args);

    String trainFile = args[0]+".train";
    String testFile = args[0]+".test";

    ProbabilisticClassifier<String, String> maximumEntropyClassifier = getClassifier(trainFile);

    List<Pair<List<String>,List<String>>> testData = loadData(testFile);    
    
    for (Pair<List<String>,List<String>> sentenceAndLabels : testData) {
      List<String> sentence = sentenceAndLabels.getFirst();
      List<String> goldLabels = sentenceAndLabels.getSecond();
      List<String> guessedLabels = labelSequence(maximumEntropyClassifier, sentence);
      System.out.println();
      for (int i = 0; i < sentence.size(); i++) {
        System.out.print(sentence.get(i)+"\t");
        System.out.print(goldLabels.get(i)+"\t");
        System.out.println(guessedLabels.get(i));
      }
    }
  }
}
